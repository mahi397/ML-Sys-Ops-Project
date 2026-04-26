#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from proj07_services.common.feedback_common import get_conn
from proj07_services.common.task_service_common import build_logger, env_float, env_int
from proj07_services.retraining.runtime import (
    RetrainingBuildConfig,
    build_retraining_snapshot,
    build_stage1_feedback_pool,
    collect_retraining_metrics,
    fetch_candidate_meeting_ids,
)


APP_NAME = "retraining_dataset_service"


def default_build_config() -> RetrainingBuildConfig:
    local_tmp_root = Path(
        os.getenv(
            "RETRAINING_LOCAL_TMP_ROOT",
            os.getenv("LOCAL_TMP_ROOT", "/mnt/block/staging/feedback_loop"),
        )
    )
    return RetrainingBuildConfig(
        dataset_name=os.getenv("RETRAINING_DATASET_NAME", "roberta_stage1").strip(),
        feedback_pool_root=Path(
            os.getenv(
                "RETRAINING_FEEDBACK_POOL_ROOT",
                str(local_tmp_root / "datasets" / "roberta_stage1_feedback_pool"),
            )
        ),
        dataset_root=Path(
            os.getenv(
                "RETRAINING_DATASET_ROOT",
                os.getenv("DATASET_ROOT", "/mnt/block/roberta_stage1"),
            )
        ),
        feedback_pool_object_prefix=os.getenv(
            "RETRAINING_FEEDBACK_POOL_PREFIX",
            os.getenv("STAGE1_FEEDBACK_POOL_PREFIX", "datasets/roberta_stage1_feedback_pool"),
        ).strip(),
        dataset_object_prefix=os.getenv(
            "RETRAINING_DATASET_OBJECT_PREFIX",
            os.getenv("FINAL_DATASET_OBJECT_PREFIX", "datasets/roberta_stage1"),
        ).strip(),
        upload_artifacts=os.getenv("RETRAINING_DATASET_UPLOAD_ARTIFACTS", "true").strip().lower()
        in {"1", "true", "yes", "on"},
        window_size=env_int(
            "RETRAINING_DATASET_WINDOW_SIZE",
            env_int("WINDOW_SIZE", env_int("STAGE1_WINDOW_SIZE", 7)),
        ),
        transition_index=env_int(
            "RETRAINING_DATASET_TRANSITION_INDEX",
            env_int("TRANSITION_INDEX", env_int("STAGE1_TRANSITION_INDEX", 3)),
        ),
        min_utterance_chars=env_int(
            "RETRAINING_DATASET_MIN_UTTERANCE_CHARS",
            env_int("MIN_UTTERANCE_CHARS", env_int("STAGE1_MIN_UTTERANCE_CHARS", 20)),
        ),
        max_words_per_utterance=env_int(
            "RETRAINING_DATASET_MAX_WORDS_PER_UTTERANCE",
            env_int("MAX_WORDS_PER_UTTERANCE", env_int("STAGE1_MAX_WORDS_PER_UTTERANCE", 50)),
        ),
        quality_psi_threshold=env_float(
            "RETRAINING_DATASET_QUALITY_PSI_THRESHOLD",
            0.2,
        ),
        quality_max_drift_share=env_float(
            "RETRAINING_DATASET_QUALITY_MAX_DRIFT_SHARE",
            0.35,
        ),
        quality_min_feature_samples=env_int(
            "RETRAINING_DATASET_QUALITY_MIN_FEATURE_SAMPLES",
            25,
        ),
        quality_numeric_bin_count=env_int(
            "RETRAINING_DATASET_QUALITY_NUMERIC_BIN_COUNT",
            10,
        ),
    )


@dataclass(frozen=True)
class RetrainingDatasetServiceConfig:
    poll_interval_seconds: float = env_float(
        "RETRAINING_DATASET_WORKER_POLL_INTERVAL_SECONDS",
        300.0,
    )
    log_dir: Path = Path(
        os.getenv(
            "RETRAINING_DATASET_WORKER_LOG_DIR",
            "/mnt/block/ingest_logs/retraining_dataset_service",
        )
    )
    valid_meeting_threshold: int = env_int(
        "RETRAINING_DATASET_VALID_MEETING_THRESHOLD",
        10,
    )
    feedback_event_threshold: int = env_int(
        "RETRAINING_DATASET_FEEDBACK_EVENT_THRESHOLD",
        30,
    )
    advisory_lock_key: int = env_int(
        "RETRAINING_DATASET_ADVISORY_LOCK_KEY",
        700507,
    )
    build_config: RetrainingBuildConfig = field(default_factory=default_build_config)


@dataclass
class RetrainingDatasetService:
    config: RetrainingDatasetServiceConfig
    logger: object
    schema_wait_logged: bool = False

    def run_cycle(self, *, force_run: bool, dry_run: bool) -> bool:
        conn = get_conn()
        lock_acquired = False
        try:
            if not self.schema_ready(conn):
                if not self.schema_wait_logged:
                    self.logger.info(
                        "Skipping retraining dataset scan until the meetings validity schema is available"
                    )
                    self.schema_wait_logged = True
                conn.rollback()
                return False

            self.schema_wait_logged = False
            lock_acquired = self.try_advisory_lock(conn)
            if not lock_acquired:
                self.logger.info("Another retraining dataset build is already in progress; skipping this cycle")
                return False

            metrics = collect_retraining_metrics(conn)
            self.logger.info(
                "Retraining metrics | valid_unversioned_meetings=%s stage1_segmented_meetings=%s structural_feedback_events=%s structural_feedback_meetings=%s thresholds=(segmented_meetings>=%s feedback_events>=%s)",
                metrics.valid_unversioned_meeting_count,
                metrics.stage1_segmented_meeting_count,
                metrics.structural_feedback_event_count,
                metrics.structural_feedback_meeting_count,
                self.config.valid_meeting_threshold,
                self.config.feedback_event_threshold,
            )

            threshold_met = (
                metrics.stage1_segmented_meeting_count >= self.config.valid_meeting_threshold
                or metrics.structural_feedback_event_count >= self.config.feedback_event_threshold
            )
            if not force_run and not threshold_met:
                return False

            candidate_meeting_ids = fetch_candidate_meeting_ids(conn)
            if not candidate_meeting_ids:
                self.logger.info(
                    "Retraining trigger %s, but no valid unversioned meetings with usable Stage 1 segments were found",
                    "forced" if force_run else "met",
                )
                return False

            if dry_run:
                self.logger.info(
                    "Dry-run only | candidate_meetings=%s first_candidates=%s",
                    len(candidate_meeting_ids),
                    ", ".join(candidate_meeting_ids[:10]),
                )
                return True

            feedback_pool = build_stage1_feedback_pool(
                conn,
                config=self.config.build_config,
                logger=self.logger,
                candidate_meetings=candidate_meeting_ids,
                metrics=metrics,
                force_publish=force_run,
            )
            if feedback_pool is None:
                self.logger.info(
                    "Retraining dataset build skipped because no eligible feedback-pool rows were produced | candidate_meetings=%s structural_feedback_events=%s structural_feedback_meetings=%s",
                    len(candidate_meeting_ids),
                    metrics.structural_feedback_event_count,
                    metrics.structural_feedback_meeting_count,
                )
                return False

            snapshot = build_retraining_snapshot(
                conn,
                config=self.config.build_config,
                logger=self.logger,
                feedback_pool=feedback_pool,
                selected_meeting_ids=feedback_pool.eligible_meeting_ids,
                force_publish=force_run,
            )
            self.logger.info(
                "Retraining dataset build completed | feedback_pool=v%s snapshot=v%s meetings=%s",
                feedback_pool.version,
                snapshot.snapshot_version,
                len(snapshot.selected_meeting_ids),
            )
            return True
        finally:
            try:
                conn.rollback()
            except Exception:
                pass
            if lock_acquired:
                self.release_advisory_lock(conn)
            conn.close()

    def schema_ready(self, conn) -> bool:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'meetings'
                          AND column_name = 'is_valid'
                    ) AS has_meeting_validity,
                    EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'meetings'
                          AND column_name = 'dataset_version'
                    ) AS has_dataset_version,
                    EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = 'public'
                          AND table_name = 'feedback_events'
                    ) AS has_feedback_events,
                    EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = 'public'
                          AND table_name = 'dataset_versions'
                    ) AS has_dataset_versions
                """
            )
            row = cur.fetchone()
        return (
            bool(row["has_meeting_validity"])
            and bool(row["has_dataset_version"])
            and bool(row["has_feedback_events"])
            and bool(row["has_dataset_versions"])
        )

    def try_advisory_lock(self, conn) -> bool:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s) AS locked", (self.config.advisory_lock_key,))
            row = cur.fetchone()
        return bool(row["locked"])

    def release_advisory_lock(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (self.config.advisory_lock_key,))


def validate_config(config: RetrainingDatasetServiceConfig) -> None:
    if config.poll_interval_seconds <= 0:
        raise ValueError("RETRAINING_DATASET_WORKER_POLL_INTERVAL_SECONDS must be > 0")
    if config.valid_meeting_threshold < 0:
        raise ValueError("RETRAINING_DATASET_VALID_MEETING_THRESHOLD must be >= 0")
    if config.feedback_event_threshold < 0:
        raise ValueError("RETRAINING_DATASET_FEEDBACK_EVENT_THRESHOLD must be >= 0")
    if config.build_config.window_size <= 0:
        raise ValueError("RETRAINING_DATASET_WINDOW_SIZE must be > 0")
    if not (0 <= config.build_config.transition_index < config.build_config.window_size):
        raise ValueError("RETRAINING_DATASET_TRANSITION_INDEX must be between 0 and window_size - 1")
    if config.build_config.min_utterance_chars <= 0:
        raise ValueError("RETRAINING_DATASET_MIN_UTTERANCE_CHARS must be > 0")
    if config.build_config.max_words_per_utterance <= 0:
        raise ValueError("RETRAINING_DATASET_MAX_WORDS_PER_UTTERANCE must be > 0")
    if config.build_config.quality_numeric_bin_count < 2:
        raise ValueError("RETRAINING_DATASET_QUALITY_NUMERIC_BIN_COUNT must be >= 2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one evaluation cycle and exit.")
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Bypass thresholds and publish a retraining dataset immediately if eligible meetings exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate thresholds and candidate meetings without building artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RetrainingDatasetServiceConfig()
    validate_config(config)
    logger = build_logger(APP_NAME, config.log_dir)
    service = RetrainingDatasetService(config=config, logger=logger)

    logger.info("Starting %s", APP_NAME)
    logger.info(
        "Config | poll_interval=%ss valid_meeting_threshold=%s feedback_event_threshold=%s upload_artifacts=%s dataset_root=%s feedback_pool_root=%s",
        config.poll_interval_seconds,
        config.valid_meeting_threshold,
        config.feedback_event_threshold,
        config.build_config.upload_artifacts,
        config.build_config.dataset_root,
        config.build_config.feedback_pool_root,
    )

    if args.once:
        service.run_cycle(force_run=args.force_run, dry_run=args.dry_run)
        return

    forced_first_cycle = args.force_run
    while True:
        try:
            built = service.run_cycle(force_run=forced_first_cycle, dry_run=args.dry_run)
            forced_first_cycle = False
            if not built:
                time.sleep(config.poll_interval_seconds)
        except KeyboardInterrupt:
            logger.info("Stopping %s", APP_NAME)
            return
        except Exception:
            logger.exception("Retraining dataset loop failed; retrying after sleep")
            time.sleep(config.poll_interval_seconds)


if __name__ == "__main__":
    main()
