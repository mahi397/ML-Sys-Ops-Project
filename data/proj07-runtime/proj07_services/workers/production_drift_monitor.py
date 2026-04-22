#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from proj07_services.common.feedback_common import (
    fetch_source_utterances,
    get_conn,
    write_json,
    insert_dataset_quality_report,
    build_model_utterances_by_meeting,
)
from proj07_services.common.task_service_common import build_logger, env_float, env_int
from proj07_services.quality.drift_control import (
    DriftGateConfig,
    build_reference_profile,
    compare_feature_columns_to_reference,
    extract_live_feature_columns,
)
from proj07_services.retraining.runtime import latest_reference_profile


APP_NAME = "production_drift_monitor"


@dataclass(frozen=True)
class ProductionDriftMonitorConfig:
    poll_interval_seconds: float = env_float(
        "PRODUCTION_DRIFT_MONITOR_POLL_INTERVAL_SECONDS",
        900.0,
    )
    log_dir: Path = Path(
        os.getenv(
            "PRODUCTION_DRIFT_MONITOR_LOG_DIR",
            "/mnt/block/ingest_logs/production_drift_monitor",
        )
    )
    report_root: Path = Path(
        os.getenv(
            "PRODUCTION_DRIFT_MONITOR_REPORT_ROOT",
            "/mnt/block/staging/feedback_loop/production_drift_reports",
        )
    )
    lookback_hours: int = env_int("PRODUCTION_DRIFT_MONITOR_LOOKBACK_HOURS", 24)
    min_valid_meetings: int = env_int("PRODUCTION_DRIFT_MONITOR_MIN_VALID_MEETINGS", 5)
    dataset_name: str = os.getenv("PRODUCTION_DRIFT_MONITOR_DATASET_NAME", "roberta_stage1").strip()
    dataset_root: Path = Path(
        os.getenv(
            "PRODUCTION_DRIFT_MONITOR_DATASET_ROOT",
            os.getenv("RETRAINING_DATASET_ROOT", "/mnt/block/roberta_stage1"),
        )
    )
    window_size: int = env_int(
        "PRODUCTION_DRIFT_MONITOR_WINDOW_SIZE",
        env_int("RETRAINING_DATASET_WINDOW_SIZE", 7),
    )
    transition_index: int = env_int(
        "PRODUCTION_DRIFT_MONITOR_TRANSITION_INDEX",
        env_int("RETRAINING_DATASET_TRANSITION_INDEX", 3),
    )
    min_utterance_chars: int = env_int(
        "PRODUCTION_DRIFT_MONITOR_MIN_UTTERANCE_CHARS",
        env_int("RETRAINING_DATASET_MIN_UTTERANCE_CHARS", 20),
    )
    max_words_per_utterance: int = env_int(
        "PRODUCTION_DRIFT_MONITOR_MAX_WORDS_PER_UTTERANCE",
        env_int("RETRAINING_DATASET_MAX_WORDS_PER_UTTERANCE", 50),
    )
    psi_threshold: float = env_float(
        "PRODUCTION_DRIFT_MONITOR_PSI_THRESHOLD",
        env_float("RETRAINING_DATASET_QUALITY_PSI_THRESHOLD", 0.2),
    )
    max_drift_share: float = env_float(
        "PRODUCTION_DRIFT_MONITOR_MAX_DRIFT_SHARE",
        env_float("RETRAINING_DATASET_QUALITY_MAX_DRIFT_SHARE", 0.35),
    )
    min_feature_samples: int = env_int(
        "PRODUCTION_DRIFT_MONITOR_MIN_FEATURE_SAMPLES",
        env_int("RETRAINING_DATASET_QUALITY_MIN_FEATURE_SAMPLES", 25),
    )
    numeric_bin_count: int = env_int(
        "PRODUCTION_DRIFT_MONITOR_NUMERIC_BIN_COUNT",
        env_int("RETRAINING_DATASET_QUALITY_NUMERIC_BIN_COUNT", 10),
    )


def validate_config(config: ProductionDriftMonitorConfig) -> None:
    if config.poll_interval_seconds <= 0:
        raise ValueError("PRODUCTION_DRIFT_MONITOR_POLL_INTERVAL_SECONDS must be > 0")
    if config.lookback_hours <= 0:
        raise ValueError("PRODUCTION_DRIFT_MONITOR_LOOKBACK_HOURS must be > 0")
    if config.min_valid_meetings <= 0:
        raise ValueError("PRODUCTION_DRIFT_MONITOR_MIN_VALID_MEETINGS must be > 0")
    if config.window_size <= 1:
        raise ValueError("PRODUCTION_DRIFT_MONITOR_WINDOW_SIZE must be > 1")
    if not (0 <= config.transition_index < config.window_size - 1):
        raise ValueError("PRODUCTION_DRIFT_MONITOR_TRANSITION_INDEX must be between 0 and window_size - 2")
    if config.min_utterance_chars <= 0:
        raise ValueError("PRODUCTION_DRIFT_MONITOR_MIN_UTTERANCE_CHARS must be > 0")
    if config.max_words_per_utterance <= 0:
        raise ValueError("PRODUCTION_DRIFT_MONITOR_MAX_WORDS_PER_UTTERANCE must be > 0")
    if config.psi_threshold <= 0:
        raise ValueError("PRODUCTION_DRIFT_MONITOR_PSI_THRESHOLD must be > 0")
    if config.max_drift_share < 0:
        raise ValueError("PRODUCTION_DRIFT_MONITOR_MAX_DRIFT_SHARE must be >= 0")
    if config.min_feature_samples <= 0:
        raise ValueError("PRODUCTION_DRIFT_MONITOR_MIN_FEATURE_SAMPLES must be > 0")
    if config.numeric_bin_count < 2:
        raise ValueError("PRODUCTION_DRIFT_MONITOR_NUMERIC_BIN_COUNT must be >= 2")


def fetch_recent_valid_meeting_ids(conn, *, since: datetime) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT meeting_id
            FROM meetings
            WHERE source_type = 'jitsi'
              AND is_valid = TRUE
              AND ended_at >= %s
            ORDER BY ended_at DESC
            """,
            (since,),
        )
        rows = cur.fetchall()
    return [str(row["meeting_id"]) for row in rows]


def load_current_reference_profile(config: ProductionDriftMonitorConfig) -> tuple[int | None, dict | None]:
    return latest_reference_profile(config.dataset_root)


def monitor_gate_config(config: ProductionDriftMonitorConfig) -> DriftGateConfig:
    return DriftGateConfig(
        psi_threshold=config.psi_threshold,
        max_drift_share=config.max_drift_share,
        min_feature_samples=config.min_feature_samples,
        numeric_bin_count=config.numeric_bin_count,
    )


@dataclass
class ProductionDriftMonitor:
    config: ProductionDriftMonitorConfig
    logger: object

    def run_cycle(self) -> bool:
        reference_version, reference_profile = load_current_reference_profile(self.config)
        if reference_profile is None:
            self.logger.info(
                "Skipping production drift check because no usable Stage 1 reference profile was found"
            )
            return False

        until = datetime.now(timezone.utc)
        since = until - timedelta(hours=self.config.lookback_hours)

        conn = get_conn()
        try:
            meeting_ids = fetch_recent_valid_meeting_ids(conn, since=since)
            if len(meeting_ids) < self.config.min_valid_meetings:
                report = {
                    "status": "skipped",
                    "reason": "insufficient_valid_meetings",
                    "current_metadata": {
                        "meeting_count": len(meeting_ids),
                        "row_count": 0,
                    },
                }
                self.persist_report(
                    conn=conn,
                    report=report,
                    since=since,
                    until=until,
                    reference_version=reference_version,
                )
                self.logger.info(
                    "Skipping production drift check | recent_valid_meetings=%s minimum=%s",
                    len(meeting_ids),
                    self.config.min_valid_meetings,
                )
                return False

            source_rows = fetch_source_utterances(conn, meeting_ids)
            model_utterances = build_model_utterances_by_meeting(
                source_rows=source_rows,
                max_words=self.config.max_words_per_utterance,
                min_chars=self.config.min_utterance_chars,
            )
            live_columns, live_metadata = extract_live_feature_columns(
                model_utterances,
                window_size=self.config.window_size,
                transition_index=self.config.transition_index,
            )
            live_profile = build_reference_profile(
                live_columns,
                metadata=live_metadata,
                bin_count=self.config.numeric_bin_count,
            )
            report = compare_feature_columns_to_reference(
                reference_profile=reference_profile,
                current_feature_columns=live_columns,
                current_metadata=live_metadata,
                config=monitor_gate_config(self.config),
            )
            report["live_profile"] = live_profile

            self.persist_report(
                conn=conn,
                report=report,
                since=since,
                until=until,
                reference_version=reference_version,
            )
            if report["status"] == "failed":
                self.logger.warning(
                    "Production drift gate FAILED | reference=v%s share_drifted=%.3f meetings=%s rows=%s",
                    reference_version,
                    report["share_drifted_features"],
                    live_metadata.get("meeting_count", 0),
                    live_metadata.get("row_count", 0),
                )
            else:
                self.logger.info(
                    "Production drift gate %s | reference=v%s share_drifted=%.3f meetings=%s rows=%s",
                    report["status"].upper(),
                    reference_version,
                    report["share_drifted_features"],
                    live_metadata.get("meeting_count", 0),
                    live_metadata.get("row_count", 0),
                )
            return report["status"] == "failed"
        finally:
            conn.close()

    def persist_report(
        self,
        *,
        conn,
        report: dict,
        since: datetime,
        until: datetime,
        reference_version: int | None,
    ) -> None:
        stamp = until.strftime("%Y%m%dT%H%M%SZ")
        report_dir = self.config.report_root / stamp
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "quality_report.json"
        write_json(report_path, report)

        insert_dataset_quality_report(
            conn,
            dataset_name=self.config.dataset_name,
            report_scope="production_live",
            report_status=report["status"],
            dataset_version=None,
            reference_dataset_name=self.config.dataset_name,
            reference_dataset_version=None if reference_version is None else str(reference_version),
            report_path=str(report_path.resolve()),
            share_drifted_features=report.get("share_drifted_features"),
            drifted_feature_count=report.get("drifted_feature_count"),
            total_feature_count=report.get("total_feature_count"),
            window_started_at=since.isoformat(),
            window_ended_at=until.isoformat(),
            details_json=report,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one monitoring cycle and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProductionDriftMonitorConfig()
    validate_config(config)
    logger = build_logger(APP_NAME, config.log_dir)
    monitor = ProductionDriftMonitor(config=config, logger=logger)

    logger.info("Starting %s", APP_NAME)
    logger.info(
        "Config | poll_interval=%ss lookback_hours=%s min_valid_meetings=%s dataset_root=%s psi_threshold=%s max_drift_share=%s",
        config.poll_interval_seconds,
        config.lookback_hours,
        config.min_valid_meetings,
        config.dataset_root,
        config.psi_threshold,
        config.max_drift_share,
    )

    if args.once:
        monitor.run_cycle()
        return

    while True:
        try:
            monitor.run_cycle()
        except KeyboardInterrupt:
            logger.info("Stopping %s", APP_NAME)
            return
        except Exception:
            logger.exception("Production drift monitor loop failed; retrying after sleep")
        time.sleep(config.poll_interval_seconds)


if __name__ == "__main__":
    main()
