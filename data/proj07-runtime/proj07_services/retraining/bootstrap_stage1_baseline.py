#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from proj07_services.common.feedback_common import (
    build_model_utterances_by_meeting,
    build_stage1_rows,
    fetch_source_utterances,
    fetch_topic_segments,
    get_conn,
    insert_dataset_quality_report,
    insert_dataset_version,
    label_counts,
    pick_stage1_examples,
    stable_split_70_15_15,
    upload_dir,
    write_json,
    write_jsonl,
)
from proj07_services.common.task_service_common import build_logger, env_flag, env_float, env_int
from proj07_services.retraining.runtime import (
    RetrainingBuildConfig,
    candidate_staging_root,
    evaluate_stage1_quality_gate,
    existing_versions,
    mark_meetings_as_consumed,
    move_tree,
    next_version,
    quality_report_allows_publish,
    quarantine_root,
    rows_for_meetings,
)


APP_NAME = "bootstrap_stage1_baseline"


@dataclass(frozen=True)
class BootstrapStage1BaselineConfig:
    dataset_name: str = os.getenv("RETRAINING_DATASET_NAME", "roberta_stage1").strip()
    dataset_root: Path = Path(
        os.getenv("RETRAINING_DATASET_ROOT", os.getenv("DATASET_ROOT", "/mnt/block/roberta_stage1"))
    )
    dataset_object_prefix: str = os.getenv(
        "RETRAINING_DATASET_OBJECT_PREFIX",
        os.getenv("FINAL_DATASET_OBJECT_PREFIX", "datasets/roberta_stage1"),
    ).strip()
    upload_artifacts: bool = env_flag("RETRAINING_DATASET_UPLOAD_ARTIFACTS", True)
    log_dir: Path = Path(
        os.getenv("BOOTSTRAP_STAGE1_BASELINE_LOG_DIR", "/mnt/block/ingest_logs/bootstrap_stage1_baseline")
    )
    window_size: int = env_int("RETRAINING_DATASET_WINDOW_SIZE", 7)
    transition_index: int = env_int("RETRAINING_DATASET_TRANSITION_INDEX", 3)
    min_utterance_chars: int = env_int("RETRAINING_DATASET_MIN_UTTERANCE_CHARS", 20)
    max_words_per_utterance: int = env_int("RETRAINING_DATASET_MAX_WORDS_PER_UTTERANCE", 50)
    quality_psi_threshold: float = env_float("RETRAINING_DATASET_QUALITY_PSI_THRESHOLD", 0.2)
    quality_max_drift_share: float = env_float("RETRAINING_DATASET_QUALITY_MAX_DRIFT_SHARE", 0.35)
    quality_min_feature_samples: int = env_int("RETRAINING_DATASET_QUALITY_MIN_FEATURE_SAMPLES", 25)
    quality_numeric_bin_count: int = env_int("RETRAINING_DATASET_QUALITY_NUMERIC_BIN_COUNT", 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Build a new AMI-backed baseline even if versioned Stage 1 snapshots already exist.",
    )
    return parser.parse_args()


def build_config() -> BootstrapStage1BaselineConfig:
    return BootstrapStage1BaselineConfig()


def retraining_build_config(config: BootstrapStage1BaselineConfig) -> RetrainingBuildConfig:
    return RetrainingBuildConfig(
        dataset_name=config.dataset_name,
        feedback_pool_root=config.dataset_root.parent / "staging_unused_feedback_pool",
        dataset_root=config.dataset_root,
        feedback_pool_object_prefix="datasets/unused_feedback_pool",
        dataset_object_prefix=config.dataset_object_prefix,
        upload_artifacts=config.upload_artifacts,
        window_size=config.window_size,
        transition_index=config.transition_index,
        min_utterance_chars=config.min_utterance_chars,
        max_words_per_utterance=config.max_words_per_utterance,
        quality_psi_threshold=config.quality_psi_threshold,
        quality_max_drift_share=config.quality_max_drift_share,
        quality_min_feature_samples=config.quality_min_feature_samples,
        quality_numeric_bin_count=config.quality_numeric_bin_count,
    )


def discover_ami_meeting_ids(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT m.meeting_id
            FROM meetings m
            JOIN topic_segments ts
              ON ts.meeting_id = m.meeting_id
            WHERE m.source_type = 'ami'
              AND ts.segment_type = 'gold'
            ORDER BY m.meeting_id
            """
        )
        rows = cur.fetchall()
    return [str(row["meeting_id"]) for row in rows]


def main() -> int:
    args = parse_args()
    config = build_config()
    logger = build_logger(APP_NAME, config.log_dir)

    if existing_versions(config.dataset_root) and not args.force:
        logger.info(
            "Skipping Stage 1 baseline bootstrap because versioned dataset snapshots already exist under %s",
            config.dataset_root,
        )
        return 0

    conn = get_conn()
    try:
        meeting_ids = discover_ami_meeting_ids(conn)
        if not meeting_ids:
            logger.info("Skipping Stage 1 baseline bootstrap because no AMI meetings with gold segments were found")
            return 0

        source_rows = fetch_source_utterances(conn, meeting_ids)
        gold_segments = fetch_topic_segments(conn, meeting_ids, "gold")
        topic_segments_by_meeting: dict[str, list[dict]] = defaultdict(list)
        for row in gold_segments:
            topic_segments_by_meeting[row["meeting_id"]].append(row)

        model_utterances_by_meeting = build_model_utterances_by_meeting(
            source_rows=source_rows,
            max_words=config.max_words_per_utterance,
            min_chars=config.min_utterance_chars,
        )
        dataset_rows = build_stage1_rows(
            model_utterances_by_meeting=model_utterances_by_meeting,
            topic_segments_by_meeting=topic_segments_by_meeting,
            window_size=config.window_size,
            transition_index=config.transition_index,
        )
        eligible_meetings = sorted({row["input"]["meeting_id"] for row in dataset_rows})
        if not dataset_rows or not eligible_meetings:
            logger.info(
                "Skipping Stage 1 baseline bootstrap because the AMI corpus did not produce eligible Stage 1 rows"
            )
            return 0

        split_assignments = stable_split_70_15_15(eligible_meetings)
        train_rows = rows_for_meetings(dataset_rows, split_assignments["train"])
        val_rows = rows_for_meetings(dataset_rows, split_assignments["val"])
        test_rows = rows_for_meetings(dataset_rows, split_assignments["test"])
        combined_rows = train_rows + val_rows + test_rows

        version = next_version(config.dataset_root)
        profile, quality_report, reference_version = evaluate_stage1_quality_gate(
            rows=combined_rows,
            include_label=True,
            reference_root=config.dataset_root,
            config=retraining_build_config(config),
        )
        manifest = {
            "dataset_name": config.dataset_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_type": "ami",
            "composition": {
                "base_source": None,
                "feedback_pool_source": None,
            },
            "packaging": {
                "type": "bootstrap_from_ami_corpus",
                "reason": "The runtime bootstrapped the first published Stage 1 dataset from AMI meetings so live drift monitoring had a baseline reference profile.",
            },
            "ongoing_version": {
                "snapshot_version": version,
                "base_version": None,
                "feedback_pool_version": None,
            },
            "splits": {
                "train": label_counts(train_rows),
                "val": label_counts(val_rows),
                "test": label_counts(test_rows),
            },
            "meetings": {
                "base_version": None,
                "feedback_pool_version": None,
                "snapshot_version": version,
                "base_roll_forward_policy": "The initial baseline snapshot was bootstrapped directly from AMI meetings with gold topic segments.",
                "new_meeting_selection_policy": "All AMI meetings with gold topic segments were deterministically assigned with the stable 70/15/15 split.",
                "new_meeting_ids": eligible_meetings,
                "train_meeting_ids": split_assignments["train"],
                "val_meeting_ids": split_assignments["val"],
                "test_meeting_ids": split_assignments["test"],
            },
            "quality_gate": {
                "status": quality_report["status"],
                "reason": quality_report.get("reason"),
                "reference_version": reference_version,
                "share_drifted_features": quality_report["share_drifted_features"],
                "drifted_feature_count": quality_report["drifted_feature_count"],
                "total_feature_count": quality_report["total_feature_count"],
            },
        }

        staging_root = candidate_staging_root(config.dataset_root, version)
        staging_root.mkdir(parents=True, exist_ok=True)
        write_jsonl(staging_root / "train.jsonl", train_rows)
        write_jsonl(staging_root / "val.jsonl", val_rows)
        write_jsonl(staging_root / "test.jsonl", test_rows)
        write_json(
            staging_root / "split_info.json",
            {
                "base_version": None,
                "feedback_pool_version": None,
                "snapshot_version": version,
                "base_roll_forward_policy": "The initial baseline snapshot was bootstrapped directly from AMI meetings with gold topic segments.",
                "new_meeting_selection_policy": "All AMI meetings with gold topic segments were deterministically assigned with the stable 70/15/15 split.",
                "new_meeting_ids": eligible_meetings,
                "train_meeting_ids": split_assignments["train"],
                "val_meeting_ids": split_assignments["val"],
                "test_meeting_ids": split_assignments["test"],
            },
        )
        write_json(
            staging_root / "examples.json",
            {
                "train": pick_stage1_examples(train_rows),
                "val": pick_stage1_examples(val_rows),
                "test": pick_stage1_examples(test_rows),
            },
        )
        write_json(staging_root / "profile.json", profile)
        write_json(staging_root / "quality_report.json", quality_report)
        write_json(staging_root / "manifest.json", manifest)

        publish_allowed = quality_report_allows_publish(
            quality_report,
            reference_version=reference_version,
        )
        if publish_allowed:
            out_root = config.dataset_root / f"v{version}"
            move_tree(staging_root, out_root)
            object_prefix = f"{config.dataset_object_prefix.strip('/')}/v{version}"
            if config.upload_artifacts:
                upload_dir(out_root, object_prefix, logger)

            insert_dataset_version(
                conn=conn,
                dataset_name=config.dataset_name,
                stage="stage1",
                source_type="ami",
                object_key=f"{object_prefix}/manifest.json",
                manifest_json=manifest,
            )
            mark_meetings_as_consumed(conn, version, split_assignments)
        else:
            out_root = quarantine_root(config.dataset_root, version)
            move_tree(staging_root, out_root)

        insert_dataset_quality_report(
            conn,
            dataset_name=config.dataset_name,
            report_scope="retraining_snapshot",
            report_status=quality_report["status"],
            dataset_version=str(version) if publish_allowed else None,
            reference_dataset_name=config.dataset_name,
            reference_dataset_version=None if reference_version is None else str(reference_version),
            report_path=str((out_root / "quality_report.json").resolve()),
            share_drifted_features=quality_report["share_drifted_features"],
            drifted_feature_count=quality_report["drifted_feature_count"],
            total_feature_count=quality_report["total_feature_count"],
            details_json=quality_report,
        )

        if not publish_allowed:
            logger.warning(
                "AMI baseline candidate did not pass the quality gate and was quarantined | candidate_version=v%s reference=v%s status=%s reason=%s share_drifted=%.3f",
                version,
                reference_version if reference_version is not None else "none",
                quality_report["status"],
                quality_report.get("reason", "n/a"),
                quality_report["share_drifted_features"],
            )
            return 1

        logger.info(
            "Bootstrapped Stage 1 baseline v%s | meetings=%s train_rows=%s val_rows=%s test_rows=%s",
            version,
            len(eligible_meetings),
            len(train_rows),
            len(val_rows),
            len(test_rows),
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
