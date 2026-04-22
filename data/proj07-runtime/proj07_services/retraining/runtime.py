from __future__ import annotations

import json
import math
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from proj07_services.common.feedback_common import (
    build_model_utterances_by_meeting,
    build_stage1_rows,
    fetch_source_utterances,
    fetch_topic_segments,
    insert_dataset_quality_report,
    insert_dataset_version,
    label_counts,
    pick_stage1_examples,
    stable_split_70_15_15,
    upload_dir,
    write_json,
    write_jsonl,
)
from proj07_services.common.task_service_common import env_int
from proj07_services.quality.drift_control import (
    DriftGateConfig,
    build_reference_profile,
    compare_feature_columns_to_reference,
    extract_stage1_feature_columns,
    load_reference_profile,
)


STRUCTURAL_FEEDBACK_EVENT_TYPES = (
    "merge_segments",
    "split_segment",
    "boundary_correction",
)
VERSION_DIR_RE = re.compile(r"^v(\d+)$")
MEETING_ID_RE = re.compile(r"^jitsi_(?P<ts>\d{8}T\d{6}Z)_[0-9a-f]{8}$")


@dataclass(frozen=True)
class RetrainingBuildConfig:
    dataset_name: str
    feedback_pool_root: Path
    dataset_root: Path
    feedback_pool_object_prefix: str
    dataset_object_prefix: str
    upload_artifacts: bool
    window_size: int
    transition_index: int
    min_utterance_chars: int
    max_words_per_utterance: int
    quality_psi_threshold: float
    quality_max_drift_share: float
    quality_min_feature_samples: int
    quality_numeric_bin_count: int


@dataclass(frozen=True)
class RetrainingDatasetMetrics:
    valid_unversioned_meeting_count: int
    structural_feedback_event_count: int
    structural_feedback_meeting_count: int


@dataclass(frozen=True)
class FeedbackPoolBuildResult:
    version: int
    output_root: Path
    row_count: int
    candidate_meeting_ids: list[str]
    eligible_meeting_ids: list[str]


@dataclass(frozen=True)
class RetrainingSnapshotBuildResult:
    snapshot_version: int
    feedback_pool_version: int
    base_version: int | None
    output_root: Path
    selected_meeting_ids: list[str]
    split_assignments: dict[str, list[str]]
    row_counts: dict[str, int]


def existing_versions(root: Path) -> list[int]:
    if not root.exists():
        return []

    versions: list[int] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = VERSION_DIR_RE.match(child.name)
        if match:
            versions.append(int(match.group(1)))
    return sorted(versions)


def next_version(root: Path) -> int:
    return max(existing_versions(root), default=0) + 1


def latest_version(root: Path) -> int | None:
    versions = existing_versions(root)
    return versions[-1] if versions else None


def quality_gate_config(config: RetrainingBuildConfig) -> DriftGateConfig:
    return DriftGateConfig(
        psi_threshold=config.quality_psi_threshold,
        max_drift_share=config.quality_max_drift_share,
        min_feature_samples=config.quality_min_feature_samples,
        numeric_bin_count=config.quality_numeric_bin_count,
    )


def quality_report_allows_publish(report: dict[str, Any], *, reference_version: int | None) -> bool:
    status = str(report.get("status") or "")
    if status == "passed":
        return True
    return (
        status == "skipped"
        and report.get("reason") == "missing_reference_profile"
        and reference_version is None
    )


def load_or_build_dataset_profile(version_root: Path) -> dict[str, Any] | None:
    profile_path = version_root / "profile.json"
    profile = load_reference_profile(profile_path)
    if profile is not None:
        return profile

    split_paths = (
        version_root / "train.jsonl",
        version_root / "val.jsonl",
        version_root / "test.jsonl",
    )
    if not all(path.exists() for path in split_paths):
        return None

    rows: list[dict[str, Any]] = []
    for path in split_paths:
        rows.extend(_read_jsonl(path))
    if not rows:
        return None

    feature_columns, metadata = extract_stage1_feature_columns(rows, include_label=True)
    profile = build_reference_profile(
        feature_columns,
        metadata=metadata,
        bin_count=env_int("RETRAINING_DATASET_QUALITY_NUMERIC_BIN_COUNT", 10),
    )
    write_json(profile_path, profile)
    return profile


def latest_reference_profile(root: Path) -> tuple[int | None, dict[str, Any] | None]:
    version = latest_version(root)
    if version is None:
        return None, None
    return version, load_or_build_dataset_profile(root / f"v{version}")


def candidate_staging_root(base_root: Path, version_hint: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return base_root / ".staging" / f"candidate-v{version_hint}-{stamp}"


def quarantine_root(base_root: Path, version_hint: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return base_root / "_quarantine" / f"candidate-v{version_hint}-{stamp}"


def move_tree(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(source), str(destination))


def evaluate_stage1_quality_gate(
    *,
    rows: list[dict],
    include_label: bool,
    reference_root: Path,
    config: RetrainingBuildConfig,
) -> tuple[dict[str, Any], dict[str, Any], int | None]:
    feature_columns, metadata = extract_stage1_feature_columns(rows, include_label=include_label)
    current_profile = build_reference_profile(
        feature_columns,
        metadata=metadata,
        bin_count=config.quality_numeric_bin_count,
    )
    reference_version, reference_profile = latest_reference_profile(reference_root)
    report = compare_feature_columns_to_reference(
        reference_profile=reference_profile,
        current_feature_columns=feature_columns,
        current_metadata=metadata,
        config=quality_gate_config(config),
    )
    return current_profile, report, reference_version


def meeting_sort_key(meeting_id: str) -> tuple[int, str]:
    match = MEETING_ID_RE.match(meeting_id)
    if match:
        return (0, match.group("ts"))
    return (1, meeting_id)


def rows_for_meetings(rows: list[dict], meeting_ids: list[str]) -> list[dict]:
    allowed = set(meeting_ids)
    return [row for row in rows if row["input"]["meeting_id"] in allowed]


def split_new_meetings(meeting_ids: list[str]) -> tuple[list[str], list[str]]:
    if not meeting_ids:
        return [], []
    midpoint = max(1, math.ceil(len(meeting_ids) / 2))
    val_meeting_ids = meeting_ids[:midpoint]
    test_meeting_ids = meeting_ids[midpoint:]
    return val_meeting_ids, test_meeting_ids


def collect_retraining_metrics(conn) -> RetrainingDatasetMetrics:
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH eligible_meetings AS (
                SELECT meeting_id
                FROM meetings
                WHERE source_type = 'jitsi'
                  AND is_valid = TRUE
                  AND dataset_version IS NULL
            ),
            eligible_feedback AS (
                SELECT fe.feedback_event_id, fe.meeting_id
                FROM feedback_events fe
                JOIN eligible_meetings em
                  ON em.meeting_id = fe.meeting_id
                WHERE fe.event_type IN ('merge_segments', 'split_segment', 'boundary_correction')
            )
            SELECT
                (SELECT COUNT(*) FROM eligible_meetings) AS valid_unversioned_meeting_count,
                (SELECT COUNT(*) FROM eligible_feedback) AS structural_feedback_event_count,
                (SELECT COUNT(DISTINCT meeting_id) FROM eligible_feedback) AS structural_feedback_meeting_count
            """
        )
        row = cur.fetchone()

    return RetrainingDatasetMetrics(
        valid_unversioned_meeting_count=int(row["valid_unversioned_meeting_count"] or 0),
        structural_feedback_event_count=int(row["structural_feedback_event_count"] or 0),
        structural_feedback_meeting_count=int(row["structural_feedback_meeting_count"] or 0),
    )


def fetch_candidate_meeting_ids(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT m.meeting_id
            FROM meetings m
            JOIN feedback_events fe
              ON fe.meeting_id = m.meeting_id
            WHERE m.source_type = 'jitsi'
              AND m.is_valid = TRUE
              AND m.dataset_version IS NULL
              AND fe.event_type IN ('merge_segments', 'split_segment', 'boundary_correction')
            ORDER BY m.meeting_id
            """
        )
        rows = cur.fetchall()
    return [row["meeting_id"] for row in rows]


def mark_meetings_as_consumed(conn, snapshot_version: int, split_assignments: dict[str, list[str]]) -> None:
    with conn.cursor() as cur:
        for split_name, meeting_ids in split_assignments.items():
            if not meeting_ids:
                continue
            cur.execute(
                """
                UPDATE meetings
                SET dataset_version = %s,
                    dataset_split = %s
                WHERE meeting_id = ANY(%s)
                """,
                (snapshot_version, split_name, meeting_ids),
            )
    conn.commit()


def build_stage1_feedback_pool(
    conn,
    *,
    config: RetrainingBuildConfig,
    logger,
    candidate_meetings: list[str],
) -> FeedbackPoolBuildResult | None:
    candidate_meetings = sorted(set(candidate_meetings), key=meeting_sort_key)
    if not candidate_meetings:
        logger.info("No eligible retraining meetings were discovered; skipping feedback-pool build")
        return None

    source_rows = fetch_source_utterances(conn, candidate_meetings)
    corrected_segments = fetch_topic_segments(conn, candidate_meetings, "user_corrected")

    topic_segments_by_meeting: dict[str, list[dict]] = defaultdict(list)
    for row in corrected_segments:
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

    eligible_meetings = sorted(
        {row["input"]["meeting_id"] for row in dataset_rows},
        key=meeting_sort_key,
    )
    if not dataset_rows or not eligible_meetings:
        logger.info(
            "No retraining rows were produced from the eligible meetings; skipping feedback-pool build"
        )
        return None

    version = next_version(config.feedback_pool_root)
    profile, quality_report, reference_version = evaluate_stage1_quality_gate(
        rows=dataset_rows,
        include_label=True,
        reference_root=config.feedback_pool_root,
        config=config,
    )
    manifest = {
        "dataset_name": "roberta_stage1_feedback_pool",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_type": "production_feedback",
        "ongoing_version": {
            "feedback_pool_version": version,
        },
        "packaging": {
            "type": "retraining_pool",
            "reason": "Runtime service compiled the next feedback-derived Stage 1 retraining pool from valid, unconsumed Jitsi meetings.",
        },
        "params": {
            "window_size": config.window_size,
            "transition_index": config.transition_index,
            "min_utterance_chars": config.min_utterance_chars,
            "max_words_per_utterance": config.max_words_per_utterance,
            "segment_type": "user_corrected",
        },
        "meetings": {
            "candidate": len(candidate_meetings),
            "eligible": len(eligible_meetings),
            "candidate_meeting_ids": candidate_meetings,
            "eligible_meeting_ids": eligible_meetings,
        },
        "rows": label_counts(dataset_rows),
        "quality_gate": {
            "status": quality_report["status"],
            "reason": quality_report.get("reason"),
            "reference_version": reference_version,
            "share_drifted_features": quality_report["share_drifted_features"],
            "drifted_feature_count": quality_report["drifted_feature_count"],
            "total_feature_count": quality_report["total_feature_count"],
        },
    }
    staging_root = candidate_staging_root(config.feedback_pool_root, version)
    staging_root.mkdir(parents=True, exist_ok=True)

    write_jsonl(staging_root / "feedback_examples.jsonl", dataset_rows)
    write_json(
        staging_root / "meeting_ids.json",
        {
            "candidate_meeting_ids": candidate_meetings,
            "eligible_meeting_ids": eligible_meetings,
        },
    )
    write_json(staging_root / "examples.json", pick_stage1_examples(dataset_rows))
    write_json(staging_root / "profile.json", profile)
    write_json(staging_root / "quality_report.json", quality_report)
    write_json(staging_root / "manifest.json", manifest)

    publish_allowed = quality_report_allows_publish(
        quality_report,
        reference_version=reference_version,
    )
    if publish_allowed:
        out_root = config.feedback_pool_root / f"v{version}"
        move_tree(staging_root, out_root)
        object_prefix = f"{config.feedback_pool_object_prefix.strip('/')}/v{version}"
        if config.upload_artifacts:
            upload_dir(out_root, object_prefix, logger)

        insert_dataset_version(
            conn=conn,
            dataset_name="roberta_stage1_feedback_pool",
            stage="stage1",
            source_type="production_feedback",
            object_key=f"{object_prefix}/manifest.json",
            manifest_json=manifest,
        )
    else:
        out_root = quarantine_root(config.feedback_pool_root, version)
        move_tree(staging_root, out_root)

    insert_dataset_quality_report(
        conn,
        dataset_name="roberta_stage1_feedback_pool",
        report_scope="feedback_pool",
        report_status=quality_report["status"],
        dataset_version=str(version) if publish_allowed else None,
        reference_dataset_name="roberta_stage1_feedback_pool",
        reference_dataset_version=None if reference_version is None else str(reference_version),
        report_path=str((out_root / "quality_report.json").resolve()),
        share_drifted_features=quality_report["share_drifted_features"],
        drifted_feature_count=quality_report["drifted_feature_count"],
        total_feature_count=quality_report["total_feature_count"],
        details_json=quality_report,
    )

    if not publish_allowed:
        logger.warning(
            "Feedback pool candidate did not pass quality gate and was quarantined | candidate_version=v%s reference=v%s status=%s reason=%s share_drifted=%.3f",
            version,
            reference_version if reference_version is not None else "none",
            quality_report["status"],
            quality_report.get("reason", "n/a"),
            quality_report["share_drifted_features"],
        )
        return None

    logger.info(
        "Built Stage 1 feedback pool v%s | meetings=%s rows=%s",
        version,
        len(eligible_meetings),
        len(dataset_rows),
    )
    return FeedbackPoolBuildResult(
        version=version,
        output_root=out_root,
        row_count=len(dataset_rows),
        candidate_meeting_ids=candidate_meetings,
        eligible_meeting_ids=eligible_meetings,
    )


def build_retraining_snapshot(
    conn,
    *,
    config: RetrainingBuildConfig,
    logger,
    feedback_pool: FeedbackPoolBuildResult,
    selected_meeting_ids: list[str] | None = None,
) -> RetrainingSnapshotBuildResult:
    snapshot_version = next_version(config.dataset_root)
    base_version = latest_version(config.dataset_root)
    feedback_dir = feedback_pool.output_root

    feedback_rows = _read_jsonl(feedback_dir / "feedback_examples.jsonl")
    feedback_manifest = _read_json(feedback_dir / "meeting_ids.json")
    if not feedback_rows:
        raise RuntimeError("Feedback pool is empty; cannot build retraining snapshot")

    if selected_meeting_ids:
        new_meeting_ids = sorted(set(selected_meeting_ids), key=meeting_sort_key)
    else:
        new_meeting_ids = sorted(
            feedback_manifest.get("eligible_meeting_ids", []),
            key=meeting_sort_key,
        )

    if not new_meeting_ids:
        raise RuntimeError("No newly eligible meetings were selected for retraining snapshot")

    available_feedback_meeting_ids = {row["input"]["meeting_id"] for row in feedback_rows}
    missing_meetings = [meeting_id for meeting_id in new_meeting_ids if meeting_id not in available_feedback_meeting_ids]
    if missing_meetings:
        raise RuntimeError(
            "Selected meetings are not present in the feedback pool: " + ", ".join(missing_meetings)
        )

    if base_version is None:
        split_assignments = stable_split_70_15_15(new_meeting_ids)
        train_rows = rows_for_meetings(feedback_rows, split_assignments["train"])
        val_rows = rows_for_meetings(feedback_rows, split_assignments["val"])
        test_rows = rows_for_meetings(feedback_rows, split_assignments["test"])
        split_info = {
            "base_version": None,
            "feedback_pool_version": feedback_pool.version,
            "snapshot_version": snapshot_version,
            "base_roll_forward_policy": "No historical base dataset was available, so the snapshot was bootstrapped directly from production feedback meetings.",
            "new_meeting_selection_policy": "Only meetings whose dataset_version was NULL at retraining discovery time are eligible for assignment.",
            "new_meeting_ids": new_meeting_ids,
            "train_meeting_ids": split_assignments["train"],
            "val_meeting_ids": split_assignments["val"],
            "test_meeting_ids": split_assignments["test"],
        }
        packaging = {
            "type": "bootstrap_from_production_feedback",
            "reason": "No historical base dataset was found under the configured dataset root, so the runtime bootstrapped the first snapshot from current production feedback.",
        }
        composition = {
            "base_source": None,
            "feedback_pool_source": f"roberta_stage1_feedback_pool/v{feedback_pool.version}",
        }
    else:
        base_dir = config.dataset_root / f"v{base_version}"
        train_rows = _read_jsonl(base_dir / "train.jsonl")
        val_rows_base = _read_jsonl(base_dir / "val.jsonl")
        test_rows_base = _read_jsonl(base_dir / "test.jsonl")
        train_rows = train_rows + val_rows_base + test_rows_base
        val_meeting_ids, test_meeting_ids = split_new_meetings(new_meeting_ids)
        val_rows = rows_for_meetings(feedback_rows, val_meeting_ids)
        test_rows = rows_for_meetings(feedback_rows, test_meeting_ids)
        split_assignments = {
            "val": val_meeting_ids,
            "test": test_meeting_ids,
        }
        split_info = {
            "base_version": base_version,
            "feedback_pool_version": feedback_pool.version,
            "snapshot_version": snapshot_version,
            "base_roll_forward_policy": "The previous dataset version train/val/test are merged into the new train split before new production meetings are assigned.",
            "new_meeting_selection_policy": "Only meetings whose dataset_version was NULL at retraining discovery time are eligible for assignment.",
            "new_meeting_ids": new_meeting_ids,
            "val_meeting_ids": val_meeting_ids,
            "test_meeting_ids": test_meeting_ids,
        }
        packaging = {
            "type": "rolling_temporal_snapshot",
            "reason": "Each new retraining run rolls the previous dataset version forward into train and reserves only newly-ingested production meetings for validation and test.",
        }
        composition = {
            "base_source": f"{config.dataset_name}/v{base_version}",
            "feedback_pool_source": f"roberta_stage1_feedback_pool/v{feedback_pool.version}",
        }

    combined_rows = train_rows + val_rows + test_rows
    profile, quality_report, reference_version = evaluate_stage1_quality_gate(
        rows=combined_rows,
        include_label=True,
        reference_root=config.dataset_root,
        config=config,
    )
    manifest = {
        "dataset_name": config.dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_type": "historical_plus_production_feedback" if base_version is not None else "production_feedback_bootstrap",
        "composition": composition,
        "packaging": packaging,
        "ongoing_version": {
            "snapshot_version": snapshot_version,
            "base_version": base_version,
            "feedback_pool_version": feedback_pool.version,
        },
        "splits": {
            "train": label_counts(train_rows),
            "val": label_counts(val_rows),
            "test": label_counts(test_rows),
        },
        "meetings": split_info,
        "quality_gate": {
            "status": quality_report["status"],
            "reason": quality_report.get("reason"),
            "reference_version": reference_version,
            "share_drifted_features": quality_report["share_drifted_features"],
            "drifted_feature_count": quality_report["drifted_feature_count"],
            "total_feature_count": quality_report["total_feature_count"],
        },
    }
    staging_root = candidate_staging_root(config.dataset_root, snapshot_version)
    staging_root.mkdir(parents=True, exist_ok=True)

    write_jsonl(staging_root / "train.jsonl", train_rows)
    write_jsonl(staging_root / "val.jsonl", val_rows)
    write_jsonl(staging_root / "test.jsonl", test_rows)
    write_json(staging_root / "split_info.json", split_info)
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
        out_root = config.dataset_root / f"v{snapshot_version}"
        move_tree(staging_root, out_root)
        object_prefix = f"{config.dataset_object_prefix.strip('/')}/v{snapshot_version}"
        if config.upload_artifacts:
            upload_dir(out_root, object_prefix, logger)

        insert_dataset_version(
            conn=conn,
            dataset_name=config.dataset_name,
            stage="stage1",
            source_type="production_feedback",
            object_key=f"{object_prefix}/manifest.json",
            manifest_json=manifest,
        )
        mark_meetings_as_consumed(conn, snapshot_version, split_assignments)
    else:
        out_root = quarantine_root(config.dataset_root, snapshot_version)
        move_tree(staging_root, out_root)

    insert_dataset_quality_report(
        conn,
        dataset_name=config.dataset_name,
        report_scope="retraining_snapshot",
        report_status=quality_report["status"],
        dataset_version=str(snapshot_version) if publish_allowed else None,
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
            "Retraining snapshot candidate did not pass quality gate and was quarantined | candidate_version=v%s reference=v%s status=%s reason=%s share_drifted=%.3f",
            snapshot_version,
            reference_version if reference_version is not None else "none",
            quality_report["status"],
            quality_report.get("reason", "n/a"),
            quality_report["share_drifted_features"],
        )
        raise RuntimeError(
            f"Retraining snapshot v{snapshot_version} did not pass the quality gate and was quarantined"
        )

    row_counts = {
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
    }
    logger.info(
        "Built retraining snapshot v%s | base=%s feedback_pool=%s train_rows=%s val_rows=%s test_rows=%s",
        snapshot_version,
        f"v{base_version}" if base_version is not None else "bootstrap",
        feedback_pool.version,
        row_counts["train"],
        row_counts["val"],
        row_counts["test"],
    )
    return RetrainingSnapshotBuildResult(
        snapshot_version=snapshot_version,
        feedback_pool_version=feedback_pool.version,
        base_version=base_version,
        output_root=out_root,
        selected_meeting_ids=new_meeting_ids,
        split_assignments=split_assignments,
        row_counts=row_counts,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
