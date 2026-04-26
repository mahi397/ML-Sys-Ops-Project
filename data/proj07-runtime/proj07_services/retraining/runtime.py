from __future__ import annotations

import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from proj07_services.common.feedback_common import (
    build_model_utterances_by_meeting,
    build_stage1_rows,
    dataset_object_prefix_from_key,
    download_dir,
    fetch_source_utterances,
    fetch_topic_segments,
    insert_dataset_version,
    label_counts,
    latest_dataset_version_record,
    list_dataset_version_records,
    next_dataset_version_number,
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
PREFERRED_STAGE1_SEGMENT_TYPES = (
    "user_corrected",
    "predicted",
)
VERSION_DIR_RE = re.compile(r"^v(\d+)$")
MEETING_ID_RE = re.compile(r"^jitsi_(?P<ts>\d{8}T\d{6}Z)_[0-9a-f]{8}$")
EMULATED_HOST_EMAIL_SUFFIX = ".mock@example.com"
EMULATED_SOURCE_NAME_PREFIX = "synthetic-"


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
    stage1_segmented_meeting_count: int
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


def quality_report_allows_publish(
    report: dict[str, Any],
    *,
    reference_version: int | None,
    force_publish: bool = False,
) -> bool:
    if force_publish:
        return True
    status = str(report.get("status") or "")
    if status == "passed":
        return True
    return (
        status == "skipped"
        and report.get("reason") == "missing_reference_profile"
        and reference_version is None
    )


def is_usable_reference_profile(profile: dict[str, Any] | None) -> bool:
    if not profile:
        return False
    return bool(profile.get("features"))


def production_jitsi_meeting_filter_sql(meeting_alias: str = "m") -> str:
    alias = meeting_alias.strip() or "m"
    return f"""
        {alias}.source_type = 'jitsi'
        AND LEFT(
            LOWER(COALESCE({alias}.source_name, '')),
            LENGTH('{EMULATED_SOURCE_NAME_PREFIX}')
        ) != '{EMULATED_SOURCE_NAME_PREFIX}'
        AND NOT EXISTS (
            SELECT 1
            FROM meeting_participants host_mp
            JOIN users host_user
              ON host_user.user_id = host_mp.user_id
            WHERE host_mp.meeting_id = {alias}.meeting_id
              AND host_mp.role = 'host'
              AND RIGHT(
                  LOWER(COALESCE(host_user.email, '')),
                  LENGTH('{EMULATED_HOST_EMAIL_SUFFIX}')
              ) = '{EMULATED_HOST_EMAIL_SUFFIX}'
        )
    """


def load_or_build_dataset_profile(version_root: Path) -> dict[str, Any] | None:
    profile_path = version_root / "profile.json"
    profile = load_reference_profile(profile_path)
    if is_usable_reference_profile(profile):
        return profile

    split_paths = (
        version_root / "train.jsonl",
        version_root / "val.jsonl",
        version_root / "test.jsonl",
    )
    rows: list[dict[str, Any]] = []
    if all(path.exists() for path in split_paths):
        for path in split_paths:
            rows.extend(_read_jsonl(path))
    if not rows:
        feedback_pool_path = version_root / "feedback_examples.jsonl"
        if feedback_pool_path.exists():
            rows = _read_jsonl(feedback_pool_path)
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


def ensure_cached_version_dir(
    local_root: Path,
    *,
    version: int,
    object_key: str,
    logger,
    required_files: tuple[str, ...] = ("manifest.json",),
) -> Path:
    version_root = local_root / f"v{version}"
    if all((version_root / relative_path).exists() for relative_path in required_files):
        return version_root

    download_dir(
        dataset_object_prefix_from_key(object_key),
        version_root,
        logger,
    )
    if not all((version_root / relative_path).exists() for relative_path in required_files):
        raise RuntimeError(
            f"Dataset cache under {version_root} is incomplete after staging from object storage"
        )
    return version_root


def latest_reference_profile(
    conn,
    *,
    dataset_name: str,
    stage: str,
    reference_root: Path,
    logger,
) -> tuple[int | None, dict[str, Any] | None]:
    records = list_dataset_version_records(conn, dataset_name=dataset_name, stage=stage)
    for record in reversed(records):
        version = int(record["version"])
        version_root = reference_root / f"v{version}"
        profile = load_or_build_dataset_profile(version_root)
        if not is_usable_reference_profile(profile):
            version_root = ensure_cached_version_dir(
                reference_root,
                version=version,
                object_key=str(record["object_key"]),
                logger=logger,
                required_files=("manifest.json",),
            )
            profile = load_or_build_dataset_profile(version_root)
        if is_usable_reference_profile(profile):
            return version, profile
    return None, None


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


def dataset_quality_report_object_key(
    *,
    object_prefix: str,
    version: int,
    out_root: Path,
    publish_allowed: bool,
) -> str:
    normalized_prefix = object_prefix.strip("/")
    if publish_allowed:
        return f"{normalized_prefix}/v{version}/quality_report.json"
    return f"{normalized_prefix}/_quarantine/{out_root.name}/quality_report.json"


def evaluate_stage1_quality_gate(
    *,
    conn,
    rows: list[dict],
    include_label: bool,
    reference_dataset_name: str,
    reference_stage: str,
    reference_root: Path,
    logger,
    config: RetrainingBuildConfig,
) -> tuple[dict[str, Any], dict[str, Any], int | None]:
    feature_columns, metadata = extract_stage1_feature_columns(rows, include_label=include_label)
    current_profile = build_reference_profile(
        feature_columns,
        metadata=metadata,
        bin_count=config.quality_numeric_bin_count,
    )
    reference_version, reference_profile = latest_reference_profile(
        conn,
        dataset_name=reference_dataset_name,
        stage=reference_stage,
        reference_root=reference_root,
        logger=logger,
    )
    report = compare_feature_columns_to_reference(
        reference_profile=reference_profile,
        current_feature_columns=feature_columns,
        current_metadata=metadata,
        config=quality_gate_config(config),
    )
    return current_profile, report, reference_version


def build_stage1_profile(
    *,
    rows: list[dict],
    include_label: bool,
    config: RetrainingBuildConfig,
) -> dict[str, Any]:
    feature_columns, metadata = extract_stage1_feature_columns(rows, include_label=include_label)
    return build_reference_profile(
        feature_columns,
        metadata=metadata,
        bin_count=config.quality_numeric_bin_count,
    )


def meeting_sort_key(meeting_id: str) -> tuple[int, str]:
    match = MEETING_ID_RE.match(meeting_id)
    if match:
        return (0, match.group("ts"))
    return (1, meeting_id)


def rows_for_meetings(rows: list[dict], meeting_ids: list[str]) -> list[dict]:
    allowed = set(meeting_ids)
    return [row for row in rows if row["input"]["meeting_id"] in allowed]


def collect_retraining_metrics(conn) -> RetrainingDatasetMetrics:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            WITH eligible_meetings AS (
                SELECT meeting_id
                FROM meetings m
                WHERE {production_jitsi_meeting_filter_sql("m")}
                  AND is_valid = TRUE
                  AND dataset_version IS NULL
            ),
            eligible_segmented_meetings AS (
                SELECT DISTINCT em.meeting_id
                FROM eligible_meetings em
                JOIN topic_segments ts
                  ON ts.meeting_id = em.meeting_id
                WHERE ts.segment_type = ANY(%s)
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
                (SELECT COUNT(*) FROM eligible_segmented_meetings) AS stage1_segmented_meeting_count,
                (SELECT COUNT(*) FROM eligible_feedback) AS structural_feedback_event_count,
                (SELECT COUNT(DISTINCT meeting_id) FROM eligible_feedback) AS structural_feedback_meeting_count
            """,
            (list(PREFERRED_STAGE1_SEGMENT_TYPES),),
        )
        row = cur.fetchone()

    return RetrainingDatasetMetrics(
        valid_unversioned_meeting_count=int(row["valid_unversioned_meeting_count"] or 0),
        stage1_segmented_meeting_count=int(row["stage1_segmented_meeting_count"] or 0),
        structural_feedback_event_count=int(row["structural_feedback_event_count"] or 0),
        structural_feedback_meeting_count=int(row["structural_feedback_meeting_count"] or 0),
    )


def fetch_candidate_meeting_ids(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT DISTINCT m.meeting_id
            FROM meetings m
            WHERE {production_jitsi_meeting_filter_sql("m")}
              AND m.is_valid = TRUE
              AND m.dataset_version IS NULL
              AND EXISTS (
                  SELECT 1
                  FROM topic_segments ts
                  WHERE ts.meeting_id = m.meeting_id
                    AND ts.segment_type = ANY(%s)
              )
            ORDER BY m.meeting_id
            """,
            (list(PREFERRED_STAGE1_SEGMENT_TYPES),),
        )
        rows = cur.fetchall()
    return [row["meeting_id"] for row in rows]


def fetch_preferred_stage1_segments(
    conn,
    meeting_ids: list[str],
) -> tuple[dict[str, list[dict]], dict[str, str]]:
    preferred_segments_by_meeting: dict[str, list[dict]] = {}
    segment_source_by_meeting: dict[str, str] = {}
    segments_by_type: dict[str, dict[str, list[dict]]] = {}

    for segment_type in PREFERRED_STAGE1_SEGMENT_TYPES:
        rows = fetch_topic_segments(conn, meeting_ids, segment_type)
        grouped_rows: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped_rows[row["meeting_id"]].append(row)
        segments_by_type[segment_type] = dict(grouped_rows)

    for meeting_id in meeting_ids:
        for segment_type in PREFERRED_STAGE1_SEGMENT_TYPES:
            chosen = segments_by_type.get(segment_type, {}).get(meeting_id)
            if chosen:
                preferred_segments_by_meeting[meeting_id] = chosen
                segment_source_by_meeting[meeting_id] = segment_type
                break

    return preferred_segments_by_meeting, segment_source_by_meeting


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
    metrics: RetrainingDatasetMetrics | None = None,
    force_publish: bool = False,
) -> FeedbackPoolBuildResult | None:
    candidate_meetings = sorted(set(candidate_meetings), key=meeting_sort_key)
    structural_feedback_event_count = (
        "n/a" if metrics is None else metrics.structural_feedback_event_count
    )
    structural_feedback_meeting_count = (
        "n/a" if metrics is None else metrics.structural_feedback_meeting_count
    )
    if not candidate_meetings:
        logger.info(
            "No eligible retraining meetings were discovered; skipping feedback-pool build | candidate_meetings=0 structural_feedback_events=%s structural_feedback_meetings=%s",
            structural_feedback_event_count,
            structural_feedback_meeting_count,
        )
        return None

    source_rows = fetch_source_utterances(conn, candidate_meetings)
    topic_segments_by_meeting, segment_source_by_meeting = fetch_preferred_stage1_segments(
        conn,
        candidate_meetings,
    )

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
            "No retraining rows were produced from the eligible meetings; skipping feedback-pool build | candidate_meetings=%s eligible_meetings=%s structural_feedback_events=%s structural_feedback_meetings=%s",
            len(candidate_meetings),
            len(eligible_meetings),
            structural_feedback_event_count,
            structural_feedback_meeting_count,
        )
        return None

    segment_source_counts = Counter(
        segment_source_by_meeting[meeting_id]
        for meeting_id in eligible_meetings
        if meeting_id in segment_source_by_meeting
    )

    version = next_dataset_version_number(
        conn,
        dataset_name="roberta_stage1_feedback_pool",
        stage="stage1",
    )
    profile = build_stage1_profile(
        rows=dataset_rows,
        include_label=True,
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
            "reason": "Runtime service compiled the next Stage 1 retraining pool from valid, unconsumed Jitsi meetings, preferring user-corrected segments when available and otherwise falling back to predicted segments.",
        },
        "params": {
            "window_size": config.window_size,
            "transition_index": config.transition_index,
            "min_utterance_chars": config.min_utterance_chars,
            "max_words_per_utterance": config.max_words_per_utterance,
            "segment_selection_policy": "prefer_user_corrected_else_predicted",
        },
        "meetings": {
            "candidate": len(candidate_meetings),
            "eligible": len(eligible_meetings),
            "candidate_meeting_ids": candidate_meetings,
            "eligible_meeting_ids": eligible_meetings,
            "segment_source_counts": dict(segment_source_counts),
        },
        "rows": label_counts(dataset_rows),
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
    write_json(staging_root / "manifest.json", manifest)

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

    logger.info(
        "Built Stage 1 feedback pool v%s | candidate_meetings=%s eligible_meetings=%s structural_feedback_events=%s structural_feedback_meetings=%s rows=%s",
        version,
        len(candidate_meetings),
        len(eligible_meetings),
        structural_feedback_event_count,
        structural_feedback_meeting_count,
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
    force_publish: bool = False,
) -> RetrainingSnapshotBuildResult:
    snapshot_version = next_dataset_version_number(
        conn,
        dataset_name=config.dataset_name,
        stage="stage1",
    )
    base_record = latest_dataset_version_record(
        conn,
        dataset_name=config.dataset_name,
        stage="stage1",
    )
    base_version = None if base_record is None else int(base_record["version"])
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
        if base_record is None:
            raise RuntimeError("Latest dataset version record disappeared before base snapshot staging")
        base_dir = ensure_cached_version_dir(
            config.dataset_root,
            version=base_version,
            object_key=str(base_record["object_key"]),
            logger=logger,
            required_files=("manifest.json", "train.jsonl", "val.jsonl", "test.jsonl"),
        )
        train_rows_base = _read_jsonl(base_dir / "train.jsonl")
        val_rows_base = _read_jsonl(base_dir / "val.jsonl")
        test_rows_base = _read_jsonl(base_dir / "test.jsonl")
        new_split_assignments = stable_split_70_15_15(new_meeting_ids)
        new_train_meeting_ids = new_split_assignments["train"]
        new_val_meeting_ids = new_split_assignments["val"]
        new_test_meeting_ids = new_split_assignments["test"]
        train_rows = train_rows_base + rows_for_meetings(feedback_rows, new_train_meeting_ids)
        val_rows = val_rows_base + rows_for_meetings(feedback_rows, new_val_meeting_ids)
        test_rows = test_rows_base + rows_for_meetings(feedback_rows, new_test_meeting_ids)
        split_assignments = {
            "train": new_train_meeting_ids,
            "val": new_val_meeting_ids,
            "test": new_test_meeting_ids,
        }
        split_info = {
            "base_version": base_version,
            "feedback_pool_version": feedback_pool.version,
            "snapshot_version": snapshot_version,
            "base_roll_forward_policy": "The previous dataset version keeps its train/val/test split assignments, and only newly eligible production meetings are appended to the corresponding split.",
            "new_meeting_selection_policy": "Only meetings whose dataset_version was NULL at retraining discovery time are eligible for assignment.",
            "new_meeting_ids": new_meeting_ids,
            "new_train_meeting_ids": new_train_meeting_ids,
            "new_val_meeting_ids": new_val_meeting_ids,
            "new_test_meeting_ids": new_test_meeting_ids,
        }
        packaging = {
            "type": "rolling_temporal_snapshot",
            "reason": "Each new retraining run preserves the prior split boundaries and appends a fresh meeting-level 70/15/15 split of newly eligible production feedback meetings to train, validation, and test respectively.",
        }
        composition = {
            "base_source": f"{config.dataset_name}/v{base_version}",
            "feedback_pool_source": f"roberta_stage1_feedback_pool/v{feedback_pool.version}",
        }

    combined_rows = train_rows + val_rows + test_rows
    profile = build_stage1_profile(
        rows=combined_rows,
        include_label=True,
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
    write_json(staging_root / "manifest.json", manifest)

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

    row_counts = {
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
    }
    logger.info(
        "Built retraining snapshot v%s | base=%s feedback_pool=%s meetings=%s train_rows=%s val_rows=%s test_rows=%s",
        snapshot_version,
        f"v{base_version}" if base_version is not None else "bootstrap",
        feedback_pool.version,
        len(new_meeting_ids),
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
