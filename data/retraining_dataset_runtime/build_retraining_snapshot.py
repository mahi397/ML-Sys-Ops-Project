#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from feedback_common import (
    ensure_dir,
    get_conn,
    insert_dataset_version,
    label_counts,
    pick_stage1_examples,
    setup_logger,
    upload_dir,
    write_json,
    write_jsonl,
)


MEETING_ID_RE = re.compile(r"^jitsi_(?P<ts>\d{8}T\d{6}Z)_[0-9a-f]{8}$")
VERSION_DIR_RE = re.compile(r"^v(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/block/roberta_stage1"),
        help="Local root containing all versioned roberta_stage1 snapshots.",
    )
    parser.add_argument(
        "--base-version",
        default="auto",
        help="Base version to roll forward. Use 'auto' to use the latest existing dataset version.",
    )
    parser.add_argument(
        "--feedback-pool-root",
        type=Path,
        default=Path("/mnt/block/staging/feedback_loop/datasets/roberta_stage1_feedback_pool"),
        help="Local root containing feedback pool versions.",
    )
    parser.add_argument(
        "--feedback-pool-version",
        default="auto",
        help="Feedback-pool version to consume. Use 'auto' to use the latest existing feedback-pool version.",
    )
    parser.add_argument(
        "--snapshot-version",
        default="auto",
        help="Snapshot version to write. Use 'auto' to assign max(existing)+1.",
    )
    parser.add_argument("--dataset-name", default="roberta_stage1")
    parser.add_argument(
        "--object-prefix",
        default="datasets/roberta_stage1",
        help="Object-store prefix for uploaded snapshot artifacts.",
    )
    parser.add_argument("--upload-artifacts", action="store_true")
    parser.add_argument(
        "--meeting-id",
        action="append",
        default=[],
        help="Explicit newly-arrived meeting ids to assign into val/test. If omitted, the feedback-pool manifest is used.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


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


def resolve_version(version_arg: str, root: Path, *, mode: str) -> int:
    versions = existing_versions(root)
    if version_arg != "auto":
        return int(version_arg)
    if mode == "latest":
        if not versions:
            raise RuntimeError(f"No existing version directories found under {root}")
        return versions[-1]
    if mode == "next":
        return max(versions, default=0) + 1
    raise ValueError(f"Unsupported mode: {mode}")


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


def mark_meetings_as_consumed(
    snapshot_version: int,
    val_meeting_ids: list[str],
    test_meeting_ids: list[str],
) -> None:
    conn = get_conn()
    with conn.cursor() as cur:
        for meeting_id in val_meeting_ids:
            cur.execute(
                """
                UPDATE meetings
                SET dataset_version = %s,
                    dataset_split = 'val'
                WHERE meeting_id = %s
                """,
                (snapshot_version, meeting_id),
            )
        for meeting_id in test_meeting_ids:
            cur.execute(
                """
                UPDATE meetings
                SET dataset_version = %s,
                    dataset_split = 'test'
                WHERE meeting_id = %s
                """,
                (snapshot_version, meeting_id),
            )
    conn.commit()


def main() -> None:
    args = parse_args()
    logger = setup_logger("build_retraining_snapshot")

    snapshot_version = resolve_version(args.snapshot_version, args.dataset_root, mode="next")
    base_version = resolve_version(args.base_version, args.dataset_root, mode="latest")
    feedback_pool_version = resolve_version(args.feedback_pool_version, args.feedback_pool_root, mode="latest")

    if base_version >= snapshot_version:
        raise RuntimeError(
            f"Base version v{base_version} must be older than the new snapshot version v{snapshot_version}"
        )

    base_dir = args.dataset_root / f"v{base_version}"
    feedback_dir = args.feedback_pool_root / f"v{feedback_pool_version}"

    train_path = base_dir / "train.jsonl"
    val_path = base_dir / "val.jsonl"
    test_path = base_dir / "test.jsonl"
    feedback_path = feedback_dir / "feedback_examples.jsonl"
    meeting_ids_path = feedback_dir / "meeting_ids.json"

    for required_path in [train_path, val_path, test_path, feedback_path, meeting_ids_path]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required input file not found: {required_path}")

    base_train = read_jsonl(train_path)
    base_val = read_jsonl(val_path)
    base_test = read_jsonl(test_path)
    feedback_rows = read_jsonl(feedback_path)
    feedback_manifest = read_json(meeting_ids_path)

    if not feedback_rows:
        raise RuntimeError("Feedback pool is empty; cannot build Objective 4 snapshot")

    if args.meeting_id:
        new_meeting_ids = sorted(set(args.meeting_id), key=meeting_sort_key)
    else:
        new_meeting_ids = sorted(
            feedback_manifest.get("eligible_meeting_ids", []),
            key=meeting_sort_key,
        )

    if not new_meeting_ids:
        raise RuntimeError("No newly-arrived meetings were selected for validation/test")

    available_feedback_meeting_ids = {row["input"]["meeting_id"] for row in feedback_rows}
    missing_meetings = [meeting_id for meeting_id in new_meeting_ids if meeting_id not in available_feedback_meeting_ids]
    if missing_meetings:
        raise RuntimeError(
            "Selected meetings are not present in the feedback pool: " + ", ".join(missing_meetings)
        )

    new_train_rows = base_train + base_val + base_test
    val_meeting_ids, test_meeting_ids = split_new_meetings(new_meeting_ids)
    new_val_rows = rows_for_meetings(feedback_rows, val_meeting_ids)
    new_test_rows = rows_for_meetings(feedback_rows, test_meeting_ids)

    out_root = args.dataset_root / f"v{snapshot_version}"
    ensure_dir(out_root)

    write_jsonl(out_root / "train.jsonl", new_train_rows)
    write_jsonl(out_root / "val.jsonl", new_val_rows)
    write_jsonl(out_root / "test.jsonl", new_test_rows)

    split_info = {
        "base_version": base_version,
        "feedback_pool_version": feedback_pool_version,
        "snapshot_version": snapshot_version,
        "base_roll_forward_policy": "The previous dataset version train/val/test are merged into the new train split before new production meetings are assigned.",
        "new_meeting_selection_policy": "Only meetings whose dataset_version was NULL at Objective 4 discovery time are eligible for assignment.",
        "new_meeting_ids": new_meeting_ids,
        "val_meeting_ids": val_meeting_ids,
        "test_meeting_ids": test_meeting_ids,
    }
    write_json(out_root / "split_info.json", split_info)

    examples = {
        "train": pick_stage1_examples(new_train_rows),
        "val": pick_stage1_examples(new_val_rows),
        "test": pick_stage1_examples(new_test_rows),
    }
    write_json(out_root / "examples.json", examples)

    manifest = {
        "dataset_name": args.dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_type": "historical_plus_production_feedback",
        "composition": {
            "base_source": f"{args.dataset_name}/v{base_version}",
            "feedback_pool_source": f"roberta_stage1_feedback_pool/v{feedback_pool_version}",
        },
        "packaging": {
            "type": "rolling_temporal_snapshot",
            "reason": "Each new Objective 4 run rolls the previous dataset version forward into train and reserves only newly-ingested production meetings for validation and test.",
        },
        "ongoing_version": {
            "snapshot_version": snapshot_version,
            "base_version": base_version,
            "feedback_pool_version": feedback_pool_version,
        },
        "splits": {
            "train": label_counts(new_train_rows),
            "val": label_counts(new_val_rows),
            "test": label_counts(new_test_rows),
        },
        "meetings": split_info,
    }
    write_json(out_root / "manifest.json", manifest)

    object_prefix = f"{args.object_prefix.strip('/')}/v{snapshot_version}"
    if args.upload_artifacts:
        upload_dir(out_root, object_prefix, logger)

    conn = get_conn()
    insert_dataset_version(
        conn=conn,
        dataset_name=args.dataset_name,
        stage="stage1",
        source_type="production_feedback",
        object_key=f"{object_prefix}/manifest.json",
        manifest_json=manifest,
    )
    mark_meetings_as_consumed(snapshot_version, val_meeting_ids, test_meeting_ids)

    logger.info(
        "Built rolling retraining snapshot v%s from base=v%s feedback_pool=v%s train_rows=%d val_rows=%d test_rows=%d",
        snapshot_version,
        base_version,
        feedback_pool_version,
        len(new_train_rows),
        len(new_val_rows),
        len(new_test_rows),
    )


if __name__ == "__main__":
    main()
