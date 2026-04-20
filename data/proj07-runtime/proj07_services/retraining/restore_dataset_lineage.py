#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from proj07_services.common.feedback_common import (
    ensure_dataset_version_record,
    get_conn,
)
from proj07_services.retraining.runtime import existing_versions


APP_NAME = "restore_dataset_lineage"
VERSION_DIR_RE = re.compile(r"^v(?P<version>\d+)/?$")
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class VersionFamily:
    label: str
    local_root: Path
    object_prefix: str
    dataset_name_fallback: str
    stage: str
    metadata_filename: str
    required_files: tuple[str, ...]
    default_source_type: str
    restore_meeting_assignments: bool


@dataclass(frozen=True)
class StoredVersion:
    family: VersionFamily
    version: int
    manifest: dict[str, Any]
    metadata: dict[str, Any]
    dataset_name: str
    source_type: str
    object_key: str
    meeting_stamp_version: int | None
    split_assignments: dict[str, list[str]]


@dataclass(frozen=True)
class PlannedFamilyRestore:
    family: VersionFamily
    source: str
    versions: list[int]
    stored_versions: list[StoredVersion]


def default_local_tmp_root() -> Path:
    return Path(
        os.getenv(
            "RETRAINING_LOCAL_TMP_ROOT",
            os.getenv("LOCAL_TMP_ROOT", "/mnt/block/staging/feedback_loop"),
        )
    )


def parse_args() -> argparse.Namespace:
    local_tmp_root = default_local_tmp_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--rclone-remote", default=os.getenv("RCLONE_REMOTE", "rclone_s3"))
    parser.add_argument(
        "--bucket",
        default=(
            os.getenv("OBJECT_BUCKET", "").strip()
            or os.getenv("BUCKET", "").strip()
            or "objstore-proj07"
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default=os.getenv("RETRAINING_DATASET_NAME", "roberta_stage1").strip(),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            os.getenv(
                "RETRAINING_DATASET_ROOT",
                os.getenv("DATASET_ROOT", "/mnt/block/roberta_stage1"),
            )
        ),
    )
    parser.add_argument(
        "--dataset-object-prefix",
        default=os.getenv(
            "RETRAINING_DATASET_OBJECT_PREFIX",
            os.getenv("FINAL_DATASET_OBJECT_PREFIX", "datasets/roberta_stage1"),
        ).strip(),
    )
    parser.add_argument(
        "--feedback-pool-dataset-name",
        default=os.getenv("RETRAINING_FEEDBACK_POOL_DATASET_NAME", "roberta_stage1_feedback_pool").strip(),
    )
    parser.add_argument(
        "--feedback-pool-root",
        type=Path,
        default=Path(
            os.getenv(
                "RETRAINING_FEEDBACK_POOL_ROOT",
                str(local_tmp_root / "datasets" / "roberta_stage1_feedback_pool"),
            )
        ),
    )
    parser.add_argument(
        "--feedback-pool-object-prefix",
        default=os.getenv(
            "RETRAINING_FEEDBACK_POOL_PREFIX",
            os.getenv("STAGE1_FEEDBACK_POOL_PREFIX", "datasets/roberta_stage1_feedback_pool"),
        ).strip(),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("/mnt/block/ingest_logs/retraining_dataset_lineage_restore.log"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def setup_logger(log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def run_command(
    cmd: list[str],
    *,
    logger: logging.Logger,
    label: str,
    capture: bool = False,
    allow_failure: bool = False,
) -> subprocess.CompletedProcess:
    logger.info("Running %s: %s", label, " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=capture, check=False)
    if result.returncode != 0 and not allow_failure:
        if capture and result.stdout:
            logger.error("stdout:\n%s", result.stdout)
        if capture and result.stderr:
            logger.error("stderr:\n%s", result.stderr)
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")
    return result


def remote_dir_path(remote: str, bucket: str, prefix: str) -> str:
    normalized_prefix = prefix.strip("/")
    if normalized_prefix:
        return f"{remote}:{bucket}/{normalized_prefix}"
    return f"{remote}:{bucket}"


def remote_object_path(remote: str, bucket: str, object_key: str) -> str:
    normalized_key = object_key.strip("/")
    if normalized_key:
        return f"{remote}:{bucket}/{normalized_key}"
    return f"{remote}:{bucket}"


def list_remote_versions(
    *,
    remote: str,
    bucket: str,
    prefix: str,
    logger: logging.Logger,
) -> list[int]:
    result = run_command(
        ["rclone", "lsf", "--dirs-only", remote_dir_path(remote, bucket, prefix)],
        logger=logger,
        label=f"list remote versions under {prefix}",
        capture=True,
        allow_failure=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        if stderr:
            logger.info(
                "Remote version scan skipped for %s because rclone could not list it: %s",
                prefix,
                stderr,
            )
        return []

    versions: list[int] = []
    for line in result.stdout.splitlines():
        match = VERSION_DIR_RE.match(line.strip())
        if match:
            versions.append(int(match.group("version")))
    return sorted(set(versions))


def read_remote_json(
    *,
    remote: str,
    bucket: str,
    object_key: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    result = run_command(
        ["rclone", "cat", remote_object_path(remote, bucket, object_key)],
        logger=logger,
        label=f"read remote json {object_key}",
        capture=True,
    )
    return json.loads(result.stdout)


def read_local_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def determine_authoritative_versions(
    family: VersionFamily,
    *,
    remote: str,
    bucket: str,
    logger: logging.Logger,
) -> tuple[str, list[int]]:
    local_versions = set(existing_versions(family.local_root))
    remote_versions = set(
        list_remote_versions(
            remote=remote,
            bucket=bucket,
            prefix=family.object_prefix,
            logger=logger,
        )
    )

    if remote_versions:
        unexpected_local = sorted(local_versions - remote_versions)
        if unexpected_local:
            raise RuntimeError(
                f"{family.label} has local versions not present in object storage: "
                f"{', '.join(f'v{version}' for version in unexpected_local)}. "
                "Either clear the stale local lineage for a fresh start or restore object storage first."
            )
        return "remote", sorted(remote_versions)

    if local_versions:
        return "local", sorted(local_versions)

    return "none", []


def load_json_for_version(
    family: VersionFamily,
    version: int,
    filename: str,
    *,
    source: str,
    remote: str,
    bucket: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    if source == "remote":
        object_key = f"{family.object_prefix.strip('/')}/v{version}/{filename}"
        return read_remote_json(
            remote=remote,
            bucket=bucket,
            object_key=object_key,
            logger=logger,
        )

    if source == "local":
        return read_local_json(family.local_root / f"v{version}" / filename)

    raise RuntimeError(f"No lineage source available for {family.label}")


def parse_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"Expected integer-like value, got boolean {value!r}")
    return int(value)


def normalize_meeting_ids(raw_ids: Any) -> list[str]:
    if not isinstance(raw_ids, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_ids:
        meeting_id = str(raw_value or "").strip()
        if not meeting_id or meeting_id in seen:
            continue
        seen.add(meeting_id)
        normalized.append(meeting_id)
    return sorted(normalized)


def normalize_split_assignments(raw_assignments: Any) -> dict[str, list[str]]:
    assignments: dict[str, list[str]] = {}
    if not isinstance(raw_assignments, dict):
        return assignments

    for split_name in SPLIT_NAMES:
        meeting_ids = normalize_meeting_ids(raw_assignments.get(split_name))
        if meeting_ids:
            assignments[split_name] = meeting_ids
    return assignments


def infer_source_type(
    *,
    family: VersionFamily,
    manifest: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    direct_source_type = str(manifest.get("source_type") or "").strip().lower()
    source_payload = manifest.get("source")
    source_nested = ""
    if isinstance(source_payload, dict):
        source_nested = str(source_payload.get("source_type") or "").strip().lower()

    candidate = direct_source_type or source_nested
    if candidate == "ami":
        return "ami"
    if candidate in {
        "production_feedback",
        "historical_plus_production_feedback",
        "production_feedback_bootstrap",
    }:
        return "production_feedback"
    if candidate == "synthetic":
        return "synthetic"

    if "augmentation" in manifest:
        return "synthetic"

    metadata_source_type = str(metadata.get("source_type") or "").strip().lower()
    if metadata_source_type == "ami":
        return "ami"
    if metadata_source_type == "synthetic":
        return "synthetic"

    return family.default_source_type


def resolve_split_assignments(
    *,
    version: int,
    manifest: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[int | None, dict[str, list[str]]]:
    meeting_state = manifest.get("meeting_state")
    stamp_version = None
    if isinstance(meeting_state, dict):
        stamp_version = parse_optional_int(meeting_state.get("dataset_version_written_to_meetings"))

    metadata_version = parse_optional_int(metadata.get("dataset_version"))
    if stamp_version is None and metadata_version is not None:
        stamp_version = metadata_version

    split_assignments = normalize_split_assignments(metadata.get("meeting_ids"))
    if split_assignments:
        return stamp_version if stamp_version is not None else version, split_assignments

    split_assignments = normalize_split_assignments(
        {
            "train": metadata.get("train_meeting_ids"),
            "val": metadata.get("val_meeting_ids"),
            "test": metadata.get("test_meeting_ids"),
        }
    )
    if split_assignments:
        return stamp_version if stamp_version is not None else version, split_assignments

    rolling_assignments = normalize_split_assignments(
        {
            "val": metadata.get("val_meeting_ids"),
            "test": metadata.get("test_meeting_ids"),
        }
    )
    if rolling_assignments:
        return stamp_version if stamp_version is not None else version, rolling_assignments

    return None, {}


def validate_split_assignments(
    *,
    family: VersionFamily,
    version: int,
    metadata: dict[str, Any],
    split_assignments: dict[str, list[str]],
) -> None:
    seen_meetings: set[str] = set()
    for split_name, meeting_ids in split_assignments.items():
        overlap = seen_meetings.intersection(meeting_ids)
        if overlap:
            overlap_list = ", ".join(sorted(overlap)[:10])
            raise RuntimeError(
                f"{family.label} v{version} assigns the same meeting to multiple splits: {overlap_list}"
            )
        seen_meetings.update(meeting_ids)

    declared_new_ids = normalize_meeting_ids(metadata.get("new_meeting_ids"))
    if declared_new_ids:
        restored_new_ids = sorted(seen_meetings)
        if restored_new_ids != declared_new_ids:
            raise RuntimeError(
                f"{family.label} v{version} has inconsistent split_info.json: "
                "new_meeting_ids does not match the meetings assigned by split."
            )


def build_stored_versions(
    family: VersionFamily,
    *,
    source: str,
    versions: list[int],
    remote: str,
    bucket: str,
    logger: logging.Logger,
) -> list[StoredVersion]:
    stored_versions: list[StoredVersion] = []
    for version in versions:
        manifest = load_json_for_version(
            family,
            version,
            "manifest.json",
            source=source,
            remote=remote,
            bucket=bucket,
            logger=logger,
        )
        metadata = load_json_for_version(
            family,
            version,
            family.metadata_filename,
            source=source,
            remote=remote,
            bucket=bucket,
            logger=logger,
        )
        source_type = infer_source_type(
            family=family,
            manifest=manifest,
            metadata=metadata,
        )
        stamp_version, split_assignments = resolve_split_assignments(
            version=version,
            manifest=manifest,
            metadata=metadata,
        )
        if family.restore_meeting_assignments:
            validate_split_assignments(
                family=family,
                version=version,
                metadata=metadata,
                split_assignments=split_assignments,
            )

        dataset_name = str(manifest.get("dataset_name") or family.dataset_name_fallback).strip()
        object_key = f"{family.object_prefix.strip('/')}/v{version}/manifest.json"
        stored_versions.append(
            StoredVersion(
                family=family,
                version=version,
                manifest=manifest,
                metadata=metadata,
                dataset_name=dataset_name or family.dataset_name_fallback,
                source_type=source_type,
                object_key=object_key,
                meeting_stamp_version=stamp_version,
                split_assignments=split_assignments if family.restore_meeting_assignments else {},
            )
        )
    return stored_versions


def version_dir_complete(family: VersionFamily, version: int) -> bool:
    version_dir = family.local_root / f"v{version}"
    return all((version_dir / relative_path).exists() for relative_path in family.required_files)


def sync_remote_versions_to_local(
    family: VersionFamily,
    versions: list[int],
    *,
    remote: str,
    bucket: str,
    logger: logging.Logger,
    dry_run: bool,
) -> None:
    family.local_root.mkdir(parents=True, exist_ok=True)

    for version in versions:
        if version_dir_complete(family, version):
            continue

        remote_version_path = remote_dir_path(
            remote,
            bucket,
            f"{family.object_prefix.strip('/')}/v{version}",
        )
        local_version_dir = family.local_root / f"v{version}"
        if dry_run:
            logger.info(
                "Dry-run only | would sync %s v%s from %s -> %s",
                family.label,
                version,
                remote_version_path,
                local_version_dir,
            )
            continue

        run_command(
            ["rclone", "copy", remote_version_path, str(local_version_dir), "-P"],
            logger=logger,
            label=f"sync {family.label} v{version} to local block storage",
        )

        if not version_dir_complete(family, version):
            raise RuntimeError(
                f"{family.label} v{version} is still incomplete under {local_version_dir} after sync"
            )


def ensure_local_versions_complete(
    family: VersionFamily,
    versions: list[int],
) -> None:
    for version in versions:
        if not version_dir_complete(family, version):
            raise RuntimeError(
                f"{family.label} v{version} is incomplete under {family.local_root / f'v{version}'}"
            )


def collect_referenced_meeting_ids(stored_versions: list[StoredVersion]) -> list[str]:
    meeting_ids: set[str] = set()
    for stored in stored_versions:
        for split_ids in stored.split_assignments.values():
            meeting_ids.update(split_ids)
    return sorted(meeting_ids)


def fetch_meeting_states(conn, meeting_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not meeting_ids:
        return {}

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT meeting_id, source_type, dataset_version, dataset_split
            FROM meetings
            WHERE meeting_id = ANY(%s)
            """,
            (meeting_ids,),
        )
        rows = cur.fetchall()
    return {row["meeting_id"]: row for row in rows}


def validate_referenced_meetings_present(
    stored_versions: list[StoredVersion],
    *,
    conn,
) -> dict[str, dict[str, Any]]:
    meeting_ids = collect_referenced_meeting_ids(stored_versions)
    meeting_states = fetch_meeting_states(conn, meeting_ids)

    missing = [meeting_id for meeting_id in meeting_ids if meeting_id not in meeting_states]
    if missing:
        sample = ", ".join(missing[:10])
        extra = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
        raise RuntimeError(
            "Stored dataset lineage references meetings that are not present in Postgres: "
            f"{sample}{extra}. "
            "For a true fresh start, clear the stored lineage in object storage/local dataset roots. "
            "Otherwise restore those meetings before replaying dataset lineage."
        )

    return meeting_states


def apply_meeting_assignments(
    stored_versions: list[StoredVersion],
    *,
    conn,
    meeting_states: dict[str, dict[str, Any]],
    logger: logging.Logger,
    dry_run: bool,
) -> None:
    for stored in stored_versions:
        if stored.meeting_stamp_version is None or not stored.split_assignments:
            continue

        for split_name in SPLIT_NAMES:
            meeting_ids = stored.split_assignments.get(split_name, [])
            if not meeting_ids:
                continue

            to_update: list[str] = []
            conflicts: list[str] = []
            for meeting_id in meeting_ids:
                state = meeting_states[meeting_id]
                current_version = state["dataset_version"]
                current_split = state["dataset_split"]

                if current_version is None:
                    if current_split not in (None, split_name):
                        conflicts.append(
                            f"{meeting_id} currently has split={current_split!r} without a dataset_version"
                        )
                    else:
                        to_update.append(meeting_id)
                    continue

                if int(current_version) == stored.meeting_stamp_version and current_split in (None, split_name):
                    if current_split != split_name:
                        to_update.append(meeting_id)
                    continue

                conflicts.append(
                    f"{meeting_id} currently has dataset_version={current_version} dataset_split={current_split!r}"
                )

            if conflicts:
                sample = "; ".join(conflicts[:5])
                extra = "" if len(conflicts) <= 5 else f" (+{len(conflicts) - 5} more)"
                raise RuntimeError(
                    f"Cannot replay {stored.family.label} v{stored.version} for split {split_name}: "
                    f"{sample}{extra}"
                )

            if not to_update:
                logger.info(
                    "Meeting lineage already aligned | family=%s version=v%s split=%s meetings=%s",
                    stored.family.label,
                    stored.version,
                    split_name,
                    len(meeting_ids),
                )
                continue

            if dry_run:
                logger.info(
                    "Dry-run only | would stamp %s meeting(s) for %s v%s split=%s -> dataset_version=%s",
                    len(to_update),
                    stored.family.label,
                    stored.version,
                    split_name,
                    stored.meeting_stamp_version,
                )
            else:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE meetings
                        SET dataset_version = %s,
                            dataset_split = %s
                        WHERE meeting_id = ANY(%s)
                        """,
                        (stored.meeting_stamp_version, split_name, to_update),
                    )
                conn.commit()

            for meeting_id in to_update:
                meeting_states[meeting_id]["dataset_version"] = stored.meeting_stamp_version
                meeting_states[meeting_id]["dataset_split"] = split_name

            logger.info(
                "Replayed meeting lineage | family=%s version=v%s split=%s stamped=%s",
                stored.family.label,
                stored.version,
                split_name,
                len(to_update),
            )


def plan_family_restore(
    family: VersionFamily,
    *,
    remote: str,
    bucket: str,
    logger: logging.Logger,
) -> PlannedFamilyRestore | None:
    source, versions = determine_authoritative_versions(
        family,
        remote=remote,
        bucket=bucket,
        logger=logger,
    )
    if not versions:
        logger.info("No stored %s were found; skipping", family.label)
        return None

    logger.info(
        "Found %s via %s: %s",
        family.label,
        source,
        ", ".join(f"v{version}" for version in versions),
    )
    stored_versions = build_stored_versions(
        family,
        source=source,
        versions=versions,
        remote=remote,
        bucket=bucket,
        logger=logger,
    )
    return PlannedFamilyRestore(
        family=family,
        source=source,
        versions=versions,
        stored_versions=stored_versions,
    )


def materialize_family_restore(
    plan: PlannedFamilyRestore,
    *,
    remote: str,
    bucket: str,
    conn,
    logger: logging.Logger,
    dry_run: bool,
) -> list[StoredVersion]:
    family = plan.family

    if plan.source == "remote":
        sync_remote_versions_to_local(
            family,
            plan.versions,
            remote=remote,
            bucket=bucket,
            logger=logger,
            dry_run=dry_run,
        )
    else:
        ensure_local_versions_complete(family, plan.versions)

    for stored in plan.stored_versions:
        logger.info(
            "Lineage manifest ready | family=%s version=v%s dataset_name=%s source_type=%s stamped_meetings=%s",
            family.label,
            stored.version,
            stored.dataset_name,
            stored.source_type,
            sum(len(ids) for ids in stored.split_assignments.values()),
        )
        if dry_run:
            continue
        ensure_dataset_version_record(
            conn=conn,
            dataset_name=stored.dataset_name,
            stage=stored.family.stage,
            source_type=stored.source_type,
            object_key=stored.object_key,
            manifest_json=stored.manifest,
        )

    return plan.stored_versions


def main() -> int:
    args = parse_args()
    logger = setup_logger(args.log_file)

    families = [
        VersionFamily(
            label="Stage 1 feedback-pool lineage",
            local_root=args.feedback_pool_root.resolve(),
            object_prefix=args.feedback_pool_object_prefix,
            dataset_name_fallback=args.feedback_pool_dataset_name,
            stage="stage1",
            metadata_filename="meeting_ids.json",
            required_files=(
                "manifest.json",
                "meeting_ids.json",
                "feedback_examples.jsonl",
                "profile.json",
                "quality_report.json",
            ),
            default_source_type="production_feedback",
            restore_meeting_assignments=False,
        ),
        VersionFamily(
            label="Stage 1 dataset lineage",
            local_root=args.dataset_root.resolve(),
            object_prefix=args.dataset_object_prefix,
            dataset_name_fallback=args.dataset_name,
            stage="stage1",
            metadata_filename="split_info.json",
            required_files=(
                "manifest.json",
                "split_info.json",
                "train.jsonl",
                "val.jsonl",
                "test.jsonl",
                "profile.json",
                "quality_report.json",
            ),
            default_source_type="production_feedback",
            restore_meeting_assignments=True,
        ),
    ]

    try:
        with get_conn() as conn:
            planned_restores: list[PlannedFamilyRestore] = []
            for family in families:
                plan = plan_family_restore(
                    family,
                    remote=args.rclone_remote,
                    bucket=args.bucket,
                    logger=logger,
                )
                if plan is not None:
                    planned_restores.append(plan)

            meeting_versions = [
                stored
                for plan in planned_restores
                for stored in plan.stored_versions
                if stored.split_assignments
            ]
            meeting_states = validate_referenced_meetings_present(
                meeting_versions,
                conn=conn,
            )

            for plan in planned_restores:
                materialize_family_restore(
                    plan,
                    remote=args.rclone_remote,
                    bucket=args.bucket,
                    conn=conn,
                    logger=logger,
                    dry_run=args.dry_run,
                )
            apply_meeting_assignments(
                meeting_versions,
                conn=conn,
                meeting_states=meeting_states,
                logger=logger,
                dry_run=args.dry_run,
            )
    except Exception:
        logger.exception("Dataset lineage restore failed")
        return 1

    logger.info("Dataset lineage restore completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
