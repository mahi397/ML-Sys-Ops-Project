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
    upsert_meeting_artifact,
)
from proj07_services.pipeline.ingest_saved_jitsi_transcript import insert_rows as insert_jitsi_rows
from proj07_services.retraining.runtime import existing_versions
from proj07_services.workers.stage2_forward_service import (
    extract_response_rows,
    normalize_saved_stage2_outputs,
    register_recap_outputs,
)


APP_NAME = "restore_dataset_lineage"
VERSION_DIR_RE = re.compile(r"^v(?P<version>\d+)/?$")
VERSIONED_JSON_RE = re.compile(r"^v(?P<version>\d+)\.json$")
JITSI_MEETING_ID_RE = re.compile(r"^jitsi_[A-Za-z0-9_]+$")
SPLIT_NAMES = ("train", "val", "test")
ARTIFACT_CONTENT_TYPES = {
    "raw_transcript": "text/plain",
    "parsed_transcript": "application/json",
    "stage1_requests_jsonl": "application/x-ndjson",
    "stage1_requests_json": "application/json",
    "stage1_model_utterances_json": "application/json",
    "stage1_manifest_json": "application/json",
    "stage1_responses_jsonl": "application/x-ndjson",
    "stage1_responses_json": "application/json",
    "stage2_inputs_jsonl": "application/x-ndjson",
    "stage2_inputs_json": "application/json",
    "reconstructed_segments_json": "application/json",
    "stage2_responses_jsonl": "application/x-ndjson",
    "stage2_responses_json": "application/json",
    "summary_json": "application/json",
}


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


@dataclass(frozen=True)
class MissingMeetingRecoveryConfig:
    parsed_prefix: str
    stage1_request_prefix: str
    stage1_response_prefix: str
    stage2_request_prefix: str
    stage2_response_prefix: str
    reconstructed_segments_prefix: str
    artifact_version: int
    stage2_model_name: str
    stage2_model_version: str
    stage2_prompt_version: str


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
        "--jitsi-parsed-object-prefix",
        default=os.getenv("JITSI_PARSED_OBJECT_PREFIX", "production/jitsi/parsed_transcripts").strip(),
    )
    parser.add_argument(
        "--stage1-object-prefix",
        default=os.getenv("STAGE1_OBJECT_PREFIX", "production/inference_requests/stage1").strip(),
    )
    parser.add_argument(
        "--stage1-response-prefix",
        default=os.getenv("STAGE1_RESPONSE_PREFIX", "production/inference_responses/stage1").strip(),
    )
    parser.add_argument(
        "--stage2-object-prefix",
        default=os.getenv("STAGE2_OBJECT_PREFIX", "production/inference_requests/stage2").strip(),
    )
    parser.add_argument(
        "--stage2-response-prefix",
        default=os.getenv("STAGE2_RESPONSE_PREFIX", "production/inference_responses/stage2").strip(),
    )
    parser.add_argument(
        "--reconstructed-segments-prefix",
        default=os.getenv("SEGMENTS_PREFIX", "production/reconstructed_segments").strip(),
    )
    parser.add_argument(
        "--artifact-version",
        type=int,
        default=int(os.getenv("STAGE1_ARTIFACT_VERSION", "1")),
    )
    parser.add_argument(
        "--stage2-model-name",
        default=os.getenv("STAGE2_MODEL_NAME", "stage2-summarizer").strip(),
    )
    parser.add_argument(
        "--stage2-model-version",
        default=os.getenv("STAGE2_MODEL_VERSION", "").strip(),
    )
    parser.add_argument(
        "--stage2-prompt-version",
        default=os.getenv("STAGE2_PROMPT_VERSION", "").strip(),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("/mnt/block/ingest_logs/retraining_dataset_lineage_restore.log"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def missing_meeting_recovery_config(args: argparse.Namespace) -> MissingMeetingRecoveryConfig:
    return MissingMeetingRecoveryConfig(
        parsed_prefix=args.jitsi_parsed_object_prefix,
        stage1_request_prefix=args.stage1_object_prefix,
        stage1_response_prefix=args.stage1_response_prefix,
        stage2_request_prefix=args.stage2_object_prefix,
        stage2_response_prefix=args.stage2_response_prefix,
        reconstructed_segments_prefix=args.reconstructed_segments_prefix,
        artifact_version=args.artifact_version,
        stage2_model_name=args.stage2_model_name,
        stage2_model_version=args.stage2_model_version,
        stage2_prompt_version=args.stage2_prompt_version,
    )


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


def read_remote_jsonl(
    *,
    remote: str,
    bucket: str,
    object_key: str,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    result = run_command(
        ["rclone", "cat", remote_object_path(remote, bucket, object_key)],
        logger=logger,
        label=f"read remote jsonl {object_key}",
        capture=True,
    )
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def read_local_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def list_remote_files(
    *,
    remote: str,
    bucket: str,
    prefix: str,
    logger: logging.Logger,
) -> list[str]:
    result = run_command(
        ["rclone", "lsf", "--files-only", remote_dir_path(remote, bucket, prefix)],
        logger=logger,
        label=f"list remote files under {prefix}",
        capture=True,
        allow_failure=True,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def remote_object_exists(
    *,
    remote: str,
    bucket: str,
    object_key: str,
    logger: logging.Logger,
) -> bool:
    result = run_command(
        ["rclone", "lsf", remote_object_path(remote, bucket, object_key)],
        logger=logger,
        label=f"check remote object {object_key}",
        capture=True,
        allow_failure=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def content_type_for_artifact(artifact_type: str) -> str:
    return ARTIFACT_CONTENT_TYPES.get(artifact_type, "application/octet-stream")


def find_latest_jitsi_parsed_payload(
    meeting_id: str,
    *,
    remote: str,
    bucket: str,
    config: MissingMeetingRecoveryConfig,
    logger: logging.Logger,
) -> tuple[str, int] | None:
    prefix = f"{config.parsed_prefix.strip('/')}/{meeting_id}"
    files = list_remote_files(
        remote=remote,
        bucket=bucket,
        prefix=prefix,
        logger=logger,
    )
    candidates: list[tuple[int, str]] = []
    for file_name in files:
        match = VERSIONED_JSON_RE.match(file_name)
        if not match:
            continue
        version = int(match.group("version"))
        candidates.append((version, f"{prefix}/{file_name}"))

    if not candidates:
        fallback_key = f"{prefix}/v{config.artifact_version}.json"
        if remote_object_exists(
            remote=remote,
            bucket=bucket,
            object_key=fallback_key,
            logger=logger,
        ):
            return fallback_key, config.artifact_version
        return None

    candidates.sort()
    return candidates[-1][1], candidates[-1][0]


def normalize_parsed_jitsi_payload(
    payload: dict[str, Any],
    *,
    meeting_id: str,
    parsed_object_key: str,
    logger: logging.Logger,
    remote: str,
    bucket: str,
    conn,
) -> tuple[dict[str, Any], int]:
    normalized_payload = json.loads(json.dumps(payload))

    meeting = normalized_payload.get("meeting")
    if not isinstance(meeting, dict):
        raise RuntimeError(f"Parsed transcript payload for {meeting_id} is missing meeting metadata")

    payload_meeting_id = str(meeting.get("meeting_id") or "").strip()
    if payload_meeting_id != meeting_id:
        raise RuntimeError(
            f"Parsed transcript payload meeting_id mismatch: expected {meeting_id}, found {payload_meeting_id or 'none'}"
        )
    
    # Backward-compatible host normalization:
    # historical meetings may not have host metadata.
    # The ingester expects string fields here, not None.
    if not isinstance(normalized_payload.get("host"), dict):
        nested_host = meeting.get("host") if isinstance(meeting.get("host"), dict) else {}

        normalized_payload["host"] = {
            "external_key": str(
                nested_host.get("external_key")
                or meeting.get("host_external_key")
                or ""
            ).strip(),
            "user_id": str(
                nested_host.get("user_id")
                or meeting.get("host_user_id")
                or ""
            ).strip(),
            "display_name": str(
                nested_host.get("display_name")
                or meeting.get("host_display_name")
                or ""
            ).strip(),
            "email": str(
                nested_host.get("email")
                or meeting.get("host_email")
                or ""
            ).strip(),
        }

        if any(normalized_payload["host"].values()):
            logger.warning(
                "Synthesized partial host metadata during restore | meeting_id=%s",
                meeting_id,
            )
        else:
            logger.warning(
                "No host metadata found for historical meeting; restoring with blank host fields | meeting_id=%s",
                meeting_id,
            )

    # Backward-compatible participants normalization:
    # historical meetings may not have explicit participant records.
    # If missing, derive lightweight participant rows from known speakers.
    participants_rows: list[dict[str, Any]] = []

    if isinstance(normalized_payload.get("meeting_participants"), list):
        participants_rows = [
            row for row in normalized_payload["meeting_participants"]
            if isinstance(row, dict)
        ]

    elif isinstance(normalized_payload.get("participants"), list):
        participants_rows = [
            row for row in normalized_payload["participants"]
            if isinstance(row, dict)
        ]

    elif isinstance(meeting.get("participants"), list):
        participants_rows = [
            row for row in meeting["participants"]
            if isinstance(row, dict)
        ]

    else:
        speakers = None

        if isinstance(normalized_payload.get("meeting_speakers"), list):
            speakers = normalized_payload["meeting_speakers"]
        elif isinstance(meeting.get("meeting_speakers"), list):
            speakers = meeting["meeting_speakers"]
        elif isinstance(normalized_payload.get("speakers"), list):
            speakers = normalized_payload["speakers"]
        elif isinstance(meeting.get("speakers"), list):
            speakers = meeting["speakers"]

        if isinstance(speakers, list):
            seen: set[str] = set()

            for speaker in speakers:
                if not isinstance(speaker, dict):
                    continue

                dedupe_key = str(
                    speaker.get("speaker_id")
                    or speaker.get("speaker_label")
                    or speaker.get("display_name")
                    or speaker.get("speaker_name")
                    or ""
                ).strip()

                if dedupe_key and dedupe_key in seen:
                    continue
                if dedupe_key:
                    seen.add(dedupe_key)

                participants_rows.append(
                    {
                        "user_id": speaker.get("user_id"),  # can be None
                        "external_key": speaker.get("external_key"),
                        "display_name": str(
                            speaker.get("display_name")
                            or speaker.get("speaker_name")
                            or speaker.get("speaker_label")
                            or speaker.get("speaker_id")
                            or ""
                        ).strip(),
                        "email": (str(speaker.get("email") or "").strip() or None),
                    }
                )

        if participants_rows:
            logger.warning(
                "Synthesized %d participant row(s) from speakers during restore | meeting_id=%s",
                len(participants_rows),
                meeting_id,
            )
        else:
            logger.warning(
                "No participant metadata found for historical meeting; restoring with empty participant rows | meeting_id=%s",
                meeting_id,
            )

    # Important: preserve both keys because older payloads and current ingester
    # may look for different field names.
    normalized_payload["participants"] = participants_rows
    normalized_payload["meeting_participants"] = participants_rows

    artifact_rows = normalized_payload.get("meeting_artifacts")
    if not isinstance(artifact_rows, list):
        artifact_rows = fetch_meeting_artifacts_from_db(conn, meeting_id)
        if artifact_rows:
            logger.warning(
                "Parsed transcript payload for %s is missing embedded meeting_artifacts; restored %d artifact row(s) from DB",
                meeting_id,
                len(artifact_rows),
            )
        else:
            logger.warning(
                "Parsed transcript payload for %s is missing embedded meeting_artifacts and no DB artifact rows were found",
                meeting_id,
            )
            artifact_rows = []

    verified_base_artifacts: list[dict[str, Any]] = []
    parsed_artifact_version = None

    for artifact in artifact_rows:
        if not isinstance(artifact, dict):
            continue

        artifact_type = str(artifact.get("artifact_type") or "").strip()
        object_key = str(artifact.get("object_key") or "").strip()

        if artifact_type not in {"raw_transcript", "parsed_transcript"}:
            continue

        if artifact_type == "parsed_transcript":
            object_key = parsed_object_key

        if not object_key:
            continue

        if not remote_object_exists(
            remote=remote,
            bucket=bucket,
            object_key=object_key,
            logger=logger,
        ):
            logger.warning(
                "Skipping missing stored artifact while restoring %s | meeting_id=%s | object_key=%s",
                artifact_type,
                meeting_id,
                object_key,
            )
            continue

        normalized_artifact = dict(artifact)
        normalized_artifact["meeting_id"] = meeting_id
        normalized_artifact["object_key"] = object_key
        normalized_artifact["content_type"] = (
            str(artifact.get("content_type") or "").strip()
            or content_type_for_artifact(artifact_type)
        )
        artifact_version = int(artifact.get("artifact_version") or 1)
        normalized_artifact["artifact_version"] = artifact_version

        if artifact_type == "parsed_transcript":
            parsed_artifact_version = artifact_version

        verified_base_artifacts.append(normalized_artifact)

    # Critical fallback:
    # if we already loaded the parsed payload from object storage, synthesize the
    # parsed_transcript artifact row even when payload/DB metadata is missing.
    has_parsed = any(
        str(row.get("artifact_type") or "").strip() == "parsed_transcript"
        for row in verified_base_artifacts
    )

    if not has_parsed:
        if not parsed_object_key:
            raise RuntimeError(
                f"Cannot restore historical Jitsi meeting {meeting_id} because the parsed_transcript artifact is unavailable"
            )

        verified_base_artifacts.append(
            {
                "meeting_id": meeting_id,
                "artifact_type": "parsed_transcript",
                "object_key": parsed_object_key,
                "content_type": content_type_for_artifact("parsed_transcript"),
                "artifact_version": 1,
            }
        )
        parsed_artifact_version = parsed_artifact_version or 1

        logger.warning(
            "Synthesized parsed_transcript artifact metadata during restore | meeting_id=%s object_key=%s",
            meeting_id,
            parsed_object_key,
        )

    normalized_payload["meeting_artifacts"] = verified_base_artifacts
    return normalized_payload, parsed_artifact_version or 1

def candidate_jitsi_artifact_keys(
    meeting_id: str,
    version: int,
    config: MissingMeetingRecoveryConfig,
) -> dict[str, str]:
    stage1_prefix = f"{config.stage1_request_prefix.strip('/')}/{meeting_id}/v{version}"
    stage1_response_prefix = f"{config.stage1_response_prefix.strip('/')}/{meeting_id}/v{version}"
    stage2_prefix = f"{config.stage2_request_prefix.strip('/')}/{meeting_id}/v{version}"
    stage2_response_prefix = f"{config.stage2_response_prefix.strip('/')}/{meeting_id}/v{version}"
    reconstructed_segments_key = (
        f"{config.reconstructed_segments_prefix.strip('/')}/{meeting_id}/v{version}.json"
    )

    return {
        "stage1_requests_jsonl": f"{stage1_prefix}/requests.jsonl",
        "stage1_requests_json": f"{stage1_prefix}/requests.json",
        "stage1_model_utterances_json": f"{stage1_prefix}/model_utterances.json",
        "stage1_manifest_json": f"{stage1_prefix}/manifest.json",
        "stage1_responses_jsonl": f"{stage1_response_prefix}/responses.jsonl",
        "stage1_responses_json": f"{stage1_response_prefix}/responses.json",
        "stage2_inputs_jsonl": f"{stage2_prefix}/inputs.jsonl",
        "stage2_inputs_json": f"{stage2_prefix}/inputs.json",
        "reconstructed_segments_json": reconstructed_segments_key,
        "stage2_responses_jsonl": f"{stage2_response_prefix}/responses.jsonl",
        "stage2_responses_json": f"{stage2_response_prefix}/responses.json",
        "summary_json": f"{stage2_response_prefix}/summary.json",
    }


def restore_verified_artifact_rows(
    conn,
    *,
    meeting_id: str,
    version: int,
    remote: str,
    bucket: str,
    config: MissingMeetingRecoveryConfig,
    logger: logging.Logger,
    dry_run: bool,
) -> dict[str, str]:
    restored_keys: dict[str, str] = {}
    for artifact_type, object_key in candidate_jitsi_artifact_keys(meeting_id, version, config).items():
        if not remote_object_exists(
            remote=remote,
            bucket=bucket,
            object_key=object_key,
            logger=logger,
        ):
            continue

        restored_keys[artifact_type] = object_key
        if dry_run:
            continue

        upsert_meeting_artifact(
            conn,
            meeting_id,
            artifact_type,
            object_key,
            content_type_for_artifact(artifact_type),
            version,
        )

    if not dry_run:
        conn.commit()
    return restored_keys


def restore_materialized_summary_rows(
    conn,
    *,
    meeting_id: str,
    version: int,
    restored_artifacts: dict[str, str],
    remote: str,
    bucket: str,
    config: MissingMeetingRecoveryConfig,
    logger: logging.Logger,
    dry_run: bool,
) -> bool:
    stage2_inputs_key = restored_artifacts.get("stage2_inputs_json")
    response_json_key = restored_artifacts.get("stage2_responses_json")
    summary_json_key = restored_artifacts.get("summary_json")
    if not stage2_inputs_key or not response_json_key or not summary_json_key:
        return False

    stage2_payload = read_remote_json(
        remote=remote,
        bucket=bucket,
        object_key=stage2_inputs_key,
        logger=logger,
    )
    stage2_inputs = stage2_payload.get("segments") if isinstance(stage2_payload, dict) else None
    if not isinstance(stage2_inputs, list) or not stage2_inputs:
        logger.info(
            "Skipping recap materialization because stored Stage 2 inputs are empty | meeting_id=%s",
            meeting_id,
        )
        return False

    response_payload = read_remote_json(
        remote=remote,
        bucket=bucket,
        object_key=response_json_key,
        logger=logger,
    )
    response_rows = extract_response_rows(response_payload) or []
    if not response_rows and "stage2_responses_jsonl" in restored_artifacts:
        response_rows = read_remote_jsonl(
            remote=remote,
            bucket=bucket,
            object_key=restored_artifacts["stage2_responses_jsonl"],
            logger=logger,
        )

    normalized_outputs = normalize_saved_stage2_outputs(response_rows, stage2_inputs)
    if len(normalized_outputs) != len(stage2_inputs):
        logger.warning(
            "Skipping recap materialization because stored Stage 2 outputs are incomplete | meeting_id=%s inputs=%s outputs=%s",
            meeting_id,
            len(stage2_inputs),
            len(normalized_outputs),
        )
        return False

    if dry_run:
        logger.info(
            "Dry-run only | would materialize llm_generated summary rows for restored historical meeting %s",
            meeting_id,
        )
        return True

    register_recap_outputs(
        conn,
        meeting_id=meeting_id,
        version=version,
        recap_uri=summary_json_key,
        stage2_inputs=stage2_inputs,
        stage2_outputs=normalized_outputs,
        model_name=config.stage2_model_name,
        model_version=config.stage2_model_version,
        prompt_version=config.stage2_prompt_version,
    )
    conn.commit()
    return True


def restore_missing_jitsi_meetings(
    meeting_ids: list[str],
    *,
    conn,
    remote: str,
    bucket: str,
    config: MissingMeetingRecoveryConfig,
    logger: logging.Logger,
    dry_run: bool,
) -> set[str]:
    restored_ids: set[str] = set()
    for meeting_id in meeting_ids:
        if not JITSI_MEETING_ID_RE.match(meeting_id):
            continue

        parsed_payload_info = find_latest_jitsi_parsed_payload(
            meeting_id,
            remote=remote,
            bucket=bucket,
            config=config,
            logger=logger,
        )
        if parsed_payload_info is None:
            logger.warning(
                "No stored parsed transcript artifact was found for historical meeting %s",
                meeting_id,
            )
            continue

        parsed_object_key, parsed_version = parsed_payload_info
        parsed_payload = read_remote_json(
            remote=remote,
            bucket=bucket,
            object_key=parsed_object_key,
            logger=logger,
        )
        normalized_payload, artifact_version = normalize_parsed_jitsi_payload(
                parsed_payload,
                meeting_id=meeting_id,
                parsed_object_key=parsed_object_key,
                logger=logger,
                remote=remote,
                bucket=bucket,
                conn=conn,
        )

        logger.info(
            "Restoring historical Jitsi meeting from stored parsed payload | meeting_id=%s parsed_version=v%s artifact_version=%s",
            meeting_id,
            parsed_version,
            artifact_version,
        )

        if not dry_run:
            insert_jitsi_rows(
                conn=conn,
                payload=normalized_payload,
                replace_existing=False,
                artifact_version=artifact_version,
                logger=logger,
            )

        restored_artifacts = restore_verified_artifact_rows(
            conn,
            meeting_id=meeting_id,
            version=artifact_version,
            remote=remote,
            bucket=bucket,
            config=config,
            logger=logger,
            dry_run=dry_run,
        )
        restore_materialized_summary_rows(
            conn,
            meeting_id=meeting_id,
            version=artifact_version,
            restored_artifacts=restored_artifacts,
            remote=remote,
            bucket=bucket,
            config=config,
            logger=logger,
            dry_run=dry_run,
        )
        restored_ids.add(meeting_id)

    if restored_ids:
        logger.info(
            "Historical Jitsi recovery completed | restored=%s",
            ", ".join(sorted(restored_ids)),
        )
    return restored_ids


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

def fetch_meeting_artifacts_from_db(conn, meeting_id: str) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT artifact_type, object_key, content_type, artifact_version, created_at
            FROM meeting_artifacts
            WHERE meeting_id = %s
            ORDER BY artifact_id
            """,
            (meeting_id,),
        )
        rows = cur.fetchall()

    artifacts: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            artifacts.append(
                {
                    "meeting_id": meeting_id,
                    "artifact_type": row["artifact_type"],
                    "object_key": row["object_key"],
                    "content_type": row["content_type"],
                    "artifact_version": int(row.get("artifact_version") or 1),
                    "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
                }
            )
        else:
            artifact_type, object_key, content_type, artifact_version, created_at = row
            artifacts.append(
                {
                    "meeting_id": meeting_id,
                    "artifact_type": artifact_type,
                    "object_key": object_key,
                    "content_type": content_type,
                    "artifact_version": int(artifact_version or 1),
                    "created_at": created_at.isoformat() if created_at else None,
                }
            )

    return artifacts

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
    remote: str,
    bucket: str,
    missing_meeting_config: MissingMeetingRecoveryConfig,
    logger: logging.Logger,
    dry_run: bool,
) -> dict[str, dict[str, Any]]:
    meeting_ids = collect_referenced_meeting_ids(stored_versions)
    meeting_states = fetch_meeting_states(conn, meeting_ids)

    missing = [meeting_id for meeting_id in meeting_ids if meeting_id not in meeting_states]
    if missing:
        restored_ids = restore_missing_jitsi_meetings(
            missing,
            conn=conn,
            remote=remote,
            bucket=bucket,
            config=missing_meeting_config,
            logger=logger,
            dry_run=dry_run,
        )
        if restored_ids:
            if dry_run:
                for meeting_id in restored_ids:
                    meeting_states[meeting_id] = {
                        "meeting_id": meeting_id,
                        "source_type": "jitsi",
                        "dataset_version": None,
                        "dataset_split": None,
                    }
            else:
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
                remote=args.rclone_remote,
                bucket=args.bucket,
                missing_meeting_config=missing_meeting_recovery_config(args),
                logger=logger,
                dry_run=args.dry_run,
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
