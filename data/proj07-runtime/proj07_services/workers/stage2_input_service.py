#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from proj07_services.common.feedback_common import (
    get_conn,
    upsert_meeting_artifact,
    upload_file,
    write_json,
    write_jsonl,
)
from proj07_services.common.task_service_common import (
    build_logger,
    env_flag,
    env_float,
    env_int,
    load_json,
)
from proj07_services.common.workflow_task_common import (
    TaskHeartbeat,
    WorkflowTaskLease,
    claim_next_workflow_task,
    ensure_workflow_schema,
    make_worker_id,
    mark_task_cancelled,
    mark_task_retry,
    mark_task_succeeded,
)
from proj07_services.pipeline.build_online_inference_payloads import (
    build_reconstructed_segments,
    build_stage2_inputs,
    load_stage1_responses,
    reconstructed_segments_local_path,
    stage1_local_artifact_paths,
    stage2_local_artifact_paths,
)


APP_NAME = "stage2_input_service"
STAGE2_BUILD_TASK = "stage2_build"
STAGE2_FORWARD_TASK = "stage2_forward"


def stage1_response_output_dir(output_root: Path, meeting_id: str, version: int) -> Path:
    return output_root / meeting_id / f"v{version}"


def stage1_response_paths(
    output_root: Path,
    meeting_id: str,
    version: int,
) -> dict[str, Path]:
    out_root = stage1_response_output_dir(output_root, meeting_id, version)
    return {
        "stage1_responses_json": out_root / "responses.json",
        "stage1_responses_jsonl": out_root / "responses.jsonl",
    }


def load_stage1_model_utterances(path: Path) -> list[dict]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")

    utterances = payload.get("utterances")
    if not isinstance(utterances, list):
        raise RuntimeError(f"Expected 'utterances' list in {path}")

    rows = [row for row in utterances if isinstance(row, dict)]
    if len(rows) != len(utterances):
        raise RuntimeError(f"Expected all model utterance rows in {path} to be JSON objects")
    return rows


@dataclass(frozen=True)
class Stage2InputServiceConfig:
    poll_interval_seconds: float = env_float(
        "STAGE2_BUILD_WORKER_POLL_INTERVAL_SECONDS",
        5.0,
    )
    log_dir: Path = Path(
        os.getenv("STAGE2_BUILD_WORKER_LOG_DIR", "/mnt/block/ingest_logs/stage2_input_service")
    )
    stage1_request_root: Path = Path(
        os.getenv("STAGE1_OUTPUT_ROOT", "/mnt/block/user-behaviour/inference_requests/stage1")
    )
    stage1_response_root: Path = Path(
        os.getenv(
            "STAGE1_RESPONSE_ROOT",
            "/mnt/block/user-behaviour/inference_responses/stage1",
        )
    )
    stage2_input_root: Path = Path(
        os.getenv("STAGE2_INPUT_ROOT", "/mnt/block/user-behaviour/inference_requests/stage2")
    )
    segments_root: Path = Path(
        os.getenv("SEGMENTS_ROOT", "/mnt/block/user-behaviour/reconstructed_segments")
    )
    boundary_threshold: float = env_float("STAGE2_BOUNDARY_THRESHOLD", 0.5)
    upload_artifacts: bool = env_flag("STAGE2_UPLOAD_ARTIFACTS", True)
    stage2_object_prefix: str = os.getenv(
        "STAGE2_OBJECT_PREFIX",
        "production/inference_requests/stage2",
    ).strip()
    segments_prefix: str = os.getenv(
        "SEGMENTS_PREFIX",
        "production/reconstructed_segments",
    ).strip()
    heartbeat_interval_seconds: float = env_float(
        "WORKFLOW_TASK_HEARTBEAT_INTERVAL_SECONDS",
        15.0,
    )
    backoff_base_seconds: float = env_float(
        "WORKFLOW_TASK_BACKOFF_BASE_SECONDS",
        15.0,
    )
    backoff_max_seconds: float = env_float(
        "WORKFLOW_TASK_BACKOFF_MAX_SECONDS",
        900.0,
    )
    stage2_forward_max_attempts: int = env_int("STAGE2_FORWARD_MAX_ATTEMPTS", 8)


@dataclass
class Stage2InputService:
    config: Stage2InputServiceConfig
    logger: object
    worker_id: str = field(default_factory=lambda: make_worker_id(APP_NAME))
    required_artifact_types: tuple[str, ...] = field(
        default=(
            "stage2_inputs_jsonl",
            "stage2_inputs_json",
            "reconstructed_segments_json",
        )
    )

    def run_once(self) -> bool:
        conn = get_conn()
        lease: WorkflowTaskLease | None = None
        try:
            lease = claim_next_workflow_task(
                conn,
                task_type=STAGE2_BUILD_TASK,
                worker_id=self.worker_id,
            )
            if lease is None:
                return False

            source_type = self.fetch_meeting_source_type(conn, lease.meeting_id)
            if source_type != "jitsi":
                self.logger.info(
                    "Cancelling non-Jitsi Stage 2 build task | meeting_id=%s | source_type=%s",
                    lease.meeting_id,
                    source_type,
                )
                mark_task_cancelled(
                    conn,
                    lease=lease,
                    worker_id=self.worker_id,
                    error_summary=f"stage2_build is only enabled for jitsi meetings (source_type={source_type})",
                )
                return True

            needs_build, reasons = self.needs_build(
                conn,
                lease.meeting_id,
                lease.artifact_version,
            )
            if not needs_build:
                self.logger.info(
                    "Stage 2 input build already satisfied; marking task succeeded | meeting_id=%s",
                    lease.meeting_id,
                )
                downstream_tasks = [self.make_forward_task(lease.meeting_id, lease.artifact_version)]
                mark_task_succeeded(
                    conn,
                    lease=lease,
                    worker_id=self.worker_id,
                    downstream_tasks=downstream_tasks,
                )
                return True

            self.logger.info(
                "Processing Stage 2 input build task | meeting_id=%s | attempt=%s | reasons=%s",
                lease.meeting_id,
                lease.attempt_number,
                ", ".join(reasons),
            )
            with TaskHeartbeat(
                task_id=lease.task_id,
                worker_id=self.worker_id,
                interval_seconds=self.config.heartbeat_interval_seconds,
                logger=self.logger,
            ):
                self.build_stage2_artifacts(
                    conn,
                    lease.meeting_id,
                    lease.artifact_version,
                )

            downstream_tasks = [self.make_forward_task(lease.meeting_id, lease.artifact_version)]
            mark_task_succeeded(
                conn,
                lease=lease,
                worker_id=self.worker_id,
                downstream_tasks=downstream_tasks,
            )
            return True
        except Exception as exc:
            if lease is not None:
                next_status = mark_task_retry(
                    conn,
                    lease=lease,
                    worker_id=self.worker_id,
                    error_summary=str(exc),
                    stderr_tail=getattr(exc, "stderr_tail", None),
                    backoff_base_seconds=self.config.backoff_base_seconds,
                    backoff_max_seconds=self.config.backoff_max_seconds,
                )
                self.logger.exception(
                    "Stage 2 input build task failed | meeting_id=%s | status=%s",
                    lease.meeting_id,
                    next_status,
                )
                return False
            self.logger.exception("Stage 2 input service loop failed before claiming a task")
            return False
        finally:
            conn.close()

    def fetch_meeting_source_type(self, conn, meeting_id: str) -> str | None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source_type
                FROM meetings
                WHERE meeting_id = %s
                """,
                (meeting_id,),
            )
            row = cur.fetchone()
        return None if row is None else row["source_type"]

    def make_forward_task(self, meeting_id: str, version: int) -> dict:
        return {
            "task_type": STAGE2_FORWARD_TASK,
            "meeting_id": meeting_id,
            "artifact_version": version,
            "payload_json": {
                "task_name": STAGE2_FORWARD_TASK,
                "meeting_id": meeting_id,
                "artifact_version": version,
                "phase": "summarize_pending",
                "enqueued_by": APP_NAME,
                "source_task": STAGE2_BUILD_TASK,
            },
            "max_attempts": self.config.stage2_forward_max_attempts,
            "revive_succeeded": False,
        }

    def needs_build(
        self,
        conn,
        meeting_id: str,
        version: int,
    ) -> tuple[bool, list[str]]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT artifact_type, object_key
                FROM meeting_artifacts
                WHERE meeting_id = %s
                  AND artifact_version = %s
                  AND artifact_type = ANY(%s)
                """,
                (
                    meeting_id,
                    version,
                    list(self.required_artifact_types),
                ),
            )
            rows = cur.fetchall()

        artifact_map = {
            row["artifact_type"]: row["object_key"]
            for row in rows
            if row["object_key"]
        }
        missing_db_refs = [
            artifact_type
            for artifact_type in self.required_artifact_types
            if artifact_type not in artifact_map
        ]
        upload_pending = [
            artifact_type
            for artifact_type, object_key in artifact_map.items()
            if self.config.upload_artifacts and str(object_key).startswith("local://")
        ]

        local_paths = stage2_local_artifact_paths(
            self.config.stage2_input_root,
            meeting_id,
            version,
        )
        local_paths["reconstructed_segments_json"] = reconstructed_segments_local_path(
            self.config.segments_root,
            meeting_id,
            version,
        )
        missing_local_files = [
            artifact_type
            for artifact_type, path in local_paths.items()
            if not path.exists()
        ]

        reasons: list[str] = []
        if missing_db_refs:
            reasons.append(f"missing_db_refs={','.join(missing_db_refs)}")
        if missing_local_files:
            reasons.append(f"missing_local_files={','.join(missing_local_files)}")
        if upload_pending:
            reasons.append(f"upload_pending={','.join(sorted(upload_pending))}")
        return bool(reasons), reasons

    def build_stage2_artifacts(
        self,
        conn,
        meeting_id: str,
        version: int,
    ) -> None:
        request_paths = stage1_local_artifact_paths(
            self.config.stage1_request_root,
            meeting_id,
            version,
        )
        response_paths = stage1_response_paths(
            self.config.stage1_response_root,
            meeting_id,
            version,
        )

        model_utterances_path = request_paths["stage1_model_utterances_json"]
        responses_jsonl_path = response_paths["stage1_responses_jsonl"]
        if not model_utterances_path.exists():
            raise RuntimeError(
                f"Missing Stage 1 model utterances for meeting {meeting_id}: {model_utterances_path}"
            )
        if not responses_jsonl_path.exists():
            raise RuntimeError(
                f"Missing Stage 1 responses jsonl for meeting {meeting_id}: {responses_jsonl_path}"
            )

        utterances = load_stage1_model_utterances(model_utterances_path)
        stage1_responses = load_stage1_responses(responses_jsonl_path, meeting_id)
        stage2_inputs = build_stage2_inputs(
            meeting_id,
            utterances,
            stage1_responses,
            boundary_threshold=self.config.boundary_threshold,
        )
        reconstructed_segments = build_reconstructed_segments(stage2_inputs)

        input_paths = stage2_local_artifact_paths(
            self.config.stage2_input_root,
            meeting_id,
            version,
        )
        reconstructed_path = reconstructed_segments_local_path(
            self.config.segments_root,
            meeting_id,
            version,
        )

        write_jsonl(input_paths["stage2_inputs_jsonl"], stage2_inputs)
        write_json(
            input_paths["stage2_inputs_json"],
            {
                "meeting_id": meeting_id,
                "input_count": len(stage2_inputs),
                "segments": stage2_inputs,
            },
        )
        write_json(reconstructed_path, reconstructed_segments)

        self.register_stage2_artifacts(
            conn=conn,
            meeting_id=meeting_id,
            version=version,
            stage2_inputs_jsonl_path=input_paths["stage2_inputs_jsonl"],
            stage2_inputs_json_path=input_paths["stage2_inputs_json"],
            reconstructed_segments_path=reconstructed_path,
        )

        self.logger.info(
            "Built Stage 2 input artifacts | meeting_id=%s | stage2_inputs=%d",
            meeting_id,
            len(stage2_inputs),
        )

    def register_stage2_artifacts(
        self,
        *,
        conn,
        meeting_id: str,
        version: int,
        stage2_inputs_jsonl_path: Path,
        stage2_inputs_json_path: Path,
        reconstructed_segments_path: Path,
    ) -> None:
        stage2_prefix = f"{self.config.stage2_object_prefix.strip('/')}/{meeting_id}/v{version}"
        stage2_inputs_jsonl_key = f"{stage2_prefix}/inputs.jsonl"
        stage2_inputs_json_key = f"{stage2_prefix}/inputs.json"
        reconstructed_segments_key = (
            f"{self.config.segments_prefix.strip('/')}/{meeting_id}/v{version}.json"
        )

        upsert_meeting_artifact(
            conn,
            meeting_id,
            "stage2_inputs_jsonl",
            f"local://{stage2_inputs_jsonl_path.resolve()}",
            "application/x-ndjson",
            version,
        )
        upsert_meeting_artifact(
            conn,
            meeting_id,
            "stage2_inputs_json",
            f"local://{stage2_inputs_json_path.resolve()}",
            "application/json",
            version,
        )
        upsert_meeting_artifact(
            conn,
            meeting_id,
            "reconstructed_segments_json",
            f"local://{reconstructed_segments_path.resolve()}",
            "application/json",
            version,
        )

        if self.config.upload_artifacts:
            upload_file(stage2_inputs_jsonl_path, stage2_inputs_jsonl_key, self.logger)
            upload_file(stage2_inputs_json_path, stage2_inputs_json_key, self.logger)
            upload_file(reconstructed_segments_path, reconstructed_segments_key, self.logger)

            upsert_meeting_artifact(
                conn,
                meeting_id,
                "stage2_inputs_jsonl",
                stage2_inputs_jsonl_key,
                "application/x-ndjson",
                version,
            )
            upsert_meeting_artifact(
                conn,
                meeting_id,
                "stage2_inputs_json",
                stage2_inputs_json_key,
                "application/json",
                version,
            )
            upsert_meeting_artifact(
                conn,
                meeting_id,
                "reconstructed_segments_json",
                reconstructed_segments_key,
                "application/json",
                version,
            )
        conn.commit()


def validate_config(config: Stage2InputServiceConfig) -> None:
    if config.poll_interval_seconds <= 0:
        raise ValueError("STAGE2_BUILD_WORKER_POLL_INTERVAL_SECONDS must be > 0")
    if config.boundary_threshold < 0 or config.boundary_threshold > 1:
        raise ValueError("STAGE2_BOUNDARY_THRESHOLD must be between 0 and 1")
    if config.heartbeat_interval_seconds <= 0:
        raise ValueError("WORKFLOW_TASK_HEARTBEAT_INTERVAL_SECONDS must be > 0")
    if config.backoff_base_seconds <= 0:
        raise ValueError("WORKFLOW_TASK_BACKOFF_BASE_SECONDS must be > 0")
    if config.backoff_max_seconds < config.backoff_base_seconds:
        raise ValueError(
            "WORKFLOW_TASK_BACKOFF_MAX_SECONDS must be >= WORKFLOW_TASK_BACKOFF_BASE_SECONDS"
        )
    if config.stage2_forward_max_attempts <= 0:
        raise ValueError("STAGE2_FORWARD_MAX_ATTEMPTS must be > 0")


def main() -> None:
    config = Stage2InputServiceConfig()
    validate_config(config)
    logger = build_logger(APP_NAME, config.log_dir)
    schema_conn = get_conn()
    try:
        ensure_workflow_schema(schema_conn)
    finally:
        schema_conn.close()

    service = Stage2InputService(config=config, logger=logger)

    logger.info("Starting %s", APP_NAME)
    logger.info(
        "Stage 2 input service config | poll_interval=%ss stage1_request_root=%s stage1_response_root=%s stage2_input_root=%s segments_root=%s upload=%s worker_id=%s",
        config.poll_interval_seconds,
        config.stage1_request_root.resolve(),
        config.stage1_response_root.resolve(),
        config.stage2_input_root.resolve(),
        config.segments_root.resolve(),
        config.upload_artifacts,
        service.worker_id,
    )

    while True:
        try:
            processed = service.run_once()
            if not processed:
                time.sleep(config.poll_interval_seconds)
        except KeyboardInterrupt:
            logger.info("Stopping %s", APP_NAME)
            return
        except Exception:
            logger.exception("Stage 2 input service loop failed; retrying after sleep")
            time.sleep(config.poll_interval_seconds)


if __name__ == "__main__":
    main()
