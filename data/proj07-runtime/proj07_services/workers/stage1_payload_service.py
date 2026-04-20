#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from proj07_services.common.feedback_common import get_conn
from proj07_services.common.task_service_common import build_logger, env_flag, env_float, env_int
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
    STAGE1_ARTIFACT_FILES,
    stage1_local_artifact_paths,
)


APP_NAME = "stage1_payload_service"
STAGE1_BUILD_TASK = "stage1_build"
STAGE1_FORWARD_TASK = "stage1_forward"


class TaskExecutionError(RuntimeError):
    def __init__(self, message: str, *, stderr_tail: str | None = None) -> None:
        super().__init__(message)
        self.stderr_tail = stderr_tail


@dataclass(frozen=True)
class BuildServiceConfig:
    poll_interval_seconds: float = env_float(
        "STAGE1_BUILD_WORKER_POLL_INTERVAL_SECONDS",
        5.0,
    )
    log_dir: Path = Path(
        os.getenv("STAGE1_BUILD_WORKER_LOG_DIR", "/mnt/block/ingest_logs/stage1_payload_service")
    )
    output_root: Path = Path(
        os.getenv("STAGE1_OUTPUT_ROOT", "/mnt/block/user-behaviour/inference_requests/stage1")
    )
    window_size: int = env_int("STAGE1_WINDOW_SIZE", 7)
    transition_index: int = env_int("STAGE1_TRANSITION_INDEX", 3)
    min_utterance_chars: int = env_int("STAGE1_MIN_UTTERANCE_CHARS", 20)
    max_words_per_utterance: int = env_int("STAGE1_MAX_WORDS_PER_UTTERANCE", 50)
    min_inference_utterances: int = env_int("STAGE1_MIN_INFERENCE_UTTERANCES", 2)
    short_meeting_max_utterances: int = env_int("STAGE1_SHORT_MEETING_MAX_UTTERANCES", 6)
    upload_artifacts: bool = env_flag("STAGE1_UPLOAD_ARTIFACTS", True)
    object_prefix: str = os.getenv(
        "STAGE1_OBJECT_PREFIX",
        "production/inference_requests/stage1",
    ).strip()
    artifact_version: int = env_int("STAGE1_ARTIFACT_VERSION", 1)
    build_timeout_seconds: int = env_int("STAGE1_BUILD_TIMEOUT_SECONDS", 300)
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
    forward_max_attempts: int = env_int("STAGE1_FORWARD_MAX_ATTEMPTS", 8)


@dataclass
class Stage1PayloadService:
    config: BuildServiceConfig
    logger: object
    worker_id: str = field(default_factory=lambda: make_worker_id(APP_NAME))
    builder_module: str = field(
        default=os.getenv(
            "STAGE1_BUILDER_MODULE",
            "proj07_services.pipeline.build_online_inference_payloads",
        ).strip()
    )
    required_artifact_types: tuple[str, ...] = field(
        default=tuple(STAGE1_ARTIFACT_FILES.keys())
    )

    def run_once(self) -> bool:
        conn = get_conn()
        lease: WorkflowTaskLease | None = None
        try:
            lease = claim_next_workflow_task(
                conn,
                task_type=STAGE1_BUILD_TASK,
                worker_id=self.worker_id,
            )
            if lease is None:
                return False

            source_type = self.fetch_meeting_source_type(conn, lease.meeting_id)
            if source_type != "jitsi":
                self.logger.info(
                    "Cancelling non-Jitsi Stage 1 build task | meeting_id=%s | source_type=%s",
                    lease.meeting_id,
                    source_type,
                )
                mark_task_cancelled(
                    conn,
                    lease=lease,
                    worker_id=self.worker_id,
                    error_summary=f"stage1_build is only enabled for jitsi meetings (source_type={source_type})",
                )
                return True

            needs_build, reasons = self.needs_build(
                conn,
                lease.meeting_id,
                lease.artifact_version,
            )
            downstream_tasks = [self.make_forward_task(lease.meeting_id, lease.artifact_version)]

            if not needs_build:
                self.logger.info(
                    "Stage 1 build already satisfied; marking task succeeded | meeting_id=%s",
                    lease.meeting_id,
                )
                mark_task_succeeded(
                    conn,
                    lease=lease,
                    worker_id=self.worker_id,
                    downstream_tasks=downstream_tasks,
                )
                return True

            self.logger.info(
                "Processing Stage 1 build task | meeting_id=%s | attempt=%s | reasons=%s",
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
                self.run_builder(lease.meeting_id, lease.artifact_version)

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
                    "Stage 1 build task failed | meeting_id=%s | status=%s",
                    lease.meeting_id,
                    next_status,
                )
                return False
            self.logger.exception("Stage 1 build service loop failed before claiming a task")
            return False
        finally:
            conn.close()

    def make_forward_task(self, meeting_id: str, version: int) -> dict:
        return {
            "task_type": STAGE1_FORWARD_TASK,
            "meeting_id": meeting_id,
            "artifact_version": version,
            "payload_json": {
                "task_name": STAGE1_FORWARD_TASK,
                "meeting_id": meeting_id,
                "artifact_version": version,
                "phase": "post_pending",
                "enqueued_by": APP_NAME,
            },
            "max_attempts": self.config.forward_max_attempts,
            "revive_succeeded": False,
        }

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

        local_paths = stage1_local_artifact_paths(
            self.config.output_root,
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
        return bool(reasons), reasons

    def run_builder(self, meeting_id: str, version: int) -> None:
        cmd = [
            sys.executable,
            "-m",
            self.builder_module,
            "--meeting-id",
            meeting_id,
            "--window-size",
            str(self.config.window_size),
            "--transition-index",
            str(self.config.transition_index),
            "--min-utterance-chars",
            str(self.config.min_utterance_chars),
            "--max-words-per-utterance",
            str(self.config.max_words_per_utterance),
            "--min-inference-utterances",
            str(self.config.min_inference_utterances),
            "--short-meeting-max-utterances",
            str(self.config.short_meeting_max_utterances),
            "--output-root",
            str(self.config.output_root),
            "--version",
            str(version),
        ]
        if self.config.upload_artifacts:
            cmd.extend(
                [
                    "--upload-artifacts",
                    "--stage1-object-prefix",
                    self.config.object_prefix,
                ]
            )

        self.logger.info("Running builder: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=self.config.build_timeout_seconds,
        )
        if result.returncode != 0:
            if result.stdout:
                self.logger.error("Builder stdout:\n%s", result.stdout.strip())
            if result.stderr:
                self.logger.error("Builder stderr:\n%s", result.stderr.strip())
            error_message = "Stage 1 builder exited with a non-zero status"
            if result.stderr and result.stderr.strip():
                error_message = result.stderr.strip().splitlines()[-1]
            elif result.stdout and result.stdout.strip():
                error_message = result.stdout.strip().splitlines()[-1]
            raise TaskExecutionError(
                error_message,
                stderr_tail=(result.stderr or result.stdout or "").strip()[-4000:],
            )

        if result.stdout:
            self.logger.info("Builder stdout:\n%s", result.stdout.strip())
        if result.stderr:
            self.logger.info("Builder stderr:\n%s", result.stderr.strip())


def validate_config(config: BuildServiceConfig) -> None:
    if config.poll_interval_seconds <= 0:
        raise ValueError("STAGE1_BUILD_WORKER_POLL_INTERVAL_SECONDS must be > 0")
    if config.build_timeout_seconds <= 0:
        raise ValueError("STAGE1_BUILD_TIMEOUT_SECONDS must be > 0")
    if config.heartbeat_interval_seconds <= 0:
        raise ValueError("WORKFLOW_TASK_HEARTBEAT_INTERVAL_SECONDS must be > 0")
    if config.backoff_base_seconds <= 0:
        raise ValueError("WORKFLOW_TASK_BACKOFF_BASE_SECONDS must be > 0")
    if config.backoff_max_seconds < config.backoff_base_seconds:
        raise ValueError(
            "WORKFLOW_TASK_BACKOFF_MAX_SECONDS must be >= WORKFLOW_TASK_BACKOFF_BASE_SECONDS"
        )
    if config.forward_max_attempts <= 0:
        raise ValueError("STAGE1_FORWARD_MAX_ATTEMPTS must be > 0")
    if not (0 <= config.transition_index < config.window_size):
        raise ValueError(
            "STAGE1_TRANSITION_INDEX must be between 0 and STAGE1_WINDOW_SIZE - 1"
        )


def main() -> None:
    config = BuildServiceConfig()
    validate_config(config)
    logger = build_logger(APP_NAME, config.log_dir)
    schema_conn = get_conn()
    try:
        ensure_workflow_schema(schema_conn)
    finally:
        schema_conn.close()

    service = Stage1PayloadService(config=config, logger=logger)

    logger.info("Starting %s", APP_NAME)
    logger.info(
        "Build service config | poll_interval=%ss output_root=%s upload=%s worker_id=%s",
        config.poll_interval_seconds,
        config.output_root.resolve(),
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
            logger.exception("Build service loop failed; retrying after sleep")
            time.sleep(config.poll_interval_seconds)


if __name__ == "__main__":
    main()
