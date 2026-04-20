#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

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
    ensure_dir,
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
from proj07_services.pipeline.build_online_inference_payloads import stage1_local_artifact_paths


APP_NAME = "stage1_forward_service"
STAGE1_FORWARD_TASK = "stage1_forward"
STAGE2_BUILD_TASK = "stage2_build"


@dataclass(frozen=True)
class ForwardServiceConfig:
    poll_interval_seconds: float = env_float(
        "STAGE1_FORWARD_WORKER_POLL_INTERVAL_SECONDS",
        5.0,
    )
    log_dir: Path = Path(
        os.getenv("STAGE1_FORWARD_WORKER_LOG_DIR", "/mnt/block/ingest_logs/stage1_forward_service")
    )
    request_root: Path = Path(
        os.getenv("STAGE1_OUTPUT_ROOT", "/mnt/block/user-behaviour/inference_requests/stage1")
    )
    response_root: Path = Path(
        os.getenv(
            "STAGE1_RESPONSE_ROOT",
            "/mnt/block/user-behaviour/inference_responses/stage1",
        )
    )
    version: int = env_int("STAGE1_ARTIFACT_VERSION", 1)
    endpoint_url: str = os.getenv(
        "STAGE1_FORWARD_URL",
        "http://192.5.86.194:8000/segment",
    ).strip()
    payload_format: str = os.getenv(
        "STAGE1_FORWARD_PAYLOAD_FORMAT",
        "requests_json",
    ).strip()
    timeout_seconds: int = env_int("STAGE1_FORWARD_TIMEOUT_SECONDS", 300)
    response_prefix: str = os.getenv(
        "STAGE1_RESPONSE_PREFIX",
        "production/inference_responses/stage1",
    ).strip()
    upload_artifacts: bool = env_flag("STAGE1_FORWARD_UPLOAD_ARTIFACTS", True)
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
    stage2_build_max_attempts: int = env_int("STAGE2_BUILD_MAX_ATTEMPTS", 8)


def load_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Expected JSON object rows in {path}")
        rows.append(parsed)
    return rows


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


def post_stage1_payload(
    *,
    url: str,
    body: bytes,
    content_type: str,
    timeout_seconds: int,
) -> tuple[int, str]:
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            status_code = getattr(response, "status", response.getcode())
            return int(status_code), response.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} calling {url}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc}") from exc


def parse_response_body(body_text: str) -> object:
    if not body_text.strip():
        return {"raw_text": "", "content_type": "empty"}
    try:
        return json.loads(body_text)
    except json.JSONDecodeError:
        return {"raw_text": body_text}


def extract_response_rows(response_payload: object) -> list[dict] | None:
    if isinstance(response_payload, list):
        rows = [row for row in response_payload if isinstance(row, dict)]
        return rows or None

    if isinstance(response_payload, dict):
        for key in ("responses", "predictions", "results", "items"):
            value = response_payload.get(key)
            if isinstance(value, list):
                rows = [row for row in value if isinstance(row, dict)]
                return rows or None
    return None


def load_stage1_manifest(request_root: Path, meeting_id: str, version: int) -> dict:
    request_paths = stage1_local_artifact_paths(request_root, meeting_id, version)
    manifest_path = request_paths["stage1_manifest_json"]
    if not manifest_path.exists():
        return {}
    payload = load_json(manifest_path)
    if isinstance(payload, dict):
        return payload
    return {}


@dataclass
class Stage1ForwardService:
    config: ForwardServiceConfig
    logger: object
    worker_id: str = field(default_factory=lambda: make_worker_id(APP_NAME))

    def run_once(self) -> bool:
        conn = get_conn()
        lease: WorkflowTaskLease | None = None
        try:
            lease = claim_next_workflow_task(
                conn,
                task_type=STAGE1_FORWARD_TASK,
                worker_id=self.worker_id,
            )
            if lease is None:
                return False

            source_type = self.fetch_meeting_source_type(conn, lease.meeting_id)
            if source_type != "jitsi":
                self.logger.info(
                    "Cancelling non-Jitsi Stage 1 forward task | meeting_id=%s | source_type=%s",
                    lease.meeting_id,
                    source_type,
                )
                mark_task_cancelled(
                    conn,
                    lease=lease,
                    worker_id=self.worker_id,
                    error_summary=f"stage1_forward is only enabled for jitsi meetings (source_type={source_type})",
                )
                return True

            should_process, reason = self.should_process(
                conn,
                lease.meeting_id,
                lease.artifact_version,
            )
            if not should_process:
                if reason == "already_forwarded":
                    self.logger.info(
                        "Stage 1 forward already complete; marking task succeeded | meeting_id=%s",
                        lease.meeting_id,
                    )
                    mark_task_succeeded(
                        conn,
                        lease=lease,
                        worker_id=self.worker_id,
                    )
                    return True
                raise RuntimeError(f"Stage 1 forward task not ready: {reason}")

            self.logger.info(
                "Processing Stage 1 forward task | meeting_id=%s | attempt=%s | reason=%s",
                lease.meeting_id,
                lease.attempt_number,
                reason,
            )
            with TaskHeartbeat(
                task_id=lease.task_id,
                worker_id=self.worker_id,
                interval_seconds=self.config.heartbeat_interval_seconds,
                logger=self.logger,
            ):
                stage2_ready = self.forward_stage1_artifact(
                    conn,
                    lease.meeting_id,
                    lease.artifact_version,
                    reason=reason,
                )

            downstream_tasks = (
                [self.make_stage2_build_task(lease.meeting_id, lease.artifact_version)]
                if stage2_ready
                else None
            )
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
                    "Stage 1 forward task failed | meeting_id=%s | status=%s",
                    lease.meeting_id,
                    next_status,
                )
                return False
            self.logger.exception("Stage 1 forward service loop failed before claiming a task")
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

    def make_stage2_build_task(self, meeting_id: str, version: int) -> dict:
        return {
            "task_type": STAGE2_BUILD_TASK,
            "meeting_id": meeting_id,
            "artifact_version": version,
            "payload_json": {
                "task_name": STAGE2_BUILD_TASK,
                "meeting_id": meeting_id,
                "artifact_version": version,
                "phase": "stage2_build_pending",
                "enqueued_by": APP_NAME,
                "source_task": STAGE1_FORWARD_TASK,
            },
            "max_attempts": self.config.stage2_build_max_attempts,
            "revive_succeeded": False,
        }

    def should_process(
        self,
        conn,
        meeting_id: str,
        version: int,
    ) -> tuple[bool, str]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT object_key
                FROM meeting_artifacts
                WHERE meeting_id = %s
                  AND artifact_type = 'stage1_responses_json'
                  AND artifact_version = %s
                """,
                (meeting_id, version),
            )
            response_artifact = cur.fetchone()

        if response_artifact is not None:
            object_key = str(response_artifact["object_key"] or "")
            if self.config.upload_artifacts and object_key.startswith("local://"):
                return True, "upload_pending"
            return False, "already_forwarded"

        request_paths = stage1_local_artifact_paths(
            self.config.request_root,
            meeting_id,
            version,
        )
        request_jsonl_path = request_paths["stage1_requests_jsonl"]
        if not request_jsonl_path.exists():
            return False, "waiting_for_local_jsonl"

        return True, "post_pending"

    def forward_stage1_artifact(
        self,
        conn,
        meeting_id: str,
        version: int,
        *,
        reason: str,
    ) -> bool:
        response_paths = stage1_response_paths(self.config.response_root, meeting_id, version)
        ensure_dir(response_paths["stage1_responses_json"].parent)

        if reason == "upload_pending":
            response_jsonl_path = (
                response_paths["stage1_responses_jsonl"]
                if response_paths["stage1_responses_jsonl"].exists()
                else None
            )
            if not response_paths["stage1_responses_json"].exists():
                raise RuntimeError(
                    f"Cannot resume Stage 1 response upload for {meeting_id}; local response json is missing"
                )
            self.register_response_artifacts(
                conn=conn,
                meeting_id=meeting_id,
                version=version,
                response_json_path=response_paths["stage1_responses_json"],
                response_jsonl_path=response_jsonl_path,
            )
            return response_jsonl_path is not None

        request_paths = stage1_local_artifact_paths(
            self.config.request_root,
            meeting_id,
            version,
        )
        request_jsonl_path = request_paths["stage1_requests_jsonl"]
        request_json_path = request_paths["stage1_requests_json"]
        manifest_payload = load_stage1_manifest(
            self.config.request_root,
            meeting_id,
            version,
        )
        request_rows = load_jsonl_rows(request_jsonl_path)
        self.validate_request_rows(meeting_id, request_rows)

        if not request_rows:
            response_payload = {
                "meeting_id": meeting_id,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "endpoint_url": self.config.endpoint_url,
                "payload_format": self.config.payload_format,
                "request_count": 0,
                "status": "skipped",
                "reason": manifest_payload.get("stage1_skip_reason", "stage1_requests_empty"),
                "inference_status": manifest_payload.get("stage1_inference_status"),
                "meeting_flags": manifest_payload.get("meeting_flags", []),
                "recap_notice": manifest_payload.get("recap_notice"),
            }
            write_json(response_paths["stage1_responses_json"], response_payload)
            self.register_response_artifacts(
                conn=conn,
                meeting_id=meeting_id,
                version=version,
                response_json_path=response_paths["stage1_responses_json"],
                response_jsonl_path=None,
            )
            self.logger.info(
                "Skipped Stage 1 forward because requests are empty | meeting_id=%s",
                meeting_id,
            )
            return False

        request_json_payload = self.build_request_json_payload(
            meeting_id=meeting_id,
            request_rows=request_rows,
            request_json_path=request_json_path,
        )
        raw_jsonl_text = request_jsonl_path.read_text(encoding="utf-8")

        if self.config.payload_format == "requests_json":
            request_body = json.dumps(request_json_payload).encode("utf-8")
            content_type = "application/json"
        elif self.config.payload_format == "requests_jsonl":
            request_body = raw_jsonl_text.encode("utf-8")
            content_type = "application/x-ndjson"
        else:
            raise RuntimeError(
                "STAGE1_FORWARD_PAYLOAD_FORMAT must be 'requests_json' or 'requests_jsonl'"
            )

        status_code, response_text = post_stage1_payload(
            url=self.config.endpoint_url,
            body=request_body,
            content_type=content_type,
            timeout_seconds=self.config.timeout_seconds,
        )
        parsed_response = parse_response_body(response_text)
        response_rows = extract_response_rows(parsed_response)

        response_payload = {
            "meeting_id": meeting_id,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "endpoint_url": self.config.endpoint_url,
            "payload_format": self.config.payload_format,
            "request_count": len(request_rows),
            "http_status": status_code,
            "status": "submitted",
            "inference_status": manifest_payload.get("stage1_inference_status"),
            "meeting_flags": manifest_payload.get("meeting_flags", []),
            "recap_notice": manifest_payload.get("recap_notice"),
            "response": parsed_response,
        }
        write_json(response_paths["stage1_responses_json"], response_payload)

        response_jsonl_path: Path | None = None
        if response_rows:
            response_jsonl_path = response_paths["stage1_responses_jsonl"]
            write_jsonl(response_jsonl_path, response_rows)

        self.register_response_artifacts(
            conn=conn,
            meeting_id=meeting_id,
            version=version,
            response_json_path=response_paths["stage1_responses_json"],
            response_jsonl_path=response_jsonl_path,
        )
        return response_jsonl_path is not None

    def build_request_json_payload(
        self,
        *,
        meeting_id: str,
        request_rows: list[dict],
        request_json_path: Path,
    ) -> dict:
        if request_json_path.exists():
            payload = load_json(request_json_path)
            if isinstance(payload, dict):
                return payload
        return {
            "meeting_id": meeting_id,
            "request_count": len(request_rows),
            "requests": request_rows,
        }

    def validate_request_rows(self, meeting_id: str, request_rows: list[dict]) -> None:
        for index, row in enumerate(request_rows, start=1):
            if row.get("meeting_id") != meeting_id:
                raise RuntimeError(
                    f"Stage 1 request row {index} meeting_id mismatch for {meeting_id}"
                )
            if "request_id" not in row or "window" not in row or "transition_index" not in row:
                raise RuntimeError(
                    f"Stage 1 request row {index} is missing required keys"
                )

    def register_response_artifacts(
        self,
        *,
        conn,
        meeting_id: str,
        version: int,
        response_json_path: Path,
        response_jsonl_path: Path | None,
    ) -> None:
        response_prefix = f"{self.config.response_prefix.strip('/')}/{meeting_id}/v{version}"
        response_json_key = f"{response_prefix}/responses.json"
        response_jsonl_key = f"{response_prefix}/responses.jsonl"

        upsert_meeting_artifact(
            conn,
            meeting_id,
            "stage1_responses_json",
            f"local://{response_json_path.resolve()}",
            "application/json",
            version,
        )

        if response_jsonl_path is not None:
            upsert_meeting_artifact(
                conn,
                meeting_id,
                "stage1_responses_jsonl",
                f"local://{response_jsonl_path.resolve()}",
                "application/x-ndjson",
                version,
            )

        if self.config.upload_artifacts:
            upload_file(response_json_path, response_json_key, self.logger)
            upsert_meeting_artifact(
                conn,
                meeting_id,
                "stage1_responses_json",
                response_json_key,
                "application/json",
                version,
            )

            if response_jsonl_path is not None:
                upload_file(response_jsonl_path, response_jsonl_key, self.logger)
                upsert_meeting_artifact(
                    conn,
                    meeting_id,
                    "stage1_responses_jsonl",
                    response_jsonl_key,
                    "application/x-ndjson",
                    version,
                )
        conn.commit()


def validate_config(config: ForwardServiceConfig) -> None:
    if config.poll_interval_seconds <= 0:
        raise ValueError("STAGE1_FORWARD_WORKER_POLL_INTERVAL_SECONDS must be > 0")
    if not config.endpoint_url:
        raise ValueError("STAGE1_FORWARD_URL cannot be empty")
    if config.timeout_seconds <= 0:
        raise ValueError("STAGE1_FORWARD_TIMEOUT_SECONDS must be > 0")
    if config.heartbeat_interval_seconds <= 0:
        raise ValueError("WORKFLOW_TASK_HEARTBEAT_INTERVAL_SECONDS must be > 0")
    if config.backoff_base_seconds <= 0:
        raise ValueError("WORKFLOW_TASK_BACKOFF_BASE_SECONDS must be > 0")
    if config.backoff_max_seconds < config.backoff_base_seconds:
        raise ValueError(
            "WORKFLOW_TASK_BACKOFF_MAX_SECONDS must be >= WORKFLOW_TASK_BACKOFF_BASE_SECONDS"
        )
    if config.stage2_build_max_attempts <= 0:
        raise ValueError("STAGE2_BUILD_MAX_ATTEMPTS must be > 0")


def main() -> None:
    config = ForwardServiceConfig()
    validate_config(config)
    logger = build_logger(APP_NAME, config.log_dir)
    schema_conn = get_conn()
    try:
        ensure_workflow_schema(schema_conn)
    finally:
        schema_conn.close()

    service = Stage1ForwardService(config=config, logger=logger)

    logger.info("Starting %s", APP_NAME)
    logger.info(
        "Forward service config | poll_interval=%ss request_root=%s response_root=%s url=%s format=%s upload=%s worker_id=%s",
        config.poll_interval_seconds,
        config.request_root.resolve(),
        config.response_root.resolve(),
        config.endpoint_url,
        config.payload_format,
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
            logger.exception("Forward service loop failed; retrying after sleep")
            time.sleep(config.poll_interval_seconds)


if __name__ == "__main__":
    main()
