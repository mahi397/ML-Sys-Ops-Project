#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from psycopg.types.json import Json

from proj07_services.common.feedback_common import (
    fetch_meeting_utterance_lookup,
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
from proj07_services.pipeline.build_online_inference_payloads import stage2_local_artifact_paths


APP_NAME = "stage2_forward_service"
STAGE2_FORWARD_TASK = "stage2_forward"


def load_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parsed_row = json.loads(stripped)
        if not isinstance(parsed_row, dict):
            raise RuntimeError(f"Expected JSON object rows in {path}")
        rows.append(parsed_row)
    return rows


def stage2_response_output_dir(output_root: Path, meeting_id: str, version: int) -> Path:
    return output_root / meeting_id / f"v{version}"


def stage2_response_paths(
    output_root: Path,
    meeting_id: str,
    version: int,
) -> dict[str, Path]:
    out_root = stage2_response_output_dir(output_root, meeting_id, version)
    return {
        "stage2_responses_json": out_root / "responses.json",
        "stage2_responses_jsonl": out_root / "responses.jsonl",
        "summary_json": out_root / "summary.json",
    }


def post_stage2_payload(
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


def looks_like_stage2_row(payload: dict) -> bool:
    return any(key in payload for key in ("segment_id", "topic_label", "summary_bullets"))


def parse_embedded_json(text: str) -> object | None:
    candidate = text.strip()
    if not candidate:
        return None

    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start_obj = candidate.find("{")
        end_obj = candidate.rfind("}")
        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            try:
                return json.loads(candidate[start_obj:end_obj + 1])
            except json.JSONDecodeError:
                pass
        start_arr = candidate.find("[")
        end_arr = candidate.rfind("]")
        if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
            try:
                return json.loads(candidate[start_arr:end_arr + 1])
            except json.JSONDecodeError:
                pass
    return None


def extract_response_rows(response_payload: object) -> list[dict] | None:
    if isinstance(response_payload, list):
        rows = [row for row in response_payload if isinstance(row, dict)]
        return rows or None

    if isinstance(response_payload, dict):
        if looks_like_stage2_row(response_payload):
            return [response_payload]
        for key in ("responses", "predictions", "results", "items", "summaries", "recap"):
            value = response_payload.get(key)
            if isinstance(value, list):
                rows = [row for row in value if isinstance(row, dict)]
                return rows or None
        for key in ("response", "result", "output", "message", "text"):
            value = response_payload.get(key)
            if isinstance(value, str):
                embedded = parse_embedded_json(value)
                if embedded is not None:
                    return extract_response_rows(embedded)
    return None


def normalize_summary_bullets(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def normalize_stage2_response(
    raw_response: object,
    segment: dict,
) -> dict:
    candidate: dict | None = None

    rows = extract_response_rows(raw_response)
    if rows:
        if len(rows) == 1:
            candidate = rows[0]
        else:
            candidate = next(
                (
                    row
                    for row in rows
                    if row.get("segment_id") == segment["segment_id"]
                ),
                rows[0],
            )
    elif isinstance(raw_response, dict):
        candidate = raw_response

    if candidate is None:
        raise RuntimeError(f"Stage 2 response could not be normalized for segment {segment['segment_id']}")

    topic_label = str(candidate.get("topic_label", "")).strip()
    summary_bullets = normalize_summary_bullets(candidate.get("summary_bullets"))
    if not topic_label:
        raise RuntimeError(f"Stage 2 response missing topic_label for segment {segment['segment_id']}: {candidate}")
    if not summary_bullets:
        raise RuntimeError(
            f"Stage 2 response missing summary_bullets for segment {segment['segment_id']}: {candidate}"
        )

    return {
        "meeting_id": segment["meeting_id"],
        "segment_id": segment["segment_id"],
        "t_start": segment["t_start"],
        "t_end": segment["t_end"],
        "topic_label": topic_label,
        "summary_bullets": summary_bullets,
        "status": str(candidate.get("status", "complete")).strip() or "complete",
        "raw_response": raw_response,
    }


def assemble_recap(
    meeting_id: str,
    stage2_inputs: list[dict[str, Any]],
    stage2_outputs: list[dict[str, Any]],
    *,
    model_name: str,
    model_version: str,
    prompt_version: str,
) -> dict[str, Any]:
    outputs_by_segment = {row["segment_id"]: row for row in stage2_outputs}
    recap_rows: list[dict[str, Any]] = []

    for segment in stage2_inputs:
        output = outputs_by_segment.get(segment["segment_id"])
        if output is None:
            raise RuntimeError(
                f"Missing Stage 2 output for segment_id={segment['segment_id']} in meeting {meeting_id}"
            )
        recap_rows.append(
            {
                "segment_id": segment["segment_id"],
                "t_start": segment["t_start"],
                "t_end": segment["t_end"],
                "topic_label": output["topic_label"],
                "summary_bullets": output["summary_bullets"],
                "status": output["status"],
            }
        )

    return {
        "meeting_id": meeting_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_version": model_version,
        "prompt_version": prompt_version,
        "total_segments": len(recap_rows),
        "recap": recap_rows,
    }


def normalize_saved_stage2_outputs(
    rows: list[dict[str, Any]],
    stage2_inputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    segments_by_id = {
        int(segment["segment_id"]): segment
        for segment in stage2_inputs
    }
    normalized_rows: list[dict[str, Any]] = []

    for row in rows:
        segment_id = row.get("segment_id")
        if segment_id is None:
            continue

        try:
            segment = segments_by_id[int(segment_id)]
        except (KeyError, TypeError, ValueError):
            continue

        topic_label = str(row.get("topic_label", "")).strip()
        summary_bullets = normalize_summary_bullets(row.get("summary_bullets"))
        if not topic_label or not summary_bullets:
            continue

        normalized_rows.append(
            {
                "meeting_id": segment["meeting_id"],
                "segment_id": segment["segment_id"],
                "t_start": segment["t_start"],
                "t_end": segment["t_end"],
                "topic_label": topic_label,
                "summary_bullets": summary_bullets,
                "status": str(row.get("status", "complete")).strip() or "complete",
                "raw_response": row.get("raw_response", row),
            }
        )

    return normalized_rows


def register_recap_outputs(
    conn,
    *,
    meeting_id: str,
    version: int,
    recap_uri: str,
    stage2_inputs: list[dict[str, Any]],
    stage2_outputs: list[dict[str, Any]],
    model_name: str,
    model_version: str,
    prompt_version: str,
) -> None:
    if not stage2_inputs or not stage2_outputs:
        return

    outputs_by_segment_id = {int(row["segment_id"]): row for row in stage2_outputs}
    utterance_lookup = fetch_meeting_utterance_lookup(conn, meeting_id)

    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM summaries
            WHERE meeting_id = %s
              AND summary_type = 'llm_generated'
              AND version = %s
            """,
            (meeting_id, version),
        )
        cur.execute(
            """
            DELETE FROM topic_segments
            WHERE meeting_id = %s
              AND segment_type = 'predicted'
            """,
            (meeting_id,),
        )

        inserted_segment_ids: list[tuple[int, int]] = []
        for segment in stage2_inputs:
            metadata = segment.get("metadata") or {}
            start_idx = metadata.get("start_source_utterance_index")
            end_idx = metadata.get("end_source_utterance_index")
            if start_idx is None or end_idx is None:
                raise RuntimeError(
                    f"Stage 2 input is missing source utterance indexes for segment_id={segment['segment_id']}"
                )

            try:
                start_row = utterance_lookup[int(start_idx)]
                end_row = utterance_lookup[int(end_idx)]
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Could not map Stage 2 segment_id={segment['segment_id']} to utterances "
                    f"(start_index={start_idx}, end_index={end_idx})"
                ) from exc

            output = outputs_by_segment_id.get(int(segment["segment_id"]))
            if output is None:
                raise RuntimeError(
                    f"Missing Stage 2 output for segment_id={segment['segment_id']} in meeting {meeting_id}"
                )

            cur.execute(
                """
                INSERT INTO topic_segments (
                    meeting_id, segment_type, segment_index,
                    start_utterance_id, end_utterance_id,
                    start_time_sec, end_time_sec, topic_label
                )
                VALUES (%s, 'predicted', %s, %s, %s, %s, %s, %s)
                RETURNING topic_segment_id
                """,
                (
                    meeting_id,
                    int(segment["segment_id"]),
                    start_row["utterance_id"],
                    end_row["utterance_id"],
                    float(segment["t_start"]),
                    float(segment["t_end"]),
                    output["topic_label"],
                ),
            )
            inserted_segment_ids.append(
                (int(segment["segment_id"]), cur.fetchone()["topic_segment_id"])
            )

        created_at = datetime.now(timezone.utc)

        cur.execute(
            """
            INSERT INTO summaries (
                meeting_id, summary_type, summary_object_key, created_by_user_id, version, created_at
            )
            VALUES (%s, 'llm_generated', %s, NULL, %s, %s)
            RETURNING summary_id
            """,
            (meeting_id, recap_uri, version, created_at),
        )
        summary_id = cur.fetchone()["summary_id"]

        for segment_id, topic_segment_id in inserted_segment_ids:
            output = outputs_by_segment_id[segment_id]
            cur.execute(
                """
                INSERT INTO segment_summaries (
                    meeting_id, topic_segment_id, summary_id, segment_index,
                    topic_label, summary_bullets, status,
                    model_name, model_version, prompt_version, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    meeting_id,
                    topic_segment_id,
                    summary_id,
                    segment_id,
                    output["topic_label"],
                    Json(output["summary_bullets"]),
                    output["status"],
                    model_name or None,
                    model_version or None,
                    prompt_version or None,
                    created_at,
                ),
            )


@dataclass(frozen=True)
class ForwardServiceConfig:
    poll_interval_seconds: float = env_float(
        "STAGE2_FORWARD_WORKER_POLL_INTERVAL_SECONDS",
        5.0,
    )
    log_dir: Path = Path(
        os.getenv("STAGE2_FORWARD_WORKER_LOG_DIR", "/mnt/block/ingest_logs/stage2_forward_service")
    )
    request_root: Path = Path(
        os.getenv("STAGE2_INPUT_ROOT", "/mnt/block/user-behaviour/inference_requests/stage2")
    )
    response_root: Path = Path(
        os.getenv(
            "STAGE2_RESPONSE_ROOT",
            "/mnt/block/user-behaviour/inference_responses/stage2",
        )
    )
    version: int = env_int("STAGE1_ARTIFACT_VERSION", 1)
    endpoint_url: str = os.getenv("STAGE2_FORWARD_URL", "").strip()
    payload_format: str = os.getenv(
        "STAGE2_FORWARD_PAYLOAD_FORMAT",
        "segment_json",
    ).strip()
    timeout_seconds: int = env_int("STAGE2_FORWARD_TIMEOUT_SECONDS", 300)
    response_prefix: str = os.getenv(
        "STAGE2_RESPONSE_PREFIX",
        "production/inference_responses/stage2",
    ).strip()
    model_name: str = os.getenv("STAGE2_MODEL_NAME", "stage2-summarizer").strip()
    model_version: str = os.getenv("STAGE2_MODEL_VERSION", "").strip()
    prompt_version: str = os.getenv("STAGE2_PROMPT_VERSION", "").strip()
    upload_artifacts: bool = env_flag("STAGE2_FORWARD_UPLOAD_ARTIFACTS", True)
    retraining_validity_window_size: int = env_int(
        "RETRAINING_DATASET_WINDOW_SIZE",
        env_int("STAGE1_WINDOW_SIZE", 7),
    )
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


@dataclass
class Stage2ForwardService:
    config: ForwardServiceConfig
    logger: object
    worker_id: str = field(default_factory=lambda: make_worker_id(APP_NAME))

    def count_model_utterances(self, stage2_inputs: list[dict[str, Any]]) -> int:
        total = 0
        for segment in stage2_inputs:
            total_utterances = segment.get("total_utterances")
            if isinstance(total_utterances, int):
                total += total_utterances
                continue

            utterances = segment.get("utterances")
            if isinstance(utterances, list):
                total += len(utterances)
        return total

    def run_once(self) -> bool:
        conn = get_conn()
        lease: WorkflowTaskLease | None = None
        try:
            lease = claim_next_workflow_task(
                conn,
                task_type=STAGE2_FORWARD_TASK,
                worker_id=self.worker_id,
            )
            if lease is None:
                return False

            source_type = self.fetch_meeting_source_type(conn, lease.meeting_id)
            if source_type != "jitsi":
                self.logger.info(
                    "Cancelling non-Jitsi Stage 2 forward task | meeting_id=%s | source_type=%s",
                    lease.meeting_id,
                    source_type,
                )
                mark_task_cancelled(
                    conn,
                    lease=lease,
                    worker_id=self.worker_id,
                    error_summary=f"stage2_forward is only enabled for jitsi meetings (source_type={source_type})",
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
                        "Stage 2 forward already complete; marking task succeeded | meeting_id=%s",
                        lease.meeting_id,
                    )
                    mark_task_succeeded(
                        conn,
                        lease=lease,
                        worker_id=self.worker_id,
                    )
                    return True
                raise RuntimeError(f"Stage 2 forward task not ready: {reason}")

            self.logger.info(
                "Processing Stage 2 forward task | meeting_id=%s | attempt=%s | reason=%s",
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
                self.forward_stage2_artifact(
                    conn,
                    lease.meeting_id,
                    lease.artifact_version,
                    reason=reason,
                )

            mark_task_succeeded(
                conn,
                lease=lease,
                worker_id=self.worker_id,
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
                    "Stage 2 forward task failed | meeting_id=%s | status=%s",
                    lease.meeting_id,
                    next_status,
                )
                return False
            self.logger.exception("Stage 2 forward service loop failed before claiming a task")
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
                  AND artifact_type = 'stage2_responses_json'
                  AND artifact_version = %s
                """,
                (meeting_id, version),
            )
            response_artifact = cur.fetchone()
            cur.execute(
                """
                SELECT object_key
                FROM meeting_artifacts
                WHERE meeting_id = %s
                  AND artifact_type = 'summary_json'
                  AND artifact_version = %s
                """,
                (meeting_id, version),
            )
            summary_artifact = cur.fetchone()

        if response_artifact is not None:
            object_key = str(response_artifact["object_key"] or "")
            if self.config.upload_artifacts and object_key.startswith("local://"):
                return True, "upload_pending"
            segments: list[dict[str, Any]] = []
            request_paths = stage2_local_artifact_paths(
                self.config.request_root,
                meeting_id,
                version,
            )
            request_json_path = request_paths["stage2_inputs_json"]
            if request_json_path.exists():
                request_payload = load_json(request_json_path)
                segments = request_payload.get("segments", []) if isinstance(request_payload, dict) else []
                if segments and not self.has_materialized_summary(conn, meeting_id, version):
                    return True, "materialize_pending"
            if segments and summary_artifact is None:
                return True, "materialize_pending"
            summary_key = "" if summary_artifact is None else str(summary_artifact["object_key"] or "")
            if segments and self.config.upload_artifacts and summary_key.startswith("local://"):
                return True, "upload_pending"
            return False, "already_forwarded"

        request_paths = stage2_local_artifact_paths(
            self.config.request_root,
            meeting_id,
            version,
        )
        request_json_path = request_paths["stage2_inputs_json"]
        if not request_json_path.exists():
            return False, "waiting_for_local_json"

        return True, "post_pending"

    def has_materialized_summary(self, conn, meeting_id: str, version: int) -> bool:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM summaries
                    WHERE meeting_id = %s
                      AND summary_type = 'llm_generated'
                      AND version = %s
                ) AS has_summary,
                EXISTS (
                    SELECT 1
                    FROM topic_segments
                    WHERE meeting_id = %s
                      AND segment_type = 'predicted'
                ) AS has_topic_segments,
                EXISTS (
                    SELECT 1
                    FROM segment_summaries ss
                    JOIN summaries s ON s.summary_id = ss.summary_id
                    WHERE ss.meeting_id = %s
                      AND s.summary_type = 'llm_generated'
                      AND s.version = %s
                ) AS has_segment_summaries
                """,
                (meeting_id, version, meeting_id, meeting_id, version),
            )
            row = cur.fetchone()

        if row is None:
            return False

        return bool(row["has_summary"] and row["has_topic_segments"] and row["has_segment_summaries"])

    def forward_stage2_artifact(
        self,
        conn,
        meeting_id: str,
        version: int,
        *,
        reason: str,
    ) -> None:
        response_paths = stage2_response_paths(self.config.response_root, meeting_id, version)
        ensure_dir(response_paths["stage2_responses_json"].parent)

        request_paths = stage2_local_artifact_paths(
            self.config.request_root,
            meeting_id,
            version,
        )
        request_jsonl_path = request_paths["stage2_inputs_jsonl"]
        request_json_path = request_paths["stage2_inputs_json"]
        request_json_payload = self.build_request_json_payload(
            meeting_id=meeting_id,
            request_json_path=request_json_path,
            request_jsonl_path=request_jsonl_path,
        )
        self.validate_request_payload(meeting_id, request_json_payload)
        segments = request_json_payload.get("segments", [])

        if reason in {"upload_pending", "materialize_pending"}:
            response_jsonl_path = (
                response_paths["stage2_responses_jsonl"]
                if response_paths["stage2_responses_jsonl"].exists()
                else None
            )
            if not response_paths["stage2_responses_json"].exists():
                raise RuntimeError(
                    f"Cannot resume Stage 2 output registration for {meeting_id}; local response json is missing"
                )
            response_payload = load_json(response_paths["stage2_responses_json"])
            saved_rows = []
            if isinstance(response_payload, dict):
                payload_rows = response_payload.get("responses")
                if isinstance(payload_rows, list):
                    saved_rows = [row for row in payload_rows if isinstance(row, dict)]
            if not saved_rows and response_jsonl_path is not None:
                saved_rows = load_jsonl_rows(response_jsonl_path)

            normalized_outputs = normalize_saved_stage2_outputs(saved_rows, segments)
            self.register_stage2_outputs(
                conn=conn,
                meeting_id=meeting_id,
                version=version,
                stage2_inputs=segments,
                stage2_outputs=normalized_outputs,
                response_json_path=response_paths["stage2_responses_json"],
                response_jsonl_path=response_jsonl_path,
                summary_json_path=response_paths["summary_json"],
            )
            return

        if not segments:
            response_payload = {
                "meeting_id": meeting_id,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "endpoint_url": self.config.endpoint_url,
                "payload_format": self.config.payload_format,
                "input_count": 0,
                "status": "skipped",
                "reason": "stage2_inputs_empty",
            }
            write_json(response_paths["stage2_responses_json"], response_payload)
            self.register_stage2_outputs(
                conn=conn,
                meeting_id=meeting_id,
                version=version,
                stage2_inputs=segments,
                stage2_outputs=[],
                response_json_path=response_paths["stage2_responses_json"],
                response_jsonl_path=None,
                summary_json_path=response_paths["summary_json"],
            )
            self.logger.info(
                "Skipped Stage 2 forward because inputs are empty | meeting_id=%s",
                meeting_id,
            )
            return

        if self.config.payload_format == "segment_json":
            response_rows: list[dict] = []
            raw_results: list[dict] = []
            for segment in segments:
                status_code, response_text = post_stage2_payload(
                    url=self.config.endpoint_url,
                    body=json.dumps(segment).encode("utf-8"),
                    content_type="application/json",
                    timeout_seconds=self.config.timeout_seconds,
                )
                parsed_response = parse_response_body(response_text)
                response_rows.append(
                    normalize_stage2_response(
                        parsed_response,
                        segment,
                    )
                )
                raw_results.append(
                    {
                        "segment_id": segment["segment_id"],
                        "http_status": status_code,
                        "response": parsed_response,
                    }
                )

            response_payload = {
                "meeting_id": meeting_id,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "endpoint_url": self.config.endpoint_url,
                "payload_format": self.config.payload_format,
                "input_count": len(segments),
                "status": "submitted",
                "response_count": len(response_rows),
                "responses": response_rows,
                "raw_results": raw_results,
            }
            write_json(response_paths["stage2_responses_json"], response_payload)

            response_jsonl_path = response_paths["stage2_responses_jsonl"]
            write_jsonl(response_jsonl_path, response_rows)

            self.register_stage2_outputs(
                conn=conn,
                meeting_id=meeting_id,
                version=version,
                stage2_inputs=segments,
                stage2_outputs=response_rows,
                response_json_path=response_paths["stage2_responses_json"],
                response_jsonl_path=response_jsonl_path,
                summary_json_path=response_paths["summary_json"],
            )
            return

        if self.config.payload_format == "inputs_json":
            request_body = json.dumps(request_json_payload).encode("utf-8")
            content_type = "application/json"
        elif self.config.payload_format == "inputs_jsonl":
            if not request_jsonl_path.exists():
                raise RuntimeError(f"Missing Stage 2 jsonl request payload: {request_jsonl_path}")
            request_body = request_jsonl_path.read_text(encoding="utf-8").encode("utf-8")
            content_type = "application/x-ndjson"
        else:
            raise RuntimeError(
                "STAGE2_FORWARD_PAYLOAD_FORMAT must be 'segment_json', 'inputs_json', or 'inputs_jsonl'"
            )

        status_code, response_text = post_stage2_payload(
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
            "input_count": len(segments),
            "http_status": status_code,
            "status": "submitted",
            "response": parsed_response,
        }
        write_json(response_paths["stage2_responses_json"], response_payload)

        response_jsonl_path: Path | None = None
        if response_rows:
            response_jsonl_path = response_paths["stage2_responses_jsonl"]
            write_jsonl(response_jsonl_path, response_rows)

        normalized_outputs = normalize_saved_stage2_outputs(response_rows or [], segments)
        self.register_stage2_outputs(
            conn=conn,
            meeting_id=meeting_id,
            version=version,
            stage2_inputs=segments,
            stage2_outputs=normalized_outputs,
            response_json_path=response_paths["stage2_responses_json"],
            response_jsonl_path=response_jsonl_path,
            summary_json_path=response_paths["summary_json"],
        )

    def build_request_json_payload(
        self,
        *,
        meeting_id: str,
        request_json_path: Path,
        request_jsonl_path: Path,
    ) -> dict:
        if request_json_path.exists():
            payload = load_json(request_json_path)
            if isinstance(payload, dict):
                return payload

        rows = load_jsonl_rows(request_jsonl_path)
        return {
            "meeting_id": meeting_id,
            "input_count": len(rows),
            "segments": rows,
        }

    def validate_request_payload(self, meeting_id: str, request_payload: dict) -> None:
        if request_payload.get("meeting_id") != meeting_id:
            raise RuntimeError(f"Stage 2 request meeting_id mismatch for {meeting_id}")

        segments = request_payload.get("segments")
        if not isinstance(segments, list):
            raise RuntimeError("Stage 2 request payload must include a 'segments' list")

        for index, row in enumerate(segments, start=1):
            if not isinstance(row, dict):
                raise RuntimeError(f"Stage 2 segment row {index} is not a JSON object")
            if row.get("meeting_id") != meeting_id:
                raise RuntimeError(
                    f"Stage 2 segment row {index} meeting_id mismatch for {meeting_id}"
                )
            if "segment_id" not in row or "utterances" not in row:
                raise RuntimeError(
                    f"Stage 2 segment row {index} is missing required keys"
                )

    def register_stage2_outputs(
        self,
        *,
        conn,
        meeting_id: str,
        version: int,
        stage2_inputs: list[dict[str, Any]],
        stage2_outputs: list[dict[str, Any]],
        response_json_path: Path,
        response_jsonl_path: Path | None,
        summary_json_path: Path,
    ) -> None:
        response_prefix = f"{self.config.response_prefix.strip('/')}/{meeting_id}/v{version}"
        response_json_key = f"{response_prefix}/responses.json"
        response_jsonl_key = f"{response_prefix}/responses.jsonl"
        summary_json_key = f"{response_prefix}/summary.json"

        if stage2_inputs and len(stage2_outputs) != len(stage2_inputs):
            raise RuntimeError(
                f"Stage 2 output count mismatch for meeting {meeting_id}: "
                f"inputs={len(stage2_inputs)} outputs={len(stage2_outputs)}"
            )

        recap_payload: dict[str, Any] | None = None
        if stage2_inputs and stage2_outputs:
            recap_payload = assemble_recap(
                meeting_id,
                stage2_inputs,
                stage2_outputs,
                model_name=self.config.model_name,
                model_version=self.config.model_version,
                prompt_version=self.config.prompt_version,
            )
            write_json(summary_json_path, recap_payload)

        upsert_meeting_artifact(
            conn,
            meeting_id,
            "stage2_responses_json",
            f"local://{response_json_path.resolve()}",
            "application/json",
            version,
        )

        if response_jsonl_path is not None:
            upsert_meeting_artifact(
                conn,
                meeting_id,
                "stage2_responses_jsonl",
                f"local://{response_jsonl_path.resolve()}",
                "application/x-ndjson",
                version,
            )

        if recap_payload is not None:
            upsert_meeting_artifact(
                conn,
                meeting_id,
                "summary_json",
                f"local://{summary_json_path.resolve()}",
                "application/json",
                version,
            )

        if self.config.upload_artifacts:
            upload_file(response_json_path, response_json_key, self.logger)
            upsert_meeting_artifact(
                conn,
                meeting_id,
                "stage2_responses_json",
                response_json_key,
                "application/json",
                version,
            )

            if response_jsonl_path is not None:
                upload_file(response_jsonl_path, response_jsonl_key, self.logger)
                upsert_meeting_artifact(
                    conn,
                    meeting_id,
                    "stage2_responses_jsonl",
                    response_jsonl_key,
                    "application/x-ndjson",
                    version,
                )

            if recap_payload is not None:
                upload_file(summary_json_path, summary_json_key, self.logger)
                upsert_meeting_artifact(
                    conn,
                    meeting_id,
                    "summary_json",
                    summary_json_key,
                    "application/json",
                    version,
                )

        if recap_payload is not None:
            recap_uri = summary_json_key if self.config.upload_artifacts else f"local://{summary_json_path.resolve()}"
            register_recap_outputs(
                conn,
                meeting_id=meeting_id,
                version=version,
                recap_uri=recap_uri,
                stage2_inputs=stage2_inputs,
                stage2_outputs=stage2_outputs,
                model_name=self.config.model_name,
                model_version=self.config.model_version,
                prompt_version=self.config.prompt_version,
            )

        model_utterance_count = self.count_model_utterances(stage2_inputs)
        computed_is_valid = bool(recap_payload) and (
            model_utterance_count >= self.config.retraining_validity_window_size
        )
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE meetings
                SET is_valid = %s
                WHERE meeting_id = %s
                """,
                (computed_is_valid, meeting_id),
            )
        if not computed_is_valid:
            self.logger.info(
                "Marked meeting invalid after Stage 2 materialization | meeting_id=%s model_utterances=%s threshold=%s summary_created=%s",
                meeting_id,
                model_utterance_count,
                self.config.retraining_validity_window_size,
                bool(recap_payload),
            )
        conn.commit()


def validate_config(config: ForwardServiceConfig) -> None:
    if config.poll_interval_seconds <= 0:
        raise ValueError("STAGE2_FORWARD_WORKER_POLL_INTERVAL_SECONDS must be > 0")
    if not config.endpoint_url:
        raise ValueError("STAGE2_FORWARD_URL cannot be empty")
    if config.timeout_seconds <= 0:
        raise ValueError("STAGE2_FORWARD_TIMEOUT_SECONDS must be > 0")
    if config.payload_format not in {"segment_json", "inputs_json", "inputs_jsonl"}:
        raise ValueError(
            "STAGE2_FORWARD_PAYLOAD_FORMAT must be 'segment_json', 'inputs_json', or 'inputs_jsonl'"
        )
    if config.heartbeat_interval_seconds <= 0:
        raise ValueError("WORKFLOW_TASK_HEARTBEAT_INTERVAL_SECONDS must be > 0")
    if config.backoff_base_seconds <= 0:
        raise ValueError("WORKFLOW_TASK_BACKOFF_BASE_SECONDS must be > 0")
    if config.backoff_max_seconds < config.backoff_base_seconds:
        raise ValueError(
            "WORKFLOW_TASK_BACKOFF_MAX_SECONDS must be >= WORKFLOW_TASK_BACKOFF_BASE_SECONDS"
        )
    if config.retraining_validity_window_size <= 0:
        raise ValueError("RETRAINING_DATASET_WINDOW_SIZE must be > 0")


def main() -> None:
    config = ForwardServiceConfig()
    validate_config(config)
    logger = build_logger(APP_NAME, config.log_dir)
    schema_conn = get_conn()
    try:
        ensure_workflow_schema(schema_conn)
    finally:
        schema_conn.close()

    service = Stage2ForwardService(config=config, logger=logger)

    logger.info("Starting %s", APP_NAME)
    logger.info(
        "Stage 2 forward config | poll_interval=%ss request_root=%s response_root=%s url=%s format=%s upload=%s validity_window=%s worker_id=%s",
        config.poll_interval_seconds,
        config.request_root.resolve(),
        config.response_root.resolve(),
        config.endpoint_url,
        config.payload_format,
        config.upload_artifacts,
        config.retraining_validity_window_size,
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
            logger.exception("Stage 2 forward service loop failed; retrying after sleep")
            time.sleep(config.poll_interval_seconds)


if __name__ == "__main__":
    main()
