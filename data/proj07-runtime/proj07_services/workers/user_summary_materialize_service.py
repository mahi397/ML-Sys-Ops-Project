#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from psycopg.types.json import Json

from proj07_services.common.feedback_common import (
    fetch_meeting_utterance_lookup,
    get_conn,
    upload_file,
    write_json,
)
from proj07_services.common.task_service_common import (
    build_logger,
    env_flag,
    env_float,
    env_int,
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


APP_NAME = "user_summary_materialize_service"
USER_SUMMARY_TASK = "user_summary_materialize"
MATERIALIZABLE_EVENT_TYPES = (
    "merge_segments",
    "split_segment",
    "edit_topic_label",
    "edit_summary_bullets",
)


@dataclass(frozen=True)
class UserSummaryMaterializeConfig:
    poll_interval_seconds: float = env_float(
        "USER_SUMMARY_WORKER_POLL_INTERVAL_SECONDS",
        5.0,
    )
    log_dir: Path = Path(
        os.getenv(
            "USER_SUMMARY_WORKER_LOG_DIR",
            "/mnt/block/ingest_logs/user_summary_materialize_service",
        )
    )
    output_root: Path = Path(
        os.getenv(
            "USER_SUMMARY_OUTPUT_ROOT",
            "/mnt/block/user-behaviour/user_summary_edits",
        )
    )
    object_prefix: str = os.getenv(
        "USER_SUMMARY_OBJECT_PREFIX",
        "production/summary_edits",
    ).strip()
    upload_artifacts: bool = env_flag("USER_SUMMARY_UPLOAD_ARTIFACTS", True)
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


def normalize_summary_bullets(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def user_summary_output_dir(output_root: Path, meeting_id: str, version: int) -> Path:
    return output_root / meeting_id / f"v{version}"


def user_summary_output_path(output_root: Path, meeting_id: str, version: int) -> Path:
    return user_summary_output_dir(output_root, meeting_id, version) / "summary.json"


def build_default_topic_label(segment: dict[str, Any], utterance_lookup: dict[int, dict[str, Any]]) -> str:
    if segment.get("topic_label"):
        return str(segment["topic_label"]).strip()

    start_index = int(segment["start_utterance_index"])
    first_row = utterance_lookup.get(start_index, {})
    text = str(first_row.get("clean_text") or "").strip()
    if not text:
        return f"Edited segment {segment.get('segment_index', '')}".strip()

    words = text.split()[:5]
    if not words:
        return f"Edited segment {segment.get('segment_index', '')}".strip()

    title = " ".join(words)
    return title[:1].upper() + title[1:]


def utterance_rows_for_segment(
    segment: dict[str, Any],
    utterance_lookup: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start_index = int(segment["start_utterance_index"])
    end_index = int(segment["end_utterance_index"])
    for utterance_index in range(start_index, end_index + 1):
        row = utterance_lookup.get(utterance_index)
        if row is None:
            continue
        rows.append(
            {
                "speaker": row.get("speaker_label") or "Speaker",
                "text": row.get("clean_text") or "",
                "start_time_sec": row.get("start_time_sec"),
                "end_time_sec": row.get("end_time_sec"),
            }
        )
    return rows


def build_auto_segment_state(
    segment: dict[str, Any],
    utterance_lookup: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    return {
        **segment,
        "topic_label": str(segment.get("topic_label") or "").strip(),
        "summary_bullets": normalize_summary_bullets(segment.get("summary_bullets")),
        "status": str(segment.get("status") or "draft"),
    }


def normalize_segments(
    segments: list[dict[str, Any]],
    utterance_lookup: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(
        segments,
        key=lambda segment: (
            int(segment["start_utterance_index"]),
            int(segment["end_utterance_index"]),
        ),
    )
    normalized: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(ordered, start=1):
        start_row = utterance_lookup[int(segment["start_utterance_index"])]
        end_row = utterance_lookup[int(segment["end_utterance_index"])]
        normalized.append(
            {
                **segment,
                "segment_index": segment_index,
                "t_start": float(start_row["start_time_sec"]),
                "t_end": float(end_row["end_time_sec"]),
            }
        )
    return normalized


def assemble_user_summary_payload(
    meeting_id: str,
    segments: list[dict[str, Any]],
    *,
    based_on_summary_id: int,
    based_on_summary_type: str,
    edited_by_user_id: str | None,
    edit_session_id: str,
    version: int,
) -> dict[str, Any]:
    return {
        "meeting_id": meeting_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary_type": "user_edited",
        "summary_version": version,
        "based_on_summary_id": based_on_summary_id,
        "based_on_summary_type": based_on_summary_type,
        "edited_by_user_id": edited_by_user_id,
        "edit_session_id": edit_session_id,
        "total_segments": len(segments),
        "recap": [
            {
                "segment_id": int(segment["segment_index"]),
                "t_start": float(segment["t_start"]),
                "t_end": float(segment["t_end"]),
                "topic_label": segment["topic_label"],
                "summary_bullets": segment["summary_bullets"],
                "status": segment["status"],
            }
            for segment in segments
        ],
    }


@dataclass
class UserSummaryMaterializeService:
    config: UserSummaryMaterializeConfig
    logger: object
    worker_id: str = field(default_factory=lambda: make_worker_id(APP_NAME))

    def run_once(self) -> bool:
        conn = get_conn()
        lease: WorkflowTaskLease | None = None
        try:
            lease = claim_next_workflow_task(
                conn,
                task_type=USER_SUMMARY_TASK,
                worker_id=self.worker_id,
            )
            if lease is None:
                return False

            source_type = self.fetch_meeting_source_type(conn, lease.meeting_id)
            if source_type != "jitsi":
                mark_task_cancelled(
                    conn,
                    lease=lease,
                    worker_id=self.worker_id,
                    error_summary=f"user summary materialization is only enabled for jitsi meetings (source_type={source_type})",
                )
                return True

            requested_edit_session_id = ""
            if isinstance(lease.payload_json, dict):
                requested_edit_session_id = str(
                    lease.payload_json.get("edit_session_id") or ""
                ).strip()

            latest_session = (
                self.fetch_pending_edit_session_by_id(
                    conn,
                    lease.meeting_id,
                    requested_edit_session_id,
                )
                if requested_edit_session_id
                else self.fetch_latest_pending_edit_session(conn, lease.meeting_id)
            )
            if latest_session is None:
                self.logger.info(
                    "No pending user-summary edit session found; marking task succeeded | meeting_id=%s",
                    lease.meeting_id,
                )
                mark_task_succeeded(
                    conn,
                    lease=lease,
                    worker_id=self.worker_id,
                )
                return True

            self.logger.info(
                "Materializing user summary | meeting_id=%s | edit_session_id=%s | attempt=%s",
                lease.meeting_id,
                latest_session["edit_session_id"],
                lease.attempt_number,
            )

            with TaskHeartbeat(
                task_id=lease.task_id,
                worker_id=self.worker_id,
                interval_seconds=self.config.heartbeat_interval_seconds,
                logger=self.logger,
            ):
                self.materialize_user_summary(
                    conn,
                    meeting_id=lease.meeting_id,
                    edit_session_id=str(latest_session["edit_session_id"]),
                )

            mark_task_succeeded(
                conn,
                lease=lease,
                worker_id=self.worker_id,
            )
            return True
        except Exception as exc:
            if lease is not None:
                conn.rollback()
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
                    "User summary materialization failed | meeting_id=%s | status=%s",
                    lease.meeting_id,
                    next_status,
                )
                return False
            self.logger.exception("User summary materialization loop failed before claiming a task")
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

    def fetch_latest_user_summary_created_at(self, conn, meeting_id: str) -> Any:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT MAX(created_at) AS created_at
                FROM summaries
                WHERE meeting_id = %s
                  AND summary_type = 'user_edited'
                """,
                (meeting_id,),
            )
            row = cur.fetchone()
        return None if row is None else row["created_at"]

    def fetch_pending_edit_session_by_id(
        self,
        conn,
        meeting_id: str,
        edit_session_id: str,
    ) -> dict[str, Any] | None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    after_payload ->> 'edit_session_id' AS edit_session_id,
                    MAX(feedback_event_id) AS latest_feedback_event_id,
                    BOOL_OR(after_payload ? 'materialized_summary_id') AS is_materialized
                FROM feedback_events
                WHERE meeting_id = %s
                  AND event_type = ANY(%s)
                  AND after_payload ->> 'edit_session_id' = %s
                GROUP BY after_payload ->> 'edit_session_id'
                HAVING NOT BOOL_OR(after_payload ? 'materialized_summary_id')
                """,
                (
                    meeting_id,
                    list(MATERIALIZABLE_EVENT_TYPES),
                    edit_session_id,
                ),
            )
            return cur.fetchone()

    def fetch_latest_pending_edit_session(self, conn, meeting_id: str) -> dict[str, Any] | None:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH edit_sessions AS (
                    SELECT
                        after_payload ->> 'edit_session_id' AS edit_session_id,
                        MAX(feedback_event_id) AS latest_feedback_event_id,
                        BOOL_OR(after_payload ? 'materialized_summary_id') AS is_materialized
                    FROM feedback_events
                    WHERE meeting_id = %s
                      AND event_type = ANY(%s)
                      AND after_payload ? 'edit_session_id'
                    GROUP BY after_payload ->> 'edit_session_id'
                )
                SELECT
                    edit_sessions.edit_session_id,
                    edit_sessions.latest_feedback_event_id
                FROM edit_sessions
                WHERE NOT edit_sessions.is_materialized
                ORDER BY latest_feedback_event_id DESC
                LIMIT 1
                """,
                (
                    meeting_id,
                    list(MATERIALIZABLE_EVENT_TYPES),
                ),
            )
            return cur.fetchone()

    def fetch_edit_events(
        self,
        conn,
        meeting_id: str,
        edit_session_id: str,
    ) -> list[dict[str, Any]]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    feedback_event_id,
                    event_type,
                    segment_summary_id,
                    before_payload,
                    after_payload,
                    created_by_user_id,
                    created_at
                FROM feedback_events
                WHERE meeting_id = %s
                  AND event_type = ANY(%s)
                  AND after_payload ->> 'edit_session_id' = %s
                ORDER BY feedback_event_id
                """,
                (
                    meeting_id,
                    list(MATERIALIZABLE_EVENT_TYPES),
                    edit_session_id,
                ),
            )
            return cur.fetchall() or []

    def fetch_summary_segments(self, conn, summary_id: int) -> list[dict[str, Any]]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    ss.segment_summary_id,
                    ss.segment_index,
                    ss.topic_label,
                    ss.summary_bullets,
                    ss.status,
                    u_start.utterance_index AS start_utterance_index,
                    u_end.utterance_index AS end_utterance_index
                FROM segment_summaries ss
                JOIN topic_segments ts
                    ON ts.topic_segment_id = ss.topic_segment_id
                JOIN utterances u_start
                    ON u_start.utterance_id = ts.start_utterance_id
                JOIN utterances u_end
                    ON u_end.utterance_id = ts.end_utterance_id
                WHERE ss.summary_id = %s
                ORDER BY ss.segment_index
                """,
                (summary_id,),
            )
            rows = cur.fetchall() or []

        return [
            {
                "segment_summary_id": row["segment_summary_id"],
                "segment_index": int(row["segment_index"]),
                "start_utterance_index": int(row["start_utterance_index"]),
                "end_utterance_index": int(row["end_utterance_index"]),
                "topic_label": str(row.get("topic_label") or "").strip(),
                "summary_bullets": normalize_summary_bullets(row.get("summary_bullets")),
                "status": str(row.get("status") or "complete"),
            }
            for row in rows
        ]

    def fetch_latest_summary_id(self, conn, meeting_id: str, summary_type: str) -> int | None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT summary_id
                FROM summaries
                WHERE meeting_id = %s
                  AND summary_type = %s
                ORDER BY version DESC, created_at DESC, summary_id DESC
                LIMIT 1
                """,
                (meeting_id, summary_type),
            )
            row = cur.fetchone()
        return None if row is None else int(row["summary_id"])

    def fetch_next_user_summary_version(self, conn, meeting_id: str) -> int:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(MAX(version), 0) + 1 AS next_version
                FROM summaries
                WHERE meeting_id = %s
                  AND summary_type = 'user_edited'
                """,
                (meeting_id,),
            )
            row = cur.fetchone()
        return int(row["next_version"])

    def apply_split(
        self,
        segments: list[dict[str, Any]],
        utterance_idx: int,
        utterance_lookup: dict[int, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        for index, segment in enumerate(segments):
            start_index = int(segment["start_utterance_index"])
            end_index = int(segment["end_utterance_index"])
            if not (start_index < utterance_idx <= end_index):
                continue

            first_segment = build_auto_segment_state(
                {
                    **segment,
                    "segment_summary_id": None,
                    "start_utterance_index": start_index,
                    "end_utterance_index": utterance_idx - 1,
                },
                utterance_lookup,
            )
            second_segment = build_auto_segment_state(
                {
                    **segment,
                    "segment_summary_id": None,
                    "start_utterance_index": utterance_idx,
                    "end_utterance_index": end_index,
                },
                utterance_lookup,
            )
            next_segments = segments[:index] + [first_segment, second_segment] + segments[index + 1 :]
            return normalize_segments(next_segments, utterance_lookup)
        return segments

    def apply_merge(
        self,
        segments: list[dict[str, Any]],
        utterance_idx: int,
        utterance_lookup: dict[int, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        for index in range(len(segments) - 1):
            current = segments[index]
            following = segments[index + 1]
            if int(current["end_utterance_index"]) != utterance_idx:
                continue
            if int(following["start_utterance_index"]) != utterance_idx + 1:
                continue

            merged_segment = build_auto_segment_state(
                {
                    **current,
                    "segment_summary_id": None,
                    "start_utterance_index": int(current["start_utterance_index"]),
                    "end_utterance_index": int(following["end_utterance_index"]),
                    "topic_label": self.merge_topic_labels(current, following),
                    "summary_bullets": self.merge_summary_bullets(current, following),
                    "status": self.merge_status(current, following),
                },
                utterance_lookup,
            )
            next_segments = segments[:index] + [merged_segment] + segments[index + 2 :]
            return normalize_segments(next_segments, utterance_lookup)
        return segments

    def merge_topic_labels(
        self,
        current: dict[str, Any],
        following: dict[str, Any],
    ) -> str:
        current_label = str(current.get("topic_label") or "").strip()
        following_label = str(following.get("topic_label") or "").strip()
        if not current_label:
            return following_label
        if not following_label or following_label.casefold() == current_label.casefold():
            return current_label
        return f"{current_label} / {following_label}"

    def merge_summary_bullets(
        self,
        current: dict[str, Any],
        following: dict[str, Any],
    ) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for bullet in normalize_summary_bullets(current.get("summary_bullets")) + normalize_summary_bullets(
            following.get("summary_bullets")
        ):
            key = bullet.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(bullet)
        return merged

    def merge_status(
        self,
        current: dict[str, Any],
        following: dict[str, Any],
    ) -> str:
        statuses = {
            str(current.get("status") or "draft").strip(),
            str(following.get("status") or "draft").strip(),
        }
        return "complete" if statuses == {"complete"} else "draft"

    def apply_segment_text_edit(
        self,
        segments: list[dict[str, Any]],
        event_row: dict[str, Any],
        *,
        field_name: str,
    ) -> list[dict[str, Any]]:
        after_payload = event_row.get("after_payload") or {}
        segment_ref = after_payload.get("segment") or {}
        segment_summary_id = event_row.get("segment_summary_id")
        start_index: int | None = None
        end_index: int | None = None
        try:
            start_index = int(segment_ref.get("start_utterance_index"))
            end_index = int(segment_ref.get("end_utterance_index"))
        except (TypeError, ValueError):
            pass

        for segment in segments:
            matches_range = (
                start_index is not None
                and end_index is not None
                and int(segment["start_utterance_index"]) == start_index
                and int(segment["end_utterance_index"]) == end_index
            )
            try:
                matches_segment_id = (
                    segment_summary_id is not None
                    and segment.get("segment_summary_id") is not None
                    and int(segment["segment_summary_id"]) == int(segment_summary_id)
                )
            except (TypeError, ValueError):
                matches_segment_id = False

            if matches_range or matches_segment_id:
                if field_name == "topic_label":
                    segment["topic_label"] = str(after_payload.get("topic_label") or "").strip()
                elif field_name == "summary_bullets":
                    segment["summary_bullets"] = normalize_summary_bullets(
                        after_payload.get("summary_bullets")
                    )
                segment["status"] = "complete"
                break
        return segments

    def materialize_user_summary(
        self,
        conn,
        *,
        meeting_id: str,
        edit_session_id: str,
    ) -> None:
        event_rows = self.fetch_edit_events(conn, meeting_id, edit_session_id)
        if not event_rows:
            return

        latest_feedback_event_id = max(int(row["feedback_event_id"]) for row in event_rows)

        first_after_payload = event_rows[0].get("after_payload") or {}
        base_summary_type = str(first_after_payload.get("base_summary_type") or "").strip()
        try:
            base_summary_id = int(first_after_payload.get("base_summary_id"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Feedback session {edit_session_id} is missing base summary metadata"
            ) from exc

        if base_summary_type not in {"llm_generated", "user_edited"}:
            raise RuntimeError(
                f"Unsupported base summary type for edit session {edit_session_id}: {base_summary_type}"
            )

        segments = self.fetch_summary_segments(conn, base_summary_id)
        if not segments:
            fallback_summary_id = self.fetch_latest_summary_id(conn, meeting_id, base_summary_type)
            if fallback_summary_id is None:
                raise RuntimeError(
                    f"Could not locate base summary for meeting={meeting_id} summary_type={base_summary_type}"
                )
            base_summary_id = fallback_summary_id
            segments = self.fetch_summary_segments(conn, base_summary_id)
        if not segments:
            raise RuntimeError(
                f"Base summary {base_summary_id} for meeting {meeting_id} does not contain any segments"
            )

        utterance_lookup = fetch_meeting_utterance_lookup(conn, meeting_id)
        if not utterance_lookup:
            raise RuntimeError(f"Could not load utterances for meeting {meeting_id}")

        working_segments = normalize_segments(segments, utterance_lookup)
        for event_row in event_rows:
            after_payload = event_row.get("after_payload") or {}
            event_type = str(event_row.get("event_type") or "")
            if event_type == "split_segment":
                try:
                    utterance_idx = int(after_payload.get("utterance_idx"))
                except (TypeError, ValueError):
                    continue
                working_segments = self.apply_split(working_segments, utterance_idx, utterance_lookup)
            elif event_type == "merge_segments":
                try:
                    utterance_idx = int(after_payload.get("utterance_idx"))
                except (TypeError, ValueError):
                    continue
                working_segments = self.apply_merge(working_segments, utterance_idx, utterance_lookup)
            elif event_type == "edit_topic_label":
                working_segments = self.apply_segment_text_edit(
                    working_segments,
                    event_row,
                    field_name="topic_label",
                )
            elif event_type == "edit_summary_bullets":
                working_segments = self.apply_segment_text_edit(
                    working_segments,
                    event_row,
                    field_name="summary_bullets",
                )

        final_segments: list[dict[str, Any]] = []
        for segment in normalize_segments(working_segments, utterance_lookup):
            complete_segment = {
                **segment,
                "topic_label": str(segment.get("topic_label") or "").strip(),
                "summary_bullets": normalize_summary_bullets(segment.get("summary_bullets")),
                "status": str(segment.get("status") or "draft"),
            }
            final_segments.append(complete_segment)

        next_version = self.fetch_next_user_summary_version(conn, meeting_id)
        summary_output_path = user_summary_output_path(self.config.output_root, meeting_id, next_version)
        summary_object_key = f"{self.config.object_prefix.strip('/')}/{meeting_id}/v{next_version}/summary.json"

        recap_payload = assemble_user_summary_payload(
            meeting_id,
            final_segments,
            based_on_summary_id=base_summary_id,
            based_on_summary_type=base_summary_type,
            edited_by_user_id=event_rows[-1].get("created_by_user_id"),
            edit_session_id=edit_session_id,
            version=next_version,
        )
        write_json(summary_output_path, recap_payload)

        summary_uri = f"local://{summary_output_path.resolve()}"
        if self.config.upload_artifacts:
            upload_file(summary_output_path, summary_object_key, self.logger)
            summary_uri = summary_object_key

        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM summaries
                WHERE meeting_id = %s
                  AND summary_type = 'user_edited'
                """,
                (meeting_id,),
            )
            cur.execute(
                """
                DELETE FROM topic_segments
                WHERE meeting_id = %s
                  AND segment_type = 'user_corrected'
                """,
                (meeting_id,),
            )

            inserted_topic_segment_ids: list[int] = []
            for segment in final_segments:
                start_row = utterance_lookup[int(segment["start_utterance_index"])]
                end_row = utterance_lookup[int(segment["end_utterance_index"])]
                cur.execute(
                    """
                    INSERT INTO topic_segments (
                        meeting_id,
                        segment_type,
                        segment_index,
                        start_utterance_id,
                        end_utterance_id,
                        start_time_sec,
                        end_time_sec,
                        topic_label
                    )
                    VALUES (%s, 'user_corrected', %s, %s, %s, %s, %s, %s)
                    RETURNING topic_segment_id
                    """,
                    (
                        meeting_id,
                        int(segment["segment_index"]),
                        start_row["utterance_id"],
                        end_row["utterance_id"],
                        float(segment["t_start"]),
                        float(segment["t_end"]),
                        segment["topic_label"],
                    ),
                )
                inserted_topic_segment_ids.append(int(cur.fetchone()["topic_segment_id"]))

            created_at = datetime.now(timezone.utc)

            cur.execute(
                """
                INSERT INTO summaries (
                    meeting_id,
                    summary_type,
                    summary_object_key,
                    created_by_user_id,
                    version,
                    created_at
                )
                VALUES (%s, 'user_edited', %s, %s, %s, %s)
                RETURNING summary_id
                """,
                (
                    meeting_id,
                    summary_uri,
                    event_rows[-1].get("created_by_user_id"),
                    next_version,
                    created_at,
                ),
            )
            summary_id = int(cur.fetchone()["summary_id"])

            for segment, topic_segment_id in zip(final_segments, inserted_topic_segment_ids, strict=True):
                cur.execute(
                    """
                    INSERT INTO segment_summaries (
                        meeting_id,
                        topic_segment_id,
                        summary_id,
                        segment_index,
                        topic_label,
                        summary_bullets,
                        status,
                        model_name,
                        model_version,
                        prompt_version,
                        created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        meeting_id,
                        topic_segment_id,
                        summary_id,
                        int(segment["segment_index"]),
                        segment["topic_label"],
                        Json(segment["summary_bullets"]),
                        segment["status"],
                        "user_editor",
                        f"session:{edit_session_id[:8]}",
                        edit_session_id,
                        created_at,
                    ),
                )

            cur.execute(
                """
                UPDATE feedback_events
                SET after_payload = after_payload || jsonb_build_object(
                    'materialized_summary_id', %s,
                    'materialized_user_summary_version', %s,
                    'materialized_at', %s,
                    'materialized_edit_session_id', %s
                )
                WHERE meeting_id = %s
                  AND event_type = ANY(%s)
                  AND after_payload ? 'edit_session_id'
                  AND feedback_event_id <= %s
                """,
                (
                    summary_id,
                    next_version,
                    created_at.isoformat(),
                    edit_session_id,
                    meeting_id,
                    list(MATERIALIZABLE_EVENT_TYPES),
                    latest_feedback_event_id,
                ),
            )
        conn.commit()


def validate_config(config: UserSummaryMaterializeConfig) -> None:
    if config.poll_interval_seconds <= 0:
        raise ValueError("USER_SUMMARY_WORKER_POLL_INTERVAL_SECONDS must be > 0")
    if config.heartbeat_interval_seconds <= 0:
        raise ValueError("WORKFLOW_TASK_HEARTBEAT_INTERVAL_SECONDS must be > 0")
    if config.backoff_base_seconds <= 0:
        raise ValueError("WORKFLOW_TASK_BACKOFF_BASE_SECONDS must be > 0")
    if config.backoff_max_seconds < config.backoff_base_seconds:
        raise ValueError(
            "WORKFLOW_TASK_BACKOFF_MAX_SECONDS must be >= WORKFLOW_TASK_BACKOFF_BASE_SECONDS"
        )


def main() -> None:
    config = UserSummaryMaterializeConfig()
    validate_config(config)
    logger = build_logger(APP_NAME, config.log_dir)
    schema_conn = get_conn()
    try:
        ensure_workflow_schema(schema_conn)
    finally:
        schema_conn.close()

    service = UserSummaryMaterializeService(config=config, logger=logger)

    logger.info("Starting %s", APP_NAME)
    logger.info(
        "User summary materialize config | poll_interval=%ss output_root=%s upload=%s worker_id=%s",
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
            logger.exception("User summary materialization loop failed; retrying after sleep")
            time.sleep(config.poll_interval_seconds)


if __name__ == "__main__":
    main()
