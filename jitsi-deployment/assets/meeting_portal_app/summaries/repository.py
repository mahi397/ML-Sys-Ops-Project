from __future__ import annotations

import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.config import build_rclone_object_uri, get_rclone_timeout_seconds, stage1_fallback_enabled
from core.db import get_conn


def participant_view_condition(alias: str = "mp") -> str:
    return ""


def participant_edit_expression(alias: str = "mp") -> str:
    return "TRUE"


def participant_edit_group_by(alias: str = "mp") -> str:
    return ""


def env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def user_can_edit_summary(user_id: str, meeting_id: str) -> bool:
    view_condition = participant_view_condition("mp")
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT TRUE AS can_edit_summary
            FROM meeting_participants mp
            WHERE user_id = %s
              AND meeting_id = %s
              {view_condition}
            LIMIT 1
            """,
            (user_id, meeting_id),
        )
        row = cur.fetchone()
    return bool(row and row.get("can_edit_summary"))


def fetch_recap_rows_for_user(user_id: str) -> list[dict[str, Any]]:
    view_condition = participant_view_condition("mp")
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            WITH latest_summaries AS (
                SELECT DISTINCT ON (s.meeting_id)
                    s.meeting_id,
                    s.summary_id,
                    s.version,
                    s.created_at
                FROM summaries s
                JOIN segment_summaries ss
                    ON ss.summary_id = s.summary_id
                WHERE s.summary_type = 'llm_generated'
                ORDER BY s.meeting_id, s.version DESC, s.created_at DESC, s.summary_id DESC
            ),
            summary_rollups AS (
                SELECT
                    ss.summary_id,
                    COUNT(*) AS segment_count,
                    BOOL_AND(ss.status = 'complete') AS is_complete,
                    MAX(ss.created_at) AS summary_updated_at,
                    COALESCE(
                        MAX(NULLIF(ss.model_version, '')),
                        MAX(NULLIF(ss.model_name, '')),
                        ''
                    ) AS model_version
                FROM segment_summaries ss
                GROUP BY ss.summary_id
            )
            SELECT
                mp.meeting_id,
                COALESCE(m.source_name, mp.meeting_id) AS meeting_title,
                mp.role,
                m.started_at,
                m.ended_at,
                mp.joined_at,
                latest.summary_id,
                latest.version AS summary_version,
                latest.created_at AS summary_created_at,
                rollup.summary_updated_at,
                rollup.segment_count,
                rollup.is_complete,
                rollup.model_version
            FROM meeting_participants mp
            JOIN latest_summaries latest
                ON latest.meeting_id = mp.meeting_id
            JOIN summary_rollups rollup
                ON rollup.summary_id = latest.summary_id
            LEFT JOIN meetings m
                ON m.meeting_id = mp.meeting_id
            WHERE mp.user_id = %s
              {view_condition}
              AND rollup.segment_count > 0
            ORDER BY COALESCE(m.started_at, m.ended_at, mp.joined_at, latest.created_at) DESC, mp.meeting_id
            """,
            (user_id,),
        )
        return cur.fetchall() or []


def fetch_summary_variants_for_user(user_id: str, meeting_id: str) -> list[dict[str, Any]]:
    view_condition = participant_view_condition("mp")
    can_edit_expr = participant_edit_expression("mp")
    can_edit_group_by = participant_edit_group_by("mp")
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            WITH summary_rollups AS (
                SELECT
                    s.summary_id,
                    s.meeting_id,
                    s.summary_type,
                    s.version AS summary_version,
                    s.summary_object_key,
                    s.created_by_user_id,
                    s.created_at AS summary_created_at,
                    COUNT(ss.segment_summary_id) AS segment_count,
                    BOOL_AND(ss.status = 'complete') AS is_complete,
                    MAX(ss.created_at) AS summary_updated_at,
                    COALESCE(
                        MAX(NULLIF(ss.model_version, '')),
                        MAX(NULLIF(ss.model_name, '')),
                        ''
                    ) AS model_version
                FROM summaries s
                JOIN segment_summaries ss
                    ON ss.summary_id = s.summary_id
                WHERE s.summary_type IN ('llm_generated', 'user_edited')
                GROUP BY
                    s.summary_id,
                    s.meeting_id,
                    s.summary_type,
                    s.version,
                    s.summary_object_key,
                    s.created_by_user_id,
                    s.created_at
            ),
            latest_by_type AS (
                SELECT DISTINCT ON (meeting_id, summary_type)
                    summary_id,
                    meeting_id,
                    summary_type,
                    summary_version,
                    summary_object_key,
                    created_by_user_id,
                    summary_created_at,
                    segment_count,
                    is_complete,
                    summary_updated_at,
                    model_version
                FROM summary_rollups
                ORDER BY
                    meeting_id,
                    summary_type,
                    summary_version DESC,
                    summary_created_at DESC,
                    summary_id DESC
            )
            SELECT
                mp.meeting_id,
                COALESCE(m.source_name, mp.meeting_id) AS meeting_title,
                mp.role,
                {can_edit_expr} AS can_edit_summary,
                latest.summary_id,
                latest.summary_type,
                latest.summary_version,
                latest.summary_object_key,
                latest.created_by_user_id,
                latest.summary_created_at,
                latest.summary_updated_at,
                latest.segment_count,
                latest.is_complete,
                latest.model_version,
                m.started_at,
                m.ended_at,
                COUNT(DISTINCT all_mp.user_id) AS participant_count
            FROM meeting_participants mp
            JOIN meetings m
                ON m.meeting_id = mp.meeting_id
            JOIN latest_by_type latest
                ON latest.meeting_id = mp.meeting_id
            LEFT JOIN meeting_participants all_mp
                ON all_mp.meeting_id = mp.meeting_id
            WHERE mp.user_id = %s
              AND mp.meeting_id = %s
              {view_condition}
            GROUP BY
                mp.meeting_id,
                m.source_name,
                mp.role{can_edit_group_by},
                latest.summary_id,
                latest.summary_type,
                latest.summary_version,
                latest.summary_object_key,
                latest.created_by_user_id,
                latest.summary_created_at,
                latest.summary_updated_at,
                latest.segment_count,
                latest.is_complete,
                latest.model_version,
                m.started_at,
                m.ended_at
            ORDER BY
                CASE latest.summary_type
                    WHEN 'user_edited' THEN 0
                    ELSE 1
                END,
                latest.summary_created_at DESC,
                latest.summary_id DESC
            """,
            (user_id, meeting_id),
        )
        return cur.fetchall() or []


def fetch_summary_segments(summary_id: int) -> list[dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                ss.segment_summary_id,
                ss.segment_index,
                ss.topic_label,
                ss.summary_bullets,
                ss.status,
                ss.model_name,
                ss.model_version,
                ts.start_time_sec AS t_start,
                ts.end_time_sec AS t_end,
                u_start.utterance_index AS start_utterance_index,
                u_end.utterance_index AS end_utterance_index
            FROM segment_summaries ss
            JOIN topic_segments ts
                ON ts.topic_segment_id = ss.topic_segment_id
            LEFT JOIN utterances u_start
                ON u_start.utterance_id = ts.start_utterance_id
            LEFT JOIN utterances u_end
                ON u_end.utterance_id = ts.end_utterance_id
            WHERE ss.summary_id = %s
            ORDER BY ss.segment_index
            """,
            (summary_id,),
        )
        return cur.fetchall() or []


def fetch_stage1_artifact_keys(meeting_id: str) -> dict[str, str]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (artifact_type)
                artifact_type,
                object_key
            FROM meeting_artifacts
            WHERE meeting_id = %s
              AND artifact_type IN (
                  'stage1_requests_json',
                  'stage1_requests_jsonl',
                  'stage1_responses_json',
                  'stage1_responses_jsonl'
              )
            ORDER BY artifact_type, artifact_version DESC, created_at DESC
            """,
            (meeting_id,),
        )
        rows = cur.fetchall() or []

    return {
        str(row["artifact_type"]): str(row["object_key"])
        for row in rows
        if row.get("artifact_type") and row.get("object_key")
    }


@lru_cache(maxsize=256)
def read_artifact_text(object_key: str) -> str | None:
    if not object_key:
        return None

    if object_key.startswith("local://"):
        local_path = Path(object_key.removeprefix("local://"))
        if not local_path.exists():
            return None
        try:
            return local_path.read_text(encoding="utf-8")
        except OSError:
            return None

    try:
        completed = subprocess.run(
            ["rclone", "cat", build_rclone_object_uri(object_key)],
            check=True,
            capture_output=True,
            text=True,
            timeout=get_rclone_timeout_seconds(),
        )
    except (OSError, subprocess.SubprocessError):
        return None

    return completed.stdout


def extract_artifact_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]

    if not isinstance(payload, dict):
        return []

    for key in ("responses", "predictions", "results", "items", "requests"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]

    for key in ("response", "payload", "data"):
        nested_payload = payload.get(key)
        rows = extract_artifact_rows(nested_payload)
        if rows:
            return rows

    return []


def parse_artifact_rows(object_key: str) -> list[dict[str, Any]]:
    raw_text = read_artifact_text(object_key)
    if not raw_text:
        return []

    try:
        if object_key.endswith(".jsonl"):
            rows = [
                json.loads(line)
                for line in raw_text.splitlines()
                if line.strip()
            ]
        else:
            payload = json.loads(raw_text)
            rows = extract_artifact_rows(payload)
    except json.JSONDecodeError:
        return []

    return [row for row in rows if isinstance(row, dict)]


@lru_cache(maxsize=128)
def fetch_stage1_confidence_by_left_utterance_id(meeting_id: str) -> dict[int, float]:
    if not stage1_fallback_enabled():
        return {}

    artifact_keys = fetch_stage1_artifact_keys(meeting_id)
    request_key = artifact_keys.get("stage1_requests_jsonl") or artifact_keys.get("stage1_requests_json")
    response_key = artifact_keys.get("stage1_responses_jsonl") or artifact_keys.get("stage1_responses_json")
    if not request_key or not response_key:
        return {}

    request_rows = parse_artifact_rows(request_key)
    response_rows = parse_artifact_rows(response_key)
    if not request_rows or not response_rows:
        return {}

    request_by_id: dict[str, dict[str, Any]] = {}
    request_by_left_model_index: dict[int, dict[str, Any]] = {}
    for request_row in request_rows:
        request_id = str(request_row.get("request_id") or "").strip()
        if request_id:
            request_by_id[request_id] = request_row

        metadata = request_row.get("metadata")
        if not isinstance(metadata, dict):
            continue

        left_model_index = metadata.get("left_model_index")
        if left_model_index is None:
            continue

        try:
            request_by_left_model_index[int(left_model_index)] = request_row
        except (TypeError, ValueError):
            continue

    confidence_by_left_utterance_id: dict[int, float] = {}
    for response_row in response_rows:
        request_row: dict[str, Any] | None = None

        request_id = str(response_row.get("request_id") or "").strip()
        if request_id:
            request_row = request_by_id.get(request_id)

        if request_row is None and response_row.get("left_model_index") is not None:
            try:
                request_row = request_by_left_model_index.get(int(response_row["left_model_index"]))
            except (TypeError, ValueError):
                request_row = None

        if request_row is None:
            continue

        metadata = request_row.get("metadata")
        if not isinstance(metadata, dict):
            continue

        left_source_utterance_id = metadata.get("left_source_utterance_id")
        boundary_probability = response_row.get(
            "boundary_probability",
            response_row.get("pred_boundary_prob", response_row.get("score")),
        )
        if left_source_utterance_id is None or boundary_probability is None:
            continue

        try:
            confidence_by_left_utterance_id[int(left_source_utterance_id)] = float(boundary_probability)
        except (TypeError, ValueError):
            continue

    return confidence_by_left_utterance_id


def fetch_meeting_utterances(meeting_id: str) -> list[dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                u.utterance_id,
                u.utterance_index,
                COALESCE(NULLIF(ms.display_name, ''), NULLIF(ms.speaker_label, ''), 'Speaker') AS speaker,
                COALESCE(NULLIF(u.clean_text, ''), u.raw_text) AS text,
                u.start_time_sec,
                u.end_time_sec,
                ut.pred_boundary_prob AS boundary_confidence,
                ut.pred_boundary_label AS is_boundary
            FROM utterances u
            JOIN meeting_speakers ms
                ON ms.meeting_speaker_id = u.meeting_speaker_id
            LEFT JOIN utterance_transitions ut
                ON ut.left_utterance_id = u.utterance_id
            WHERE u.meeting_id = %s
            ORDER BY u.utterance_index
            """,
            (meeting_id,),
        )
        rows = cur.fetchall() or []

    if not rows or all(row.get("boundary_confidence") is not None for row in rows):
        return rows

    fallback_confidence_by_utterance_id = fetch_stage1_confidence_by_left_utterance_id(meeting_id)
    if not fallback_confidence_by_utterance_id:
        return rows

    for row in rows:
        if row.get("boundary_confidence") is not None:
            continue

        fallback_confidence = fallback_confidence_by_utterance_id.get(int(row["utterance_id"]))
        if fallback_confidence is not None:
            row["boundary_confidence"] = fallback_confidence

    return rows


def fetch_latest_summary_id_for_meeting(
    meeting_id: str,
    summary_type: str = "llm_generated",
) -> int | None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.summary_id
            FROM summaries s
            WHERE s.meeting_id = %s
              AND s.summary_type = %s
              AND EXISTS (
                  SELECT 1
                  FROM segment_summaries ss
                  WHERE ss.summary_id = s.summary_id
              )
            ORDER BY s.version DESC, s.created_at DESC, s.summary_id DESC
            LIMIT 1
            """,
            (meeting_id, summary_type),
        )
        row = cur.fetchone()

    return int(row["summary_id"]) if row else None


def fetch_segment_feedback_context(meeting_id: str, segment_summary_id: int) -> dict[str, Any] | None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                ss.segment_summary_id,
                ss.summary_id,
                ss.segment_index,
                ss.topic_label,
                ss.summary_bullets,
                u_start.utterance_index AS start_utterance_index,
                u_end.utterance_index AS end_utterance_index
            FROM segment_summaries ss
            JOIN topic_segments ts
                ON ts.topic_segment_id = ss.topic_segment_id
            LEFT JOIN utterances u_start
                ON u_start.utterance_id = ts.start_utterance_id
            LEFT JOIN utterances u_end
                ON u_end.utterance_id = ts.end_utterance_id
            WHERE ss.segment_summary_id = %s
              AND ss.meeting_id = %s
            LIMIT 1
            """,
            (segment_summary_id, meeting_id),
        )
        return cur.fetchone()


def fetch_summary_variant_for_user_by_id(
    user_id: str,
    meeting_id: str,
    summary_id: int,
) -> dict[str, Any] | None:
    variants = fetch_summary_variants_for_user(user_id, meeting_id)
    for variant in variants:
        if int(variant["summary_id"]) == summary_id:
            return variant
    return None


def user_can_access_recap(user_id: str, meeting_id: str) -> bool:
    view_condition = participant_view_condition("mp")
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT 1
            FROM meeting_participants mp
            WHERE mp.user_id = %s
              AND mp.meeting_id = %s
              {view_condition}
              AND EXISTS (
                  SELECT 1
                  FROM summaries s
                  JOIN segment_summaries ss
                      ON ss.summary_id = s.summary_id
                  WHERE s.meeting_id = mp.meeting_id
                    AND s.summary_type IN ('llm_generated', 'user_edited')
              )
            LIMIT 1
            """,
            (user_id, meeting_id),
        )
        return cur.fetchone() is not None


def insert_feedback_event(
    meeting_id: str,
    summary_id: int | None,
    segment_summary_id: int | None,
    event_type: str,
    before_payload: dict[str, Any],
    after_payload: dict[str, Any],
    user_id: str,
) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO feedback_events (
                meeting_id,
                summary_id,
                segment_summary_id,
                event_type,
                event_source,
                before_payload,
                after_payload,
                created_by_user_id
            )
            VALUES (%s, %s, %s, %s, 'user', %s::jsonb, %s::jsonb, %s)
            """,
            (
                meeting_id,
                summary_id,
                segment_summary_id,
                event_type,
                json.dumps(before_payload),
                json.dumps(after_payload),
                user_id,
            ),
        )
        conn.commit()


def update_user_summary_text_edits(
    meeting_id: str,
    summary_id: int,
    operations: list[tuple[int | None, str, dict[str, Any], dict[str, Any]]],
) -> int:
    updated_count = 0
    with get_conn() as conn, conn.cursor() as cur:
        for segment_summary_id, event_type, _before_payload, after_payload in operations:
            if event_type not in {"edit_topic_label", "edit_summary_bullets"}:
                continue

            set_clauses = ["status = 'complete'", "created_at = NOW()"]
            params: list[Any] = []
            if event_type == "edit_topic_label":
                set_clauses.append("topic_label = %s")
                params.append(str(after_payload.get("topic_label") or "").strip())
            elif event_type == "edit_summary_bullets":
                set_clauses.append("summary_bullets = %s::jsonb")
                params.append(json.dumps(after_payload.get("summary_bullets") or []))

            if segment_summary_id is not None:
                cur.execute(
                    f"""
                    UPDATE segment_summaries
                    SET {", ".join(set_clauses)}
                    WHERE meeting_id = %s
                      AND summary_id = %s
                      AND segment_summary_id = %s
                    """,
                    (
                        *params,
                        meeting_id,
                        summary_id,
                        segment_summary_id,
                    ),
                )
                updated_count += cur.rowcount
                if event_type == "edit_topic_label":
                    cur.execute(
                        """
                        UPDATE topic_segments ts
                        SET topic_label = %s
                        FROM segment_summaries ss
                        WHERE ss.topic_segment_id = ts.topic_segment_id
                          AND ss.meeting_id = %s
                          AND ss.summary_id = %s
                          AND ss.segment_summary_id = %s
                        """,
                        (
                            str(after_payload.get("topic_label") or "").strip(),
                            meeting_id,
                            summary_id,
                            segment_summary_id,
                        ),
                    )
                continue

            segment_ref = after_payload.get("segment") or {}
            try:
                start_index = int(segment_ref.get("start_utterance_index"))
                end_index = int(segment_ref.get("end_utterance_index"))
            except (TypeError, ValueError):
                continue

            cur.execute(
                f"""
                UPDATE segment_summaries ss
                SET {", ".join(set_clauses)}
                FROM topic_segments ts
                JOIN utterances u_start
                  ON u_start.utterance_id = ts.start_utterance_id
                JOIN utterances u_end
                  ON u_end.utterance_id = ts.end_utterance_id
                WHERE ss.topic_segment_id = ts.topic_segment_id
                  AND ss.meeting_id = %s
                  AND ss.summary_id = %s
                  AND u_start.utterance_index = %s
                  AND u_end.utterance_index = %s
                """,
                (
                    *params,
                    meeting_id,
                    summary_id,
                    start_index,
                    end_index,
                ),
            )
            updated_count += cur.rowcount
            if event_type == "edit_topic_label":
                cur.execute(
                    """
                    UPDATE topic_segments ts
                    SET topic_label = %s
                    FROM segment_summaries ss, utterances u_start, utterances u_end
                    WHERE ss.topic_segment_id = ts.topic_segment_id
                      AND u_start.utterance_id = ts.start_utterance_id
                      AND u_end.utterance_id = ts.end_utterance_id
                      AND ss.meeting_id = %s
                      AND ss.summary_id = %s
                      AND u_start.utterance_index = %s
                      AND u_end.utterance_index = %s
                    """,
                    (
                        str(after_payload.get("topic_label") or "").strip(),
                        meeting_id,
                        summary_id,
                        start_index,
                        end_index,
                    ),
                )

        conn.commit()
    return updated_count


def enqueue_user_summary_materialize_task(meeting_id: str, *, edit_session_id: str | None = None) -> None:
    artifact_version = env_int("STAGE1_ARTIFACT_VERSION", 1)
    max_attempts = env_int("USER_SUMMARY_MATERIALIZE_MAX_ATTEMPTS", 8)
    payload_json = {
        "task_name": "user_summary_materialize",
        "meeting_id": meeting_id,
        "artifact_version": artifact_version,
        "phase": "user_summary_pending",
    }
    if edit_session_id:
        payload_json["edit_session_id"] = edit_session_id

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO workflow_tasks (
                task_type,
                meeting_id,
                artifact_version,
                status,
                payload_json,
                max_attempts,
                next_attempt_at
            )
            VALUES (
                'user_summary_materialize',
                %s,
                %s,
                'pending',
                %s::jsonb,
                %s,
                NOW()
            )
            ON CONFLICT (task_type, meeting_id, artifact_version)
            DO UPDATE
            SET payload_json = EXCLUDED.payload_json,
                max_attempts = EXCLUDED.max_attempts,
                status = CASE
                    WHEN workflow_tasks.status = 'running' THEN workflow_tasks.status
                    ELSE 'pending'
                END,
                next_attempt_at = CASE
                    WHEN workflow_tasks.status = 'running' THEN workflow_tasks.next_attempt_at
                    ELSE NOW()
                END,
                locked_by = CASE
                    WHEN workflow_tasks.status = 'running' THEN workflow_tasks.locked_by
                    ELSE NULL
                END,
                locked_at = CASE
                    WHEN workflow_tasks.status = 'running' THEN workflow_tasks.locked_at
                    ELSE NULL
                END,
                heartbeat_at = CASE
                    WHEN workflow_tasks.status = 'running' THEN workflow_tasks.heartbeat_at
                    ELSE NULL
                END,
                last_error = CASE
                    WHEN workflow_tasks.status = 'running' THEN workflow_tasks.last_error
                    ELSE NULL
                END,
                updated_at = NOW()
            """,
            (
                meeting_id,
                artifact_version,
                json.dumps(payload_json),
                max_attempts,
            ),
        )
        conn.commit()
