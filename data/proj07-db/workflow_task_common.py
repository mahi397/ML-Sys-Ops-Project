from __future__ import annotations

import random
import socket
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

from psycopg.types.json import Json

from feedback_common import get_conn


TASK_STATUS_PENDING = "pending"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_RETRY_SCHEDULED = "retry_scheduled"
TASK_STATUS_SUCCEEDED = "succeeded"
TASK_STATUS_FAILED_PERMANENT = "failed_permanent"
TASK_STATUS_CANCELLED = "cancelled"

TASK_TERMINAL_STATUSES = (
    TASK_STATUS_SUCCEEDED,
    TASK_STATUS_FAILED_PERMANENT,
    TASK_STATUS_CANCELLED,
)

TASK_RETRYABLE_STATUSES = (
    TASK_STATUS_PENDING,
    TASK_STATUS_RETRY_SCHEDULED,
)

WORKFLOW_SCHEMA_LOCK_ID = 4_187_224_031_091_337_221


WORKFLOW_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS workflow_tasks (
    task_id BIGSERIAL PRIMARY KEY,
    task_type TEXT NOT NULL,
    meeting_id TEXT NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    artifact_version INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN (
            'pending',
            'running',
            'retry_scheduled',
            'succeeded',
            'failed_permanent',
            'cancelled'
        )
    ),
    payload_json JSONB,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    max_attempts INTEGER NOT NULL DEFAULT 8 CHECK (max_attempts > 0),
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    locked_by TEXT,
    locked_at TIMESTAMPTZ,
    heartbeat_at TIMESTAMPTZ,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (task_type, meeting_id, artifact_version)
);

CREATE TABLE IF NOT EXISTS workflow_task_attempts (
    attempt_id BIGSERIAL PRIMARY KEY,
    task_id BIGINT NOT NULL REFERENCES workflow_tasks(task_id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
    worker_id TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    outcome TEXT,
    error_summary TEXT,
    stderr_tail TEXT,
    duration_ms BIGINT,
    UNIQUE (task_id, attempt_number)
);

CREATE INDEX IF NOT EXISTS idx_workflow_tasks_status_next_attempt_at
    ON workflow_tasks (status, next_attempt_at);

CREATE INDEX IF NOT EXISTS idx_workflow_tasks_task_type_status_next_attempt_at
    ON workflow_tasks (task_type, status, next_attempt_at);

CREATE INDEX IF NOT EXISTS idx_workflow_tasks_meeting_id
    ON workflow_tasks (meeting_id);

CREATE INDEX IF NOT EXISTS idx_workflow_task_attempts_task_id
    ON workflow_task_attempts (task_id);
"""


@dataclass(frozen=True)
class WorkflowTaskLease:
    task_id: int
    task_type: str
    meeting_id: str
    artifact_version: int
    status: str
    payload_json: dict[str, Any] | None
    attempt_count: int
    max_attempts: int
    next_attempt_at: Any
    locked_by: str | None
    locked_at: Any
    heartbeat_at: Any
    last_error: str | None
    created_at: Any
    updated_at: Any
    attempt_id: int
    attempt_number: int


def ensure_workflow_schema(conn) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s)", (WORKFLOW_SCHEMA_LOCK_ID,))
            cur.execute(WORKFLOW_SCHEMA_SQL)
            cur.execute("SELECT pg_advisory_unlock(%s)", (WORKFLOW_SCHEMA_LOCK_ID,))
        conn.commit()
    except Exception:
        conn.rollback()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(%s)", (WORKFLOW_SCHEMA_LOCK_ID,))
            conn.commit()
        except Exception:
            conn.rollback()
        raise


def make_worker_id(app_name: str) -> str:
    return f"{app_name}:{socket.gethostname()}:{threading.get_ident()}"


def _json_payload(payload_json: dict[str, Any] | None) -> Json | None:
    if payload_json is None:
        return None
    return Json(payload_json)


def upsert_workflow_task(
    conn,
    *,
    task_type: str,
    meeting_id: str,
    artifact_version: int,
    payload_json: dict[str, Any] | None = None,
    max_attempts: int = 8,
    revive_succeeded: bool = True,
    commit: bool = True,
) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO workflow_tasks (
                task_type,
                meeting_id,
                artifact_version,
                status,
                payload_json,
                max_attempts
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (task_type, meeting_id, artifact_version)
            DO UPDATE
            SET payload_json = COALESCE(EXCLUDED.payload_json, workflow_tasks.payload_json),
                max_attempts = EXCLUDED.max_attempts,
                status = CASE
                    WHEN workflow_tasks.status = 'succeeded' AND %s THEN 'pending'
                    ELSE workflow_tasks.status
                END,
                next_attempt_at = CASE
                    WHEN workflow_tasks.status = 'succeeded' AND %s THEN NOW()
                    ELSE workflow_tasks.next_attempt_at
                END,
                locked_by = CASE
                    WHEN workflow_tasks.status = 'succeeded' AND %s THEN NULL
                    ELSE workflow_tasks.locked_by
                END,
                locked_at = CASE
                    WHEN workflow_tasks.status = 'succeeded' AND %s THEN NULL
                    ELSE workflow_tasks.locked_at
                END,
                heartbeat_at = CASE
                    WHEN workflow_tasks.status = 'succeeded' AND %s THEN NULL
                    ELSE workflow_tasks.heartbeat_at
                END,
                last_error = CASE
                    WHEN workflow_tasks.status = 'succeeded' AND %s THEN NULL
                    ELSE workflow_tasks.last_error
                END,
                updated_at = NOW()
            RETURNING task_id, task_type, meeting_id, artifact_version, status, attempt_count, max_attempts
            """,
            (
                task_type,
                meeting_id,
                artifact_version,
                TASK_STATUS_PENDING,
                _json_payload(payload_json),
                max_attempts,
                revive_succeeded,
                revive_succeeded,
                revive_succeeded,
                revive_succeeded,
                revive_succeeded,
                revive_succeeded,
            ),
        )
        row = cur.fetchone()
    if commit:
        conn.commit()
    return row


def claim_next_workflow_task(
    conn,
    *,
    task_type: str,
    worker_id: str,
) -> WorkflowTaskLease | None:
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT task_id
                FROM workflow_tasks
                WHERE task_type = %s
                  AND status IN ('pending', 'retry_scheduled')
                  AND next_attempt_at <= NOW()
                ORDER BY next_attempt_at ASC, created_at ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
                """,
                (task_type,),
            )
            task_row = cur.fetchone()
            if task_row is None:
                conn.rollback()
                return None

            cur.execute(
                """
                UPDATE workflow_tasks
                SET status = 'running',
                    attempt_count = attempt_count + 1,
                    locked_by = %s,
                    locked_at = NOW(),
                    heartbeat_at = NOW(),
                    updated_at = NOW()
                WHERE task_id = %s
                RETURNING *
                """,
                (worker_id, task_row["task_id"]),
            )
            claimed = cur.fetchone()

            attempt_number = int(claimed["attempt_count"])
            cur.execute(
                """
                INSERT INTO workflow_task_attempts (
                    task_id,
                    attempt_number,
                    worker_id
                )
                VALUES (%s, %s, %s)
                RETURNING attempt_id
                """,
                (claimed["task_id"], attempt_number, worker_id),
            )
            attempt_row = cur.fetchone()
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return WorkflowTaskLease(
        task_id=int(claimed["task_id"]),
        task_type=str(claimed["task_type"]),
        meeting_id=str(claimed["meeting_id"]),
        artifact_version=int(claimed["artifact_version"]),
        status=str(claimed["status"]),
        payload_json=claimed.get("payload_json"),
        attempt_count=int(claimed["attempt_count"]),
        max_attempts=int(claimed["max_attempts"]),
        next_attempt_at=claimed["next_attempt_at"],
        locked_by=claimed.get("locked_by"),
        locked_at=claimed.get("locked_at"),
        heartbeat_at=claimed.get("heartbeat_at"),
        last_error=claimed.get("last_error"),
        created_at=claimed["created_at"],
        updated_at=claimed["updated_at"],
        attempt_id=int(attempt_row["attempt_id"]),
        attempt_number=attempt_number,
    )


def mark_task_heartbeat(conn, *, task_id: int, worker_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE workflow_tasks
            SET heartbeat_at = NOW(),
                updated_at = NOW()
            WHERE task_id = %s
              AND status = 'running'
              AND locked_by = %s
            """,
            (task_id, worker_id),
        )
    conn.commit()


def _finish_attempt(
    conn,
    *,
    attempt_id: int,
    outcome: str,
    error_summary: str | None,
    stderr_tail: str | None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE workflow_task_attempts
            SET finished_at = NOW(),
                outcome = %s,
                error_summary = %s,
                stderr_tail = %s,
                duration_ms = GREATEST(
                    0,
                    FLOOR(EXTRACT(EPOCH FROM (NOW() - started_at)) * 1000)
                )::BIGINT
            WHERE attempt_id = %s
            """,
            (outcome, error_summary, stderr_tail, attempt_id),
        )


def mark_task_succeeded(
    conn,
    *,
    lease: WorkflowTaskLease,
    worker_id: str,
    downstream_tasks: list[dict[str, Any]] | None = None,
) -> None:
    try:
        _finish_attempt(
            conn,
            attempt_id=lease.attempt_id,
            outcome=TASK_STATUS_SUCCEEDED,
            error_summary=None,
            stderr_tail=None,
        )
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE workflow_tasks
                SET status = 'succeeded',
                    locked_by = NULL,
                    locked_at = NULL,
                    heartbeat_at = NULL,
                    last_error = NULL,
                    updated_at = NOW()
                WHERE task_id = %s
                  AND locked_by = %s
                """,
                (lease.task_id, worker_id),
            )
        for downstream_task in downstream_tasks or []:
            upsert_workflow_task(conn, commit=False, **downstream_task)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def compute_retry_delay_seconds(
    *,
    attempt_count: int,
    base_seconds: float,
    max_seconds: float,
) -> int:
    attempt_index = max(attempt_count - 1, 0)
    raw_delay = base_seconds * (2 ** attempt_index)
    bounded_delay = min(raw_delay, max_seconds)
    jitter_factor = random.uniform(0.85, 1.15)
    return max(1, int(round(bounded_delay * jitter_factor)))


def mark_task_retry(
    conn,
    *,
    lease: WorkflowTaskLease,
    worker_id: str,
    error_summary: str,
    stderr_tail: str | None,
    backoff_base_seconds: float,
    backoff_max_seconds: float,
) -> str:
    should_fail_permanent = lease.attempt_count >= lease.max_attempts
    next_status = (
        TASK_STATUS_FAILED_PERMANENT
        if should_fail_permanent
        else TASK_STATUS_RETRY_SCHEDULED
    )
    try:
        _finish_attempt(
            conn,
            attempt_id=lease.attempt_id,
            outcome=next_status,
            error_summary=error_summary,
            stderr_tail=stderr_tail,
        )
        with conn.cursor() as cur:
            if should_fail_permanent:
                cur.execute(
                    """
                    UPDATE workflow_tasks
                    SET status = 'failed_permanent',
                        locked_by = NULL,
                        locked_at = NULL,
                        heartbeat_at = NULL,
                        last_error = %s,
                        updated_at = NOW()
                    WHERE task_id = %s
                      AND locked_by = %s
                    """,
                    (error_summary, lease.task_id, worker_id),
                )
            else:
                delay_seconds = compute_retry_delay_seconds(
                    attempt_count=lease.attempt_count,
                    base_seconds=backoff_base_seconds,
                    max_seconds=backoff_max_seconds,
                )
                cur.execute(
                    """
                    UPDATE workflow_tasks
                    SET status = 'retry_scheduled',
                        next_attempt_at = NOW() + (%s * INTERVAL '1 second'),
                        locked_by = NULL,
                        locked_at = NULL,
                        heartbeat_at = NULL,
                        last_error = %s,
                        updated_at = NOW()
                    WHERE task_id = %s
                      AND locked_by = %s
                    """,
                    (delay_seconds, error_summary, lease.task_id, worker_id),
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return next_status


def sweep_stale_running_tasks(
    conn,
    *,
    stale_after_seconds: int,
) -> list[dict]:
    stale_error = (
        f"task heartbeat expired after {stale_after_seconds} seconds while task was running"
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT task_id, attempt_count, max_attempts
                FROM workflow_tasks
                WHERE status = 'running'
                  AND heartbeat_at IS NOT NULL
                  AND heartbeat_at < NOW() - (%s * INTERVAL '1 second')
                FOR UPDATE SKIP LOCKED
                """,
                (stale_after_seconds,),
            )
            stale_rows = cur.fetchall()
            results: list[dict] = []
            for row in stale_rows:
                task_id = int(row["task_id"])
                attempt_count = int(row["attempt_count"])
                max_attempts = int(row["max_attempts"])
                next_status = (
                    TASK_STATUS_FAILED_PERMANENT
                    if attempt_count >= max_attempts
                    else TASK_STATUS_RETRY_SCHEDULED
                )
                if next_status == TASK_STATUS_FAILED_PERMANENT:
                    cur.execute(
                        """
                        UPDATE workflow_tasks
                        SET status = 'failed_permanent',
                            locked_by = NULL,
                            locked_at = NULL,
                            heartbeat_at = NULL,
                            last_error = %s,
                            updated_at = NOW()
                        WHERE task_id = %s
                        RETURNING task_id, task_type, meeting_id, status, attempt_count
                        """,
                        (stale_error, task_id),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE workflow_tasks
                        SET status = 'retry_scheduled',
                            next_attempt_at = NOW(),
                            locked_by = NULL,
                            locked_at = NULL,
                            heartbeat_at = NULL,
                            last_error = %s,
                            updated_at = NOW()
                        WHERE task_id = %s
                        RETURNING task_id, task_type, meeting_id, status, attempt_count
                        """,
                        (stale_error, task_id),
                    )
                updated = cur.fetchone()
                cur.execute(
                    """
                    UPDATE workflow_task_attempts
                    SET finished_at = NOW(),
                        outcome = %s,
                        error_summary = %s,
                        duration_ms = GREATEST(
                            0,
                            FLOOR(EXTRACT(EPOCH FROM (NOW() - started_at)) * 1000)
                        )::BIGINT
                    WHERE task_id = %s
                      AND attempt_number = %s
                      AND finished_at IS NULL
                    """,
                    (next_status, stale_error, task_id, attempt_count),
                )
                results.append(updated)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return results


class TaskHeartbeat(AbstractContextManager):
    def __init__(
        self,
        *,
        task_id: int,
        worker_id: str,
        interval_seconds: float,
        logger,
    ) -> None:
        self.task_id = task_id
        self.worker_id = worker_id
        self.interval_seconds = interval_seconds
        self.logger = logger
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "TaskHeartbeat":
        if self.interval_seconds <= 0:
            return self
        self._thread = threading.Thread(
            target=self._run,
            name=f"task-heartbeat-{self.task_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_seconds, 1.0))

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            conn = None
            try:
                conn = get_conn()
                mark_task_heartbeat(
                    conn,
                    task_id=self.task_id,
                    worker_id=self.worker_id,
                )
            except Exception:
                if self.logger is not None:
                    self.logger.exception(
                        "Failed updating workflow task heartbeat | task_id=%s",
                        self.task_id,
                    )
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
