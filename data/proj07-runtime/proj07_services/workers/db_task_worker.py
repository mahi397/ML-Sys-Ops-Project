#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from proj07_services.common.feedback_common import get_conn
from proj07_services.common.task_service_common import (
    build_logger,
    env_flag,
    env_float,
    env_int,
    utcnow_iso,
)
from proj07_services.common.workflow_task_common import (
    ensure_workflow_schema,
    sweep_stale_running_tasks,
    upsert_workflow_task,
)


APP_NAME = "db_task_worker"
STAGE1_BUILD_QUEUE = "stage1_build"
STAGE1_FORWARD_QUEUE = "stage1_forward"
STAGE2_BUILD_QUEUE = "stage2_build"
STAGE2_FORWARD_QUEUE = "stage2_forward"
USER_SUMMARY_QUEUE = "user_summary_materialize"


@dataclass(frozen=True)
class DispatcherConfig:
    poll_interval_seconds: float = env_float("DB_TASK_POLL_INTERVAL_SECONDS", 5.0)
    full_scan_interval_seconds: float = env_float(
        "DB_TASK_FULL_SCAN_INTERVAL_SECONDS",
        300.0,
    )
    batch_size: int = env_int("DB_TASK_BATCH_SIZE", 25)
    full_scan_limit: int = env_int("DB_TASK_FULL_SCAN_LIMIT", 0)
    log_dir: Path = Path(
        os.getenv("DB_TASK_LOG_DIR", "/mnt/block/ingest_logs/db_task_worker")
    )
    stage1_version: int = env_int("STAGE1_ARTIFACT_VERSION", 1)
    stage1_build_max_attempts: int = env_int("STAGE1_BUILD_MAX_ATTEMPTS", 8)
    stage1_forward_max_attempts: int = env_int("STAGE1_FORWARD_MAX_ATTEMPTS", 8)
    stage2_build_max_attempts: int = env_int("STAGE2_BUILD_MAX_ATTEMPTS", 8)
    stage2_forward_max_attempts: int = env_int("STAGE2_FORWARD_MAX_ATTEMPTS", 8)
    user_summary_max_attempts: int = env_int("USER_SUMMARY_MATERIALIZE_MAX_ATTEMPTS", 8)
    stale_after_seconds: int = env_int("WORKFLOW_TASK_STALE_AFTER_SECONDS", 600)
    stage1_build_dispatch_enabled: bool = env_flag(
        "DB_TASK_STAGE1_BUILD_ENABLED",
        env_flag("DB_TASK_STAGE1_ENABLED", True),
    )
    stage1_forward_dispatch_enabled: bool = env_flag(
        "DB_TASK_STAGE1_FORWARD_ENABLED",
        env_flag("STAGE1_FORWARD_ENABLED", True),
    )
    stage2_build_dispatch_enabled: bool = env_flag(
        "DB_TASK_STAGE2_BUILD_ENABLED",
        True,
    )
    stage2_forward_dispatch_enabled: bool = env_flag(
        "DB_TASK_STAGE2_FORWARD_ENABLED",
        True,
    )
    user_summary_dispatch_enabled: bool = env_flag(
        "DB_TASK_USER_SUMMARY_ENABLED",
        True,
    )
    meeting_validity_refresh_enabled: bool = env_flag(
        "DB_TASK_MEETING_VALIDITY_ENABLED",
        True,
    )


class DispatchTask(Protocol):
    name: str

    def run_cycle(self, *, full_scan: bool) -> int:
        ...


@dataclass
class MeetingValidityRefreshTask:
    config: DispatcherConfig
    logger: object
    name: str = "meeting_validity_refresh"

    def run_cycle(self, *, full_scan: bool) -> int:
        conn = get_conn()
        try:
            refreshed = self.refresh_validity_flags(conn)
            conn.commit()
        finally:
            conn.close()

        if refreshed:
            self.logger.info("Refreshed meeting validity flags | updated=%s", refreshed)
        return refreshed

    def refresh_validity_flags(self, conn) -> int:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH validity AS (
                    SELECT
                        m.meeting_id,
                        (
                            COUNT(u.utterance_id) >= 1
                            AND raw_artifact.artifact_id IS NOT NULL
                            AND parsed_artifact.artifact_id IS NOT NULL
                            AND stage1_jsonl.artifact_id IS NOT NULL
                            AND stage1_json.artifact_id IS NOT NULL
                            AND stage1_model.artifact_id IS NOT NULL
                            AND stage1_manifest.artifact_id IS NOT NULL
                            AND stage1_resp_jsonl.artifact_id IS NOT NULL
                            AND stage1_resp_json.artifact_id IS NOT NULL
                            AND stage2_jsonl.artifact_id IS NOT NULL
                            AND stage2_json.artifact_id IS NOT NULL
                            AND reconstructed_segments.artifact_id IS NOT NULL
                            AND stage2_resp_jsonl.artifact_id IS NOT NULL
                            AND stage2_resp_json.artifact_id IS NOT NULL
                            AND summary_json.artifact_id IS NOT NULL
                        ) AS computed_is_valid
                    FROM meetings m
                    LEFT JOIN utterances u
                      ON u.meeting_id = m.meeting_id
                    LEFT JOIN meeting_artifacts raw_artifact
                      ON raw_artifact.meeting_id = m.meeting_id
                     AND raw_artifact.artifact_type = 'raw_transcript'
                     AND raw_artifact.artifact_version = %s
                    LEFT JOIN meeting_artifacts parsed_artifact
                      ON parsed_artifact.meeting_id = m.meeting_id
                     AND parsed_artifact.artifact_type = 'parsed_transcript'
                     AND parsed_artifact.artifact_version = %s
                    LEFT JOIN meeting_artifacts stage1_jsonl
                      ON stage1_jsonl.meeting_id = m.meeting_id
                     AND stage1_jsonl.artifact_type = 'stage1_requests_jsonl'
                     AND stage1_jsonl.artifact_version = %s
                    LEFT JOIN meeting_artifacts stage1_json
                      ON stage1_json.meeting_id = m.meeting_id
                     AND stage1_json.artifact_type = 'stage1_requests_json'
                     AND stage1_json.artifact_version = %s
                    LEFT JOIN meeting_artifacts stage1_model
                      ON stage1_model.meeting_id = m.meeting_id
                     AND stage1_model.artifact_type = 'stage1_model_utterances_json'
                     AND stage1_model.artifact_version = %s
                    LEFT JOIN meeting_artifacts stage1_manifest
                      ON stage1_manifest.meeting_id = m.meeting_id
                     AND stage1_manifest.artifact_type = 'stage1_manifest_json'
                     AND stage1_manifest.artifact_version = %s
                    LEFT JOIN meeting_artifacts stage1_resp_jsonl
                      ON stage1_resp_jsonl.meeting_id = m.meeting_id
                     AND stage1_resp_jsonl.artifact_type = 'stage1_responses_jsonl'
                     AND stage1_resp_jsonl.artifact_version = %s
                    LEFT JOIN meeting_artifacts stage1_resp_json
                      ON stage1_resp_json.meeting_id = m.meeting_id
                     AND stage1_resp_json.artifact_type = 'stage1_responses_json'
                     AND stage1_resp_json.artifact_version = %s
                    LEFT JOIN meeting_artifacts stage2_jsonl
                      ON stage2_jsonl.meeting_id = m.meeting_id
                     AND stage2_jsonl.artifact_type = 'stage2_inputs_jsonl'
                     AND stage2_jsonl.artifact_version = %s
                    LEFT JOIN meeting_artifacts stage2_json
                      ON stage2_json.meeting_id = m.meeting_id
                     AND stage2_json.artifact_type = 'stage2_inputs_json'
                     AND stage2_json.artifact_version = %s
                    LEFT JOIN meeting_artifacts reconstructed_segments
                      ON reconstructed_segments.meeting_id = m.meeting_id
                     AND reconstructed_segments.artifact_type = 'reconstructed_segments_json'
                     AND reconstructed_segments.artifact_version = %s
                    LEFT JOIN meeting_artifacts stage2_resp_jsonl
                      ON stage2_resp_jsonl.meeting_id = m.meeting_id
                     AND stage2_resp_jsonl.artifact_type = 'stage2_responses_jsonl'
                     AND stage2_resp_jsonl.artifact_version = %s
                    LEFT JOIN meeting_artifacts stage2_resp_json
                      ON stage2_resp_json.meeting_id = m.meeting_id
                     AND stage2_resp_json.artifact_type = 'stage2_responses_json'
                     AND stage2_resp_json.artifact_version = %s
                    LEFT JOIN meeting_artifacts summary_json
                      ON summary_json.meeting_id = m.meeting_id
                     AND summary_json.artifact_type = 'summary_json'
                     AND summary_json.artifact_version = %s
                    WHERE m.source_type = 'jitsi'
                    GROUP BY
                        m.meeting_id,
                        raw_artifact.artifact_id,
                        parsed_artifact.artifact_id,
                        stage1_jsonl.artifact_id,
                        stage1_json.artifact_id,
                        stage1_model.artifact_id,
                        stage1_manifest.artifact_id,
                        stage1_resp_jsonl.artifact_id,
                        stage1_resp_json.artifact_id,
                        stage2_jsonl.artifact_id,
                        stage2_json.artifact_id,
                        reconstructed_segments.artifact_id,
                        stage2_resp_jsonl.artifact_id,
                        stage2_resp_json.artifact_id,
                        summary_json.artifact_id
                )
                UPDATE meetings m
                SET is_valid = validity.computed_is_valid
                FROM validity
                WHERE m.meeting_id = validity.meeting_id
                  AND m.is_valid IS DISTINCT FROM validity.computed_is_valid
                RETURNING m.meeting_id
                """,
                [self.config.stage1_version] * 14,
            )
            rows = cur.fetchall()

        return len(rows)


@dataclass
class Stage1BuildDispatchTask:
    config: DispatcherConfig
    logger: object
    name: str = "stage1_build_dispatch"

    def run_cycle(self, *, full_scan: bool) -> int:
        conn = get_conn()
        try:
            meeting_ids = self.fetch_candidate_meeting_ids(conn, full_scan=full_scan)
            dispatched = 0
            for meeting_id in meeting_ids:
                payload = {
                    "task_name": STAGE1_BUILD_QUEUE,
                    "meeting_id": meeting_id,
                    "artifact_version": self.config.stage1_version,
                    "phase": "build_pending",
                    "enqueued_at": utcnow_iso(),
                }
                upsert_workflow_task(
                    conn,
                    task_type=STAGE1_BUILD_QUEUE,
                    meeting_id=meeting_id,
                    artifact_version=self.config.stage1_version,
                    payload_json=payload,
                    max_attempts=self.config.stage1_build_max_attempts,
                    commit=False,
                )
                dispatched += 1
            conn.commit()
        finally:
            conn.close()
        return dispatched

    def fetch_candidate_meeting_ids(self, conn, *, full_scan: bool) -> list[str]:
        sql = """
            SELECT m.meeting_id
            FROM meetings m
            JOIN utterances u
              ON u.meeting_id = m.meeting_id
            LEFT JOIN meeting_artifacts a_jsonl
              ON a_jsonl.meeting_id = m.meeting_id
             AND a_jsonl.artifact_type = 'stage1_requests_jsonl'
             AND a_jsonl.artifact_version = %s
            LEFT JOIN meeting_artifacts a_json
              ON a_json.meeting_id = m.meeting_id
             AND a_json.artifact_type = 'stage1_requests_json'
             AND a_json.artifact_version = %s
            LEFT JOIN meeting_artifacts a_model
              ON a_model.meeting_id = m.meeting_id
             AND a_model.artifact_type = 'stage1_model_utterances_json'
             AND a_model.artifact_version = %s
            LEFT JOIN meeting_artifacts a_manifest
              ON a_manifest.meeting_id = m.meeting_id
             AND a_manifest.artifact_type = 'stage1_manifest_json'
             AND a_manifest.artifact_version = %s
            WHERE m.source_type = 'jitsi'
            GROUP BY
                m.meeting_id,
                m.started_at,
                m.ended_at,
                a_jsonl.artifact_id,
                a_json.artifact_id,
                a_model.artifact_id,
                a_manifest.artifact_id
            HAVING COUNT(u.utterance_id) >= 1
               AND (
                    a_jsonl.artifact_id IS NULL
                    OR a_json.artifact_id IS NULL
                    OR a_model.artifact_id IS NULL
                    OR a_manifest.artifact_id IS NULL
               )
            ORDER BY COALESCE(m.ended_at, m.started_at) DESC NULLS LAST, m.meeting_id DESC
        """
        params: list[object] = [
            self.config.stage1_version,
            self.config.stage1_version,
            self.config.stage1_version,
            self.config.stage1_version,
        ]
        if full_scan and self.config.full_scan_limit > 0:
            sql += " LIMIT %s"
            params.append(self.config.full_scan_limit)
        elif not full_scan:
            sql += " LIMIT %s"
            params.append(self.config.batch_size)

        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [row["meeting_id"] for row in rows]


@dataclass
class Stage1ForwardDispatchTask:
    config: DispatcherConfig
    logger: object
    name: str = "stage1_forward_dispatch"

    def run_cycle(self, *, full_scan: bool) -> int:
        conn = get_conn()
        try:
            meeting_ids = self.fetch_candidate_meeting_ids(conn, full_scan=full_scan)
            dispatched = 0
            for meeting_id in meeting_ids:
                payload = {
                    "task_name": STAGE1_FORWARD_QUEUE,
                    "meeting_id": meeting_id,
                    "artifact_version": self.config.stage1_version,
                    "phase": "post_pending",
                    "enqueued_at": utcnow_iso(),
                }
                upsert_workflow_task(
                    conn,
                    task_type=STAGE1_FORWARD_QUEUE,
                    meeting_id=meeting_id,
                    artifact_version=self.config.stage1_version,
                    payload_json=payload,
                    max_attempts=self.config.stage1_forward_max_attempts,
                    commit=False,
                )
                dispatched += 1
            conn.commit()
        finally:
            conn.close()
        return dispatched

    def fetch_candidate_meeting_ids(self, conn, *, full_scan: bool) -> list[str]:
        sql = """
            SELECT req.meeting_id
            FROM meeting_artifacts req
            JOIN meetings m
              ON m.meeting_id = req.meeting_id
            LEFT JOIN meeting_artifacts resp
              ON resp.meeting_id = req.meeting_id
             AND resp.artifact_type = 'stage1_responses_json'
             AND resp.artifact_version = %s
            WHERE req.artifact_type = 'stage1_requests_jsonl'
              AND m.source_type = 'jitsi'
              AND req.artifact_version = %s
              AND resp.artifact_id IS NULL
            ORDER BY req.created_at DESC, req.meeting_id DESC
        """
        params: list[object] = [
            self.config.stage1_version,
            self.config.stage1_version,
        ]
        if full_scan and self.config.full_scan_limit > 0:
            sql += " LIMIT %s"
            params.append(self.config.full_scan_limit)
        elif not full_scan:
            sql += " LIMIT %s"
            params.append(self.config.batch_size)

        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [row["meeting_id"] for row in rows]


@dataclass
class Stage2BuildDispatchTask:
    config: DispatcherConfig
    logger: object
    name: str = "stage2_build_dispatch"

    def run_cycle(self, *, full_scan: bool) -> int:
        conn = get_conn()
        try:
            meeting_ids = self.fetch_candidate_meeting_ids(conn, full_scan=full_scan)
            dispatched = 0
            for meeting_id in meeting_ids:
                payload = {
                    "task_name": STAGE2_BUILD_QUEUE,
                    "meeting_id": meeting_id,
                    "artifact_version": self.config.stage1_version,
                    "phase": "stage2_build_pending",
                    "enqueued_at": utcnow_iso(),
                }
                upsert_workflow_task(
                    conn,
                    task_type=STAGE2_BUILD_QUEUE,
                    meeting_id=meeting_id,
                    artifact_version=self.config.stage1_version,
                    payload_json=payload,
                    max_attempts=self.config.stage2_build_max_attempts,
                    commit=False,
                )
                dispatched += 1
            conn.commit()
        finally:
            conn.close()
        return dispatched

    def fetch_candidate_meeting_ids(self, conn, *, full_scan: bool) -> list[str]:
        sql = """
            SELECT resp.meeting_id
            FROM meeting_artifacts resp
            JOIN meetings m
              ON m.meeting_id = resp.meeting_id
            LEFT JOIN meeting_artifacts inp_jsonl
              ON inp_jsonl.meeting_id = resp.meeting_id
             AND inp_jsonl.artifact_type = 'stage2_inputs_jsonl'
             AND inp_jsonl.artifact_version = %s
            LEFT JOIN meeting_artifacts inp_json
              ON inp_json.meeting_id = resp.meeting_id
             AND inp_json.artifact_type = 'stage2_inputs_json'
             AND inp_json.artifact_version = %s
            LEFT JOIN meeting_artifacts seg
              ON seg.meeting_id = resp.meeting_id
             AND seg.artifact_type = 'reconstructed_segments_json'
             AND seg.artifact_version = %s
            WHERE resp.artifact_type = 'stage1_responses_jsonl'
              AND m.source_type = 'jitsi'
              AND resp.artifact_version = %s
              AND (
                    inp_jsonl.artifact_id IS NULL
                    OR inp_json.artifact_id IS NULL
                    OR seg.artifact_id IS NULL
               )
            ORDER BY resp.created_at DESC, resp.meeting_id DESC
        """
        params: list[object] = [
            self.config.stage1_version,
            self.config.stage1_version,
            self.config.stage1_version,
            self.config.stage1_version,
        ]
        if full_scan and self.config.full_scan_limit > 0:
            sql += " LIMIT %s"
            params.append(self.config.full_scan_limit)
        elif not full_scan:
            sql += " LIMIT %s"
            params.append(self.config.batch_size)

        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [row["meeting_id"] for row in rows]


@dataclass
class Stage2ForwardDispatchTask:
    config: DispatcherConfig
    logger: object
    name: str = "stage2_forward_dispatch"

    def run_cycle(self, *, full_scan: bool) -> int:
        conn = get_conn()
        try:
            meeting_ids = self.fetch_candidate_meeting_ids(conn, full_scan=full_scan)
            dispatched = 0
            for meeting_id in meeting_ids:
                payload = {
                    "task_name": STAGE2_FORWARD_QUEUE,
                    "meeting_id": meeting_id,
                    "artifact_version": self.config.stage1_version,
                    "phase": "summarize_pending",
                    "enqueued_at": utcnow_iso(),
                }
                upsert_workflow_task(
                    conn,
                    task_type=STAGE2_FORWARD_QUEUE,
                    meeting_id=meeting_id,
                    artifact_version=self.config.stage1_version,
                    payload_json=payload,
                    max_attempts=self.config.stage2_forward_max_attempts,
                    commit=False,
                )
                dispatched += 1
            conn.commit()
        finally:
            conn.close()
        return dispatched

    def fetch_candidate_meeting_ids(self, conn, *, full_scan: bool) -> list[str]:
        sql = """
            SELECT req.meeting_id
            FROM meeting_artifacts req
            JOIN meetings m
              ON m.meeting_id = req.meeting_id
            LEFT JOIN meeting_artifacts resp
              ON resp.meeting_id = req.meeting_id
             AND resp.artifact_type = 'stage2_responses_json'
             AND resp.artifact_version = %s
            WHERE req.artifact_type = 'stage2_inputs_json'
              AND m.source_type = 'jitsi'
              AND req.artifact_version = %s
              AND resp.artifact_id IS NULL
            ORDER BY req.created_at DESC, req.meeting_id DESC
        """
        params: list[object] = [
            self.config.stage1_version,
            self.config.stage1_version,
        ]
        if full_scan and self.config.full_scan_limit > 0:
            sql += " LIMIT %s"
            params.append(self.config.full_scan_limit)
        elif not full_scan:
            sql += " LIMIT %s"
            params.append(self.config.batch_size)

        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [row["meeting_id"] for row in rows]


@dataclass
class UserSummaryDispatchTask:
    config: DispatcherConfig
    logger: object
    name: str = "user_summary_dispatch"

    def run_cycle(self, *, full_scan: bool) -> int:
        conn = get_conn()
        try:
            meeting_ids = self.fetch_candidate_meeting_ids(conn, full_scan=full_scan)
            dispatched = 0
            for meeting_id in meeting_ids:
                payload = {
                    "task_name": USER_SUMMARY_QUEUE,
                    "meeting_id": meeting_id,
                    "artifact_version": self.config.stage1_version,
                    "phase": "user_summary_pending",
                    "enqueued_at": utcnow_iso(),
                }
                upsert_workflow_task(
                    conn,
                    task_type=USER_SUMMARY_QUEUE,
                    meeting_id=meeting_id,
                    artifact_version=self.config.stage1_version,
                    payload_json=payload,
                    max_attempts=self.config.user_summary_max_attempts,
                    commit=False,
                )
                dispatched += 1
            conn.commit()
        finally:
            conn.close()
        return dispatched

    def fetch_candidate_meeting_ids(self, conn, *, full_scan: bool) -> list[str]:
        sql = """
            WITH latest_user_summary AS (
                SELECT
                    meeting_id,
                    MAX(created_at) AS latest_user_summary_created_at
                FROM summaries
                WHERE summary_type = 'user_edited'
                GROUP BY meeting_id
            ),
            latest_edit_feedback AS (
                SELECT
                    meeting_id,
                    MAX(created_at) AS latest_feedback_created_at
                FROM feedback_events
                WHERE event_type IN (
                    'merge_segments',
                    'split_segment',
                    'edit_topic_label',
                    'edit_summary_bullets'
                )
                  AND after_payload ? 'edit_session_id'
                GROUP BY meeting_id
            )
            SELECT feedback.meeting_id
            FROM latest_edit_feedback feedback
            JOIN meetings m
              ON m.meeting_id = feedback.meeting_id
            LEFT JOIN latest_user_summary user_summary
              ON user_summary.meeting_id = feedback.meeting_id
            WHERE m.source_type = 'jitsi'
              AND (
                    user_summary.latest_user_summary_created_at IS NULL
                    OR feedback.latest_feedback_created_at > user_summary.latest_user_summary_created_at
               )
            ORDER BY feedback.latest_feedback_created_at DESC, feedback.meeting_id DESC
        """
        params: list[object] = []
        if full_scan and self.config.full_scan_limit > 0:
            sql += " LIMIT %s"
            params.append(self.config.full_scan_limit)
        elif not full_scan:
            sql += " LIMIT %s"
            params.append(self.config.batch_size)

        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [row["meeting_id"] for row in rows]


def build_dispatch_tasks(config: DispatcherConfig, logger) -> list[DispatchTask]:
    tasks: list[DispatchTask] = []
    if config.meeting_validity_refresh_enabled:
        tasks.append(MeetingValidityRefreshTask(config=config, logger=logger))
    if config.stage1_build_dispatch_enabled:
        tasks.append(Stage1BuildDispatchTask(config=config, logger=logger))
    if config.stage1_forward_dispatch_enabled:
        tasks.append(Stage1ForwardDispatchTask(config=config, logger=logger))
    if config.stage2_build_dispatch_enabled:
        tasks.append(Stage2BuildDispatchTask(config=config, logger=logger))
    if config.stage2_forward_dispatch_enabled:
        tasks.append(Stage2ForwardDispatchTask(config=config, logger=logger))
    if config.user_summary_dispatch_enabled:
        tasks.append(UserSummaryDispatchTask(config=config, logger=logger))
    return tasks


def validate_config(config: DispatcherConfig) -> None:
    if config.poll_interval_seconds <= 0:
        raise ValueError("DB_TASK_POLL_INTERVAL_SECONDS must be > 0")
    if config.full_scan_interval_seconds <= 0:
        raise ValueError("DB_TASK_FULL_SCAN_INTERVAL_SECONDS must be > 0")
    if config.batch_size <= 0:
        raise ValueError("DB_TASK_BATCH_SIZE must be > 0")
    if config.stage1_build_max_attempts <= 0:
        raise ValueError("STAGE1_BUILD_MAX_ATTEMPTS must be > 0")
    if config.stage1_forward_max_attempts <= 0:
        raise ValueError("STAGE1_FORWARD_MAX_ATTEMPTS must be > 0")
    if config.stage2_build_max_attempts <= 0:
        raise ValueError("STAGE2_BUILD_MAX_ATTEMPTS must be > 0")
    if config.stage2_forward_max_attempts <= 0:
        raise ValueError("STAGE2_FORWARD_MAX_ATTEMPTS must be > 0")
    if config.user_summary_max_attempts <= 0:
        raise ValueError("USER_SUMMARY_MATERIALIZE_MAX_ATTEMPTS must be > 0")
    if config.stale_after_seconds <= 0:
        raise ValueError("WORKFLOW_TASK_STALE_AFTER_SECONDS must be > 0")


def main() -> None:
    config = DispatcherConfig()
    validate_config(config)
    logger = build_logger(APP_NAME, config.log_dir)
    schema_conn = get_conn()
    try:
        ensure_workflow_schema(schema_conn)
    finally:
        schema_conn.close()
    tasks = build_dispatch_tasks(config, logger)

    if not tasks:
        logger.info("No DB dispatch tasks enabled; exiting")
        return

    logger.info("Starting %s", APP_NAME)
    logger.info(
        "Dispatcher config | poll_interval=%ss full_scan_interval=%ss batch_size=%s full_scan_limit=%s version=%s stale_after=%ss",
        config.poll_interval_seconds,
        config.full_scan_interval_seconds,
        config.batch_size,
        config.full_scan_limit,
        config.stage1_version,
        config.stale_after_seconds,
    )
    logger.info(
        "Dispatch tasks | stage1_build=%s stage1_forward=%s",
        config.stage1_build_dispatch_enabled,
        config.stage1_forward_dispatch_enabled,
    )
    logger.info(
        "Dispatch tasks | stage2_build=%s",
        config.stage2_build_dispatch_enabled,
    )
    logger.info(
        "Dispatch tasks | stage2_forward=%s",
        config.stage2_forward_dispatch_enabled,
    )
    logger.info(
        "Dispatch tasks | user_summary=%s",
        config.user_summary_dispatch_enabled,
    )
    logger.info(
        "Dispatch tasks | meeting_validity_refresh=%s",
        config.meeting_validity_refresh_enabled,
    )

    next_full_scan_at = time.monotonic()
    while True:
        try:
            now = time.monotonic()
            full_scan = now >= next_full_scan_at
            if full_scan:
                next_full_scan_at = now + config.full_scan_interval_seconds
                logger.info("Running full DB dispatch scan")
                sweep_conn = get_conn()
                try:
                    stale_rows = sweep_stale_running_tasks(
                        sweep_conn,
                        stale_after_seconds=config.stale_after_seconds,
                    )
                finally:
                    sweep_conn.close()
                if stale_rows:
                    logger.warning("Recovered %s stale workflow task(s)", len(stale_rows))

            dispatched = 0
            for task in tasks:
                dispatched += task.run_cycle(full_scan=full_scan)

            if dispatched == 0:
                time.sleep(config.poll_interval_seconds)
        except KeyboardInterrupt:
            logger.info("Stopping %s", APP_NAME)
            return
        except Exception:
            logger.exception("Dispatcher loop failed; retrying after sleep")
            time.sleep(config.poll_interval_seconds)


if __name__ == "__main__":
    main()
