#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Protocol


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from build_online_inference_payloads import (  # noqa: E402
    STAGE1_ARTIFACT_FILES,
    stage1_local_artifact_paths,
)
from feedback_common import get_conn  # noqa: E402


APP_NAME = "db_task_worker"
STAGE1_TASK_NAME = "stage1_payloads"


def env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    return int(raw_value)


def env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    return float(raw_value)


def build_logger(log_dir: Path) -> logging.Logger:
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{APP_NAME}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def advisory_lock_id(task_name: str, meeting_id: str, version: int) -> int:
    digest = hashlib.blake2b(
        f"{task_name}:{meeting_id}:v{version}".encode("utf-8"),
        digest_size=8,
    ).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    value = value & 0x7FFF_FFFF_FFFF_FFFF
    return value or 1


@dataclass(frozen=True)
class WorkerConfig:
    poll_interval_seconds: float = env_float("DB_TASK_POLL_INTERVAL_SECONDS", 5.0)
    full_scan_interval_seconds: float = env_float(
        "DB_TASK_FULL_SCAN_INTERVAL_SECONDS",
        300.0,
    )
    batch_size: int = env_int("DB_TASK_BATCH_SIZE", 25)
    full_scan_limit: int = env_int("DB_TASK_FULL_SCAN_LIMIT", 0)
    failure_cooldown_seconds: float = env_float(
        "DB_TASK_FAILURE_COOLDOWN_SECONDS",
        60.0,
    )
    log_dir: Path = Path(
        os.getenv("DB_TASK_LOG_DIR", "/mnt/block/ingest_logs/db_task_worker")
    )
    stage1_enabled: bool = env_flag("DB_TASK_STAGE1_ENABLED", True)
    stage1_output_root: Path = Path(
        os.getenv("STAGE1_OUTPUT_ROOT", "/mnt/block/user-behaviour/inference_requests/stage1")
    )
    stage1_window_size: int = env_int("STAGE1_WINDOW_SIZE", 7)
    stage1_transition_index: int = env_int("STAGE1_TRANSITION_INDEX", 3)
    stage1_min_utterance_chars: int = env_int("STAGE1_MIN_UTTERANCE_CHARS", 1)
    stage1_max_words_per_utterance: int = env_int(
        "STAGE1_MAX_WORDS_PER_UTTERANCE",
        50,
    )
    stage1_upload_artifacts: bool = env_flag("STAGE1_UPLOAD_ARTIFACTS", True)
    stage1_object_prefix: str = os.getenv(
        "STAGE1_OBJECT_PREFIX",
        "production/inference_requests/stage1",
    ).strip()
    stage1_version: int = env_int("STAGE1_ARTIFACT_VERSION", 1)
    stage1_build_timeout_seconds: int = env_int(
        "STAGE1_BUILD_TIMEOUT_SECONDS",
        300,
    )


class ReconciliationTask(Protocol):
    name: str

    def run_cycle(self, *, full_scan: bool) -> int:
        ...


@dataclass
class Stage1PayloadTask:
    config: WorkerConfig
    logger: logging.Logger
    failure_backoff_until: dict[str, float] = field(default_factory=dict)
    required_artifact_types: tuple[str, ...] = field(
        default=tuple(STAGE1_ARTIFACT_FILES.keys())
    )
    builder_script: Path = field(
        default=SCRIPT_DIR / "build_online_inference_payloads.py"
    )
    name: str = STAGE1_TASK_NAME

    def run_cycle(self, *, full_scan: bool) -> int:
        processed = 0
        conn = get_conn()
        conn.autocommit = True
        try:
            meeting_ids = self.fetch_candidate_meeting_ids(conn, full_scan=full_scan)
        finally:
            conn.close()

        for meeting_id in meeting_ids:
            if self.in_failure_cooldown(meeting_id):
                continue

            lock_conn = get_conn()
            lock_conn.autocommit = True
            lock_id = advisory_lock_id(self.name, meeting_id, self.config.stage1_version)
            lock_acquired = False

            try:
                lock_acquired = self.try_advisory_lock(lock_conn, lock_id)
                if not lock_acquired:
                    continue

                needs_build, reasons = self.needs_build(lock_conn, meeting_id)
                if not needs_build:
                    continue

                self.logger.info(
                    "Reconciling Stage 1 payloads | meeting_id=%s | reasons=%s",
                    meeting_id,
                    ", ".join(reasons),
                )
                self.run_builder(meeting_id)
                self.failure_backoff_until.pop(meeting_id, None)
                processed += 1
            except Exception as exc:
                self.failure_backoff_until[meeting_id] = (
                    time.monotonic() + self.config.failure_cooldown_seconds
                )
                self.logger.exception(
                    "Stage 1 reconciliation failed | meeting_id=%s | error=%s",
                    meeting_id,
                    exc,
                )
            finally:
                if lock_acquired:
                    self.release_advisory_lock(lock_conn, lock_id)
                lock_conn.close()

        return processed

    def fetch_candidate_meeting_ids(self, conn, *, full_scan: bool) -> list[str]:
        if full_scan:
            sql = """
                SELECT m.meeting_id
                FROM meetings m
                JOIN utterances u
                  ON u.meeting_id = m.meeting_id
                GROUP BY m.meeting_id, m.started_at, m.ended_at
                HAVING COUNT(u.utterance_id) >= 1
                ORDER BY COALESCE(m.ended_at, m.started_at) DESC NULLS LAST, m.meeting_id DESC
            """
            params: list[object] = []
            if self.config.full_scan_limit > 0:
                sql += " LIMIT %s"
                params.append(self.config.full_scan_limit)
        else:
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
                LIMIT %s
            """
            params = [
                self.config.stage1_version,
                self.config.stage1_version,
                self.config.stage1_version,
                self.config.stage1_version,
                self.config.batch_size,
            ]

        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [row["meeting_id"] for row in rows]

    def needs_build(self, conn, meeting_id: str) -> tuple[bool, list[str]]:
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
                    self.config.stage1_version,
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
            self.config.stage1_output_root,
            meeting_id,
            self.config.stage1_version,
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

    def run_builder(self, meeting_id: str) -> None:
        cmd = [
            sys.executable,
            str(self.builder_script),
            "--meeting-id",
            meeting_id,
            "--window-size",
            str(self.config.stage1_window_size),
            "--transition-index",
            str(self.config.stage1_transition_index),
            "--min-utterance-chars",
            str(self.config.stage1_min_utterance_chars),
            "--max-words-per-utterance",
            str(self.config.stage1_max_words_per_utterance),
            "--output-root",
            str(self.config.stage1_output_root),
            "--version",
            str(self.config.stage1_version),
        ]
        if self.config.stage1_upload_artifacts:
            cmd.extend(
                [
                    "--upload-artifacts",
                    "--stage1-object-prefix",
                    self.config.stage1_object_prefix,
                ]
            )

        self.logger.info("Running builder: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=self.config.stage1_build_timeout_seconds,
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
            raise RuntimeError(error_message)

        if result.stdout:
            self.logger.info("Builder stdout:\n%s", result.stdout.strip())
        if result.stderr:
            self.logger.info("Builder stderr:\n%s", result.stderr.strip())

    def in_failure_cooldown(self, meeting_id: str) -> bool:
        retry_after = self.failure_backoff_until.get(meeting_id)
        if retry_after is None:
            return False
        if time.monotonic() >= retry_after:
            self.failure_backoff_until.pop(meeting_id, None)
            return False
        return True

    def try_advisory_lock(self, conn, lock_id: int) -> bool:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s) AS acquired", (lock_id,))
            row = cur.fetchone()
        return bool(row["acquired"])

    def release_advisory_lock(self, conn, lock_id: int) -> None:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))
        except Exception:
            self.logger.exception("Failed releasing advisory lock %s", lock_id)


def build_tasks(config: WorkerConfig, logger: logging.Logger) -> list[ReconciliationTask]:
    tasks: list[ReconciliationTask] = []
    if config.stage1_enabled:
        tasks.append(Stage1PayloadTask(config=config, logger=logger))
    return tasks


def validate_config(config: WorkerConfig) -> None:
    if config.poll_interval_seconds <= 0:
        raise ValueError("DB_TASK_POLL_INTERVAL_SECONDS must be > 0")
    if config.full_scan_interval_seconds <= 0:
        raise ValueError("DB_TASK_FULL_SCAN_INTERVAL_SECONDS must be > 0")
    if config.batch_size <= 0:
        raise ValueError("DB_TASK_BATCH_SIZE must be > 0")
    if config.stage1_build_timeout_seconds <= 0:
        raise ValueError("STAGE1_BUILD_TIMEOUT_SECONDS must be > 0")
    if not (0 <= config.stage1_transition_index < config.stage1_window_size):
        raise ValueError(
            "STAGE1_TRANSITION_INDEX must be between 0 and STAGE1_WINDOW_SIZE - 1"
        )


def main() -> None:
    config = WorkerConfig()
    validate_config(config)
    logger = build_logger(config.log_dir)
    tasks = build_tasks(config, logger)

    if not tasks:
        logger.info("No DB reconciliation tasks enabled; exiting")
        return

    logger.info("Starting %s", APP_NAME)
    logger.info(
        "Worker config | poll_interval=%ss full_scan_interval=%ss batch_size=%s full_scan_limit=%s",
        config.poll_interval_seconds,
        config.full_scan_interval_seconds,
        config.batch_size,
        config.full_scan_limit,
    )
    logger.info(
        "Stage 1 task | enabled=%s output_root=%s version=%s upload=%s prefix=%s",
        config.stage1_enabled,
        config.stage1_output_root.resolve(),
        config.stage1_version,
        config.stage1_upload_artifacts,
        config.stage1_object_prefix,
    )

    next_full_scan_at = time.monotonic()
    while True:
        try:
            now = time.monotonic()
            full_scan = now >= next_full_scan_at
            if full_scan:
                next_full_scan_at = now + config.full_scan_interval_seconds
                logger.info("Running full DB reconciliation scan")

            processed = 0
            for task in tasks:
                processed += task.run_cycle(full_scan=full_scan)

            if processed == 0:
                time.sleep(config.poll_interval_seconds)
        except KeyboardInterrupt:
            logger.info("Stopping %s", APP_NAME)
            return
        except Exception:
            logger.exception("Worker loop failed; retrying after sleep")
            time.sleep(config.poll_interval_seconds)


if __name__ == "__main__":
    main()
