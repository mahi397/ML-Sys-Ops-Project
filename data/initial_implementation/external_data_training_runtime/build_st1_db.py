#!/usr/bin/env python3
"""Build Stage 1 boundary-detection datasets from PostgreSQL.

This version keeps the canonical AMI transcript data in Postgres unchanged and
derives a training-only view from it:

1. Read segment-level utterances from Postgres.
2. Use clean_text only; skip utterances whose clean_text is empty.
3. Drop training utterances shorter than a minimum character threshold.
4. Split long clean_text utterances into chunks of at most max_words_per_utterance.
5. Assign proportional timestamps to split chunks.
6. Recompute topic-boundary labels from gold topic_segments using utterance midpoints.
7. Build one sliding 7-utterance window per adjacent transition.
8. Split by meeting_id into train / val / test.
9. Write dataset artifacts to block storage and upload with rclone.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# Data classes
# ============================================================

@dataclass
class SourceUtterance:
    meeting_id: str
    utterance_id: int
    utterance_index: int
    speaker_label: str
    start_time_sec: float
    end_time_sec: float
    clean_text: str


@dataclass
class TopicSegment:
    topic_segment_id: int
    meeting_id: str
    start_time_sec: float
    end_time_sec: float
    topic_label: str | None


@dataclass
class ModelUtterance:
    meeting_id: str
    model_utterance_id: str
    source_utterance_id: int
    source_utterance_index: int
    model_index: int
    speaker_label: str
    start_time_sec: float
    end_time_sec: float
    text: str
    split_part: int
    split_parts_total: int


# ============================================================
# Logging
# ============================================================

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("build_stage1_dataset")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--pg-container", default="postgres")
    parser.add_argument("--db-user", default="proj07_user")
    parser.add_argument("--db-name", default="proj07_sql_db")

    parser.add_argument(
        "--meeting-ids",
        default="",
        help="Optional comma-separated list of meeting ids to include.",
    )

    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--transition-index", type=int, default=3)

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-type", default="ami")

    parser.add_argument(
        "--min-utterance-chars",
        type=int,
        default=20,
        help="Drop training-view utterances shorter than this many characters.",
    )
    parser.add_argument(
        "--max-words-per-utterance",
        type=int,
        default=50,
        help="Split long clean_text utterances into chunks of at most this many words.",
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/block/roberta_stage1/v1"),
        help="Dataset artifact directory on mounted block storage.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("/mnt/block/roberta_stage1/logs/build_stage1_dataset.log"),
        help="Log file path on mounted block storage.",
    )

    parser.add_argument("--rclone-remote", default="rclone_s3")
    parser.add_argument("--object-bucket", default="objstore-proj07")
    parser.add_argument("--object-prefix", default="datasets/roberta_stage1/v1")
    parser.add_argument(
        "--dataset-version",
        type=int,
        default=1,
        help="Meeting-level dataset_version stamp to write for the v1 AMI snapshot.",
    )

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def ensure_rclone_available() -> None:
    if shutil.which("rclone") is None:
        raise RuntimeError("rclone is not installed or not in PATH")


def build_rclone_target(remote: str, bucket: str, prefix: str) -> str:
    prefix = prefix.strip("/")
    if prefix:
        return f"{remote}:{bucket}/{prefix}/"
    return f"{remote}:{bucket}/"


def run_command_capture(cmd: list[str], logger: logging.Logger, label: str) -> subprocess.CompletedProcess:
    logger.info("START | %s", label)
    logger.info("CMD   | %s", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        logger.error("FAIL  | %s", label)
        if result.stdout.strip():
            logger.error("STDOUT:\n%s", result.stdout)
        if result.stderr.strip():
            logger.error("STDERR:\n%s", result.stderr)
        raise RuntimeError(f"Command failed: {label}")
    logger.info("DONE  | %s", label)
    return result


def run_command_stream(cmd: list[str], logger: logging.Logger, label: str) -> None:
    logger.info("START | %s", label)
    logger.info("CMD   | %s", " ".join(cmd))

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            logger.info("[cmd] %s", line)

    return_code = process.wait()
    if return_code != 0:
        logger.error("FAIL  | %s", label)
        raise RuntimeError(f"Command failed: {label}")

    logger.info("DONE  | %s", label)


def write_jsonl(path: Path, rows: list[dict], logger: logging.Logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing JSONL: %s (%d rows)", path, len(rows))
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict, logger: logging.Logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing JSON: %s", path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def label_counts(rows: list[dict]) -> dict[str, int]:
    positives = sum(row["output"]["label"] for row in rows)
    return {
        "rows": len(rows),
        "positive": positives,
        "negative": len(rows) - positives,
    }


def pick_examples(rows: list[dict]) -> dict[str, dict | None]:
    positive = next((row for row in rows if row["output"]["label"] == 1), None)
    negative = next((row for row in rows if row["output"]["label"] == 0), None)
    return {
        "positive": positive,
        "negative": negative,
    }


# ============================================================
# Postgres reads
# ============================================================

def run_psql_query(
    pg_container: str,
    db_user: str,
    db_name: str,
    sql: str,
    logger: logging.Logger,
    label: str,
) -> str:
    cmd = [
        "docker",
        "exec",
        pg_container,
        "psql",
        "-U",
        db_user,
        "-d",
        db_name,
        "-v",
        "ON_ERROR_STOP=1",
        "-At",
        "-F",
        "\t",
        "-c",
        sql,
    ]
    result = run_command_capture(cmd, logger, f"psql query: {label}")
    return result.stdout


def run_psql_script(
    pg_container: str,
    db_user: str,
    db_name: str,
    sql: str,
    logger: logging.Logger,
    label: str,
) -> None:
    cmd = [
        "docker",
        "exec",
        pg_container,
        "psql",
        "-U",
        db_user,
        "-d",
        db_name,
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        sql,
    ]
    run_command_capture(cmd, logger, f"psql write: {label}")


def run_psql_copy_csv(
    pg_container: str,
    db_user: str,
    db_name: str,
    query: str,
    logger: logging.Logger,
    label: str,
) -> list[dict[str, str]]:
    cleaned_query = query.strip().rstrip(";").strip()
    sql = f"COPY ({cleaned_query}) TO STDOUT WITH CSV HEADER"

    cmd = [
        "docker",
        "exec",
        pg_container,
        "psql",
        "-U",
        db_user,
        "-d",
        db_name,
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        sql,
    ]

    result = run_command_capture(cmd, logger, f"psql copy: {label}")
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    logger.info("Fetched %d row(s) for %s", len(rows), label)
    return rows


def discover_meeting_ids(
    pg_container: str,
    db_user: str,
    db_name: str,
    source_type: str,
    explicit_meeting_ids: list[str],
    logger: logging.Logger,
) -> list[str]:
    if explicit_meeting_ids:
        logger.info("Using explicit meeting ids (%d): %s", len(explicit_meeting_ids), ", ".join(explicit_meeting_ids))
        return explicit_meeting_ids

    sql = f"""
        SELECT DISTINCT m.meeting_id
        FROM meetings m
        JOIN topic_segments ts
          ON ts.meeting_id = m.meeting_id
        WHERE m.source_type = {sql_quote(source_type)}
          AND ts.segment_type = 'gold'
        ORDER BY m.meeting_id;
    """
    raw = run_psql_query(pg_container, db_user, db_name, sql, logger, "discover meetings")
    meeting_ids = [line.strip() for line in raw.splitlines() if line.strip()]
    logger.info("Discovered %d meeting(s) for source_type=%s", len(meeting_ids), source_type)
    return meeting_ids


def fetch_utterances(
    pg_container: str,
    db_user: str,
    db_name: str,
    meeting_ids: list[str],
    logger: logging.Logger,
) -> list[SourceUtterance]:
    meeting_list = ", ".join(sql_quote(meeting_id) for meeting_id in meeting_ids)
    query = f"""
        SELECT
            u.meeting_id,
            u.utterance_id,
            u.utterance_index,
            ms.speaker_label,
            u.start_time_sec,
            u.end_time_sec,
            COALESCE(u.clean_text, '') AS clean_text
        FROM utterances u
        JOIN meeting_speakers ms
          ON ms.meeting_speaker_id = u.meeting_speaker_id
         AND ms.meeting_id = u.meeting_id
        WHERE u.meeting_id IN ({meeting_list})
        ORDER BY u.meeting_id, u.utterance_index
    """
    rows = run_psql_copy_csv(pg_container, db_user, db_name, query, logger, "fetch utterances")

    utterances: list[SourceUtterance] = []
    for row in rows:
        utterances.append(
            SourceUtterance(
                meeting_id=row["meeting_id"],
                utterance_id=int(row["utterance_id"]),
                utterance_index=int(row["utterance_index"]),
                speaker_label=row["speaker_label"],
                start_time_sec=float(row["start_time_sec"]),
                end_time_sec=float(row["end_time_sec"]),
                clean_text=row["clean_text"],
            )
        )
    logger.info("Built %d source utterance objects", len(utterances))
    return utterances


def fetch_topic_segments(
    pg_container: str,
    db_user: str,
    db_name: str,
    meeting_ids: list[str],
    logger: logging.Logger,
) -> list[TopicSegment]:
    meeting_list = ", ".join(sql_quote(meeting_id) for meeting_id in meeting_ids)
    query = f"""
        SELECT
            topic_segment_id,
            meeting_id,
            start_time_sec,
            end_time_sec,
            topic_label
        FROM topic_segments
        WHERE meeting_id IN ({meeting_list})
          AND segment_type = 'gold'
        ORDER BY meeting_id, start_time_sec, end_time_sec, topic_segment_id
    """
    rows = run_psql_copy_csv(pg_container, db_user, db_name, query, logger, "fetch topic segments")

    segments: list[TopicSegment] = []
    for row in rows:
        segments.append(
            TopicSegment(
                topic_segment_id=int(row["topic_segment_id"]),
                meeting_id=row["meeting_id"],
                start_time_sec=float(row["start_time_sec"]),
                end_time_sec=float(row["end_time_sec"]),
                topic_label=row["topic_label"] if row["topic_label"] else None,
            )
        )
    logger.info("Built %d topic segment objects", len(segments))
    return segments


# ============================================================
# Training-view utterance derivation
# ============================================================

def split_source_utterance(
    source: SourceUtterance,
    max_words_per_utterance: int,
    min_utterance_chars: int,
) -> list[ModelUtterance]:
    text = source.clean_text.strip()
    if not text:
        return []

    words = text.split()
    if not words:
        return []

    duration = max(source.end_time_sec - source.start_time_sec, 0.0)
    total_words = len(words)
    chunks = [words[i:i + max_words_per_utterance] for i in range(0, total_words, max_words_per_utterance)]

    derived: list[ModelUtterance] = []
    words_before = 0
    total_parts = len(chunks)

    for part_idx, chunk_words in enumerate(chunks, start=1):
        chunk_text = " ".join(chunk_words).strip()
        words_after = words_before + len(chunk_words)

        if total_words > 0 and duration > 0:
            chunk_start = source.start_time_sec + (words_before / total_words) * duration
            chunk_end = source.start_time_sec + (words_after / total_words) * duration
        else:
            chunk_start = source.start_time_sec
            chunk_end = source.end_time_sec

        words_before = words_after

        if len(chunk_text) < min_utterance_chars:
            continue

        derived.append(
            ModelUtterance(
                meeting_id=source.meeting_id,
                model_utterance_id=f"{source.utterance_id}_part{part_idx}",
                source_utterance_id=source.utterance_id,
                source_utterance_index=source.utterance_index,
                model_index=-1,
                speaker_label=source.speaker_label,
                start_time_sec=round(chunk_start, 3),
                end_time_sec=round(chunk_end, 3),
                text=chunk_text,
                split_part=part_idx,
                split_parts_total=total_parts,
            )
        )

    return derived


def build_model_utterances_by_meeting(
    source_utterances: list[SourceUtterance],
    max_words_per_utterance: int,
    min_utterance_chars: int,
    logger: logging.Logger,
) -> dict[str, list[ModelUtterance]]:
    source_by_meeting: dict[str, list[SourceUtterance]] = defaultdict(list)
    for row in source_utterances:
        source_by_meeting[row.meeting_id].append(row)

    model_by_meeting: dict[str, list[ModelUtterance]] = {}

    for meeting_id, meeting_rows in source_by_meeting.items():
        ordered = sorted(
            meeting_rows,
            key=lambda row: (
                row.start_time_sec,
                row.end_time_sec,
                row.utterance_index,
                row.utterance_id,
            ),
        )

        derived: list[ModelUtterance] = []
        skipped_empty = 0
        skipped_short = 0
        split_expansions = 0

        for row in ordered:
            if not row.clean_text.strip():
                skipped_empty += 1
                continue

            chunks = split_source_utterance(
                row,
                max_words_per_utterance=max_words_per_utterance,
                min_utterance_chars=min_utterance_chars,
            )

            if not chunks:
                skipped_short += 1
                continue

            if len(chunks) > 1:
                split_expansions += len(chunks) - 1

            derived.extend(chunks)

        # Re-sort after splitting so chronology is based on actual derived times,
        # not just append order from the source utterances.
        derived.sort(
            key=lambda row: (
                row.start_time_sec,
                row.end_time_sec,
                row.source_utterance_index,
                row.split_part,
                row.source_utterance_id,
            )
        )

        for idx, row in enumerate(derived):
            row.model_index = idx

        model_by_meeting[meeting_id] = derived
        logger.info(
            "Derived training utterances for meeting=%s | source=%d kept=%d skipped_empty=%d skipped_short=%d split_expansions=%d",
            meeting_id,
            len(ordered),
            len(derived),
            skipped_empty,
            skipped_short,
            split_expansions,
        )

    return model_by_meeting


# ============================================================
# Topic assignment + window build
# ============================================================

def assign_topic_segment_id(
    utterance: ModelUtterance,
    topic_segments: list[TopicSegment],
) -> int:
    midpoint = (utterance.start_time_sec + utterance.end_time_sec) / 2.0

    for segment in topic_segments:
        if segment.start_time_sec <= midpoint <= segment.end_time_sec:
            return segment.topic_segment_id

    nearest = min(
        topic_segments,
        key=lambda segment: abs(segment.start_time_sec - midpoint),
    )
    return nearest.topic_segment_id


def make_padding(position: int) -> dict:
    return {
        "position": position,
        "speaker": None,
        "start_time_sec": None,
        "end_time_sec": None,
        "text": "",
        "source_utterance_id": None,
        "model_utterance_id": None,
        "is_padding": True,
    }


def make_window_entry(position: int, utterance: ModelUtterance) -> dict:
    return {
        "position": position,
        "speaker": utterance.speaker_label,
        "start_time_sec": utterance.start_time_sec,
        "end_time_sec": utterance.end_time_sec,
        "text": utterance.text,
        "source_utterance_id": utterance.source_utterance_id,
        "model_utterance_id": utterance.model_utterance_id,
        "is_padding": False,
    }


def build_window_rows(
    model_utterances_by_meeting: dict[str, list[ModelUtterance]],
    topic_segments_by_meeting: dict[str, list[TopicSegment]],
    window_size: int,
    transition_index: int,
    logger: logging.Logger,
) -> list[dict]:
    if transition_index < 0 or transition_index >= window_size - 1:
        raise ValueError("transition_index must point to a valid gap inside the window")

    rows: list[dict] = []

    for meeting_id, utterances in model_utterances_by_meeting.items():
        if len(utterances) < window_size:
            logger.info(
                "Skipping meeting=%s because derived utterances=%d < window_size=%d",
                meeting_id,
                len(utterances),
                window_size,
            )
            continue

        topic_segments = topic_segments_by_meeting.get(meeting_id, [])
        if not topic_segments:
            logger.info("Skipping meeting=%s because it has no gold topic segments", meeting_id)
            continue

        topic_id_by_model_index = {
            utterance.model_index: assign_topic_segment_id(utterance, topic_segments)
            for utterance in utterances
        }

        logger.info(
            "Building sliding windows for meeting=%s | derived_utterances=%d | topic_segments=%d",
            meeting_id,
            len(utterances),
            len(topic_segments),
        )

        for left_idx in range(len(utterances) - 1):
            right_idx = left_idx + 1
            left = utterances[left_idx]
            right = utterances[right_idx]

            start_index = left_idx - transition_index
            window: list[dict] = []

            for pos in range(window_size):
                utterance_index = start_index + pos
                if 0 <= utterance_index < len(utterances):
                    window.append(make_window_entry(pos, utterances[utterance_index]))
                else:
                    window.append(make_padding(pos))

            first_real_start = next(
                (item["start_time_sec"] for item in window if item["start_time_sec"] is not None),
                0.0,
            )

            left_topic_segment_id = topic_id_by_model_index[left.model_index]
            right_topic_segment_id = topic_id_by_model_index[right.model_index]
            label = 1 if left_topic_segment_id != right_topic_segment_id else 0

            rows.append(
                {
                    "input": {
                        "meeting_id": meeting_id,
                        "window": window,
                        "transition_index": transition_index,
                        "meeting_offset_seconds": first_real_start,
                    },
                    "output": {
                        "label": label,
                    },
                    "metadata": {
                        "left_model_index": left.model_index,
                        "right_model_index": right.model_index,
                        "left_model_utterance_id": left.model_utterance_id,
                        "right_model_utterance_id": right.model_utterance_id,
                        "left_source_utterance_id": left.source_utterance_id,
                        "right_source_utterance_id": right.source_utterance_id,
                        "left_topic_segment_id": left_topic_segment_id,
                        "right_topic_segment_id": right_topic_segment_id,
                    },
                }
            )

    logger.info("Built %d total dataset row(s)", len(rows))
    return rows


# ============================================================
# Split
# ============================================================

def split_meetings(
    meeting_ids: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    if not meeting_ids:
        return {"train": [], "val": [], "test": []}

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train/val/test ratios must sum to 1.0")

    ids = meeting_ids[:]
    random.Random(seed).shuffle(ids)
    n = len(ids)

    if n == 1:
        return {"train": ids, "val": [], "test": []}
    if n == 2:
        return {"train": [ids[0]], "val": [], "test": [ids[1]]}

    test_count = max(1, round(n * test_ratio))
    val_count = max(1, round(n * val_ratio))

    if test_count + val_count >= n:
        test_count = 1
        val_count = 1

    train_count = n - test_count - val_count

    return {
        "train": ids[:train_count],
        "val": ids[train_count:train_count + val_count],
        "test": ids[train_count + val_count:],
    }


def stamp_meeting_dataset_assignments(
    pg_container: str,
    db_user: str,
    db_name: str,
    split_map: dict[str, list[str]],
    dataset_version: int,
    logger: logging.Logger,
) -> None:
    statements: list[str] = ["BEGIN;"]

    for split_name in ("train", "val", "test"):
        meeting_ids = split_map.get(split_name, [])
        if not meeting_ids:
            continue
        meeting_list = ", ".join(sql_quote(meeting_id) for meeting_id in meeting_ids)
        statements.append(
            "UPDATE meetings "
            f"SET dataset_version = {dataset_version}, dataset_split = {sql_quote(split_name)} "
            f"WHERE meeting_id IN ({meeting_list});"
        )

    statements.append("COMMIT;")
    run_psql_script(
        pg_container=pg_container,
        db_user=db_user,
        db_name=db_name,
        sql="\n".join(statements),
        logger=logger,
        label=f"stamp meetings for dataset v{dataset_version}",
    )


# ============================================================
# Upload
# ============================================================

def upload_dataset(
    dataset_root: Path,
    rclone_remote: str,
    object_bucket: str,
    object_prefix: str,
    logger: logging.Logger,
) -> str:
    ensure_rclone_available()

    if not dataset_root.exists():
        raise RuntimeError(f"Dataset root does not exist: {dataset_root}")

    target = build_rclone_target(rclone_remote, object_bucket, object_prefix)

    cmd = [
        "rclone",
        "copy",
        str(dataset_root),
        target,
        "-P",
        "--stats",
        "10s",
    ]
    run_command_stream(cmd, logger, f"upload dataset to {target}")
    logger.info("Upload complete: %s", target)
    return target


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_file)

    logger.info("============================================================")
    logger.info("Stage 1 dataset build started")
    logger.info("dataset_root            = %s", args.dataset_root)
    logger.info("log_file                = %s", args.log_file)
    logger.info("pg_container            = %s", args.pg_container)
    logger.info("db_name                 = %s", args.db_name)
    logger.info("source_type             = %s", args.source_type)
    logger.info("window_size             = %s", args.window_size)
    logger.info("transition_index        = %s", args.transition_index)
    logger.info("min_utterance_chars     = %s", args.min_utterance_chars)
    logger.info("max_words_per_utterance = %s", args.max_words_per_utterance)
    logger.info("split ratios            = train=%.3f val=%.3f test=%.3f", args.train_ratio, args.val_ratio, args.test_ratio)
    logger.info("seed                    = %s", args.seed)
    logger.info("object target           = %s", build_rclone_target(args.rclone_remote, args.object_bucket, args.object_prefix))
    logger.info("============================================================")

    meeting_ids = [m.strip() for m in args.meeting_ids.split(",") if m.strip()]
    selected_meetings = discover_meeting_ids(
        args.pg_container,
        args.db_user,
        args.db_name,
        args.source_type,
        meeting_ids,
        logger,
    )
    if not selected_meetings:
        raise RuntimeError("No meetings found for dataset build")

    source_utterances = fetch_utterances(
        args.pg_container,
        args.db_user,
        args.db_name,
        selected_meetings,
        logger,
    )
    topic_segments = fetch_topic_segments(
        args.pg_container,
        args.db_user,
        args.db_name,
        selected_meetings,
        logger,
    )

    topic_segments_by_meeting: dict[str, list[TopicSegment]] = defaultdict(list)
    for segment in topic_segments:
        topic_segments_by_meeting[segment.meeting_id].append(segment)

    model_utterances_by_meeting = build_model_utterances_by_meeting(
        source_utterances=source_utterances,
        max_words_per_utterance=args.max_words_per_utterance,
        min_utterance_chars=args.min_utterance_chars,
        logger=logger,
    )

    eligible_meetings = sorted(
        meeting_id
        for meeting_id, utterances in model_utterances_by_meeting.items()
        if len(utterances) >= args.window_size and len(topic_segments_by_meeting.get(meeting_id, [])) > 0
    )
    logger.info("Eligible meetings after training-view preprocessing: %d", len(eligible_meetings))

    filtered_model_utterances_by_meeting = {
        meeting_id: model_utterances_by_meeting[meeting_id]
        for meeting_id in eligible_meetings
    }
    filtered_topic_segments_by_meeting = {
        meeting_id: topic_segments_by_meeting[meeting_id]
        for meeting_id in eligible_meetings
    }

    dataset_rows = build_window_rows(
        model_utterances_by_meeting=filtered_model_utterances_by_meeting,
        topic_segments_by_meeting=filtered_topic_segments_by_meeting,
        window_size=args.window_size,
        transition_index=args.transition_index,
        logger=logger,
    )

    split_map = split_meetings(
        eligible_meetings,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    logger.info(
        "Meeting split counts | train=%d val=%d test=%d total=%d",
        len(split_map["train"]),
        len(split_map["val"]),
        len(split_map["test"]),
        len(eligible_meetings),
    )

    split_sets = {name: set(ids) for name, ids in split_map.items()}
    split_rows = {
        "train": [row for row in dataset_rows if row["input"]["meeting_id"] in split_sets["train"]],
        "val": [row for row in dataset_rows if row["input"]["meeting_id"] in split_sets["val"]],
        "test": [row for row in dataset_rows if row["input"]["meeting_id"] in split_sets["test"]],
    }

    logger.info(
        "Row split counts | train=%d val=%d test=%d total=%d",
        len(split_rows["train"]),
        len(split_rows["val"]),
        len(split_rows["test"]),
        len(dataset_rows),
    )

    args.dataset_root.mkdir(parents=True, exist_ok=True)

    write_jsonl(args.dataset_root / "train.jsonl", split_rows["train"], logger)
    write_jsonl(args.dataset_root / "val.jsonl", split_rows["val"], logger)
    write_jsonl(args.dataset_root / "test.jsonl", split_rows["test"], logger)

    split_info = {
        "seed": args.seed,
        "source_type": args.source_type,
        "dataset_version": args.dataset_version,
        "meeting_ids": split_map,
    }
    write_json(args.dataset_root / "split_info.json", split_info, logger)

    stamp_meeting_dataset_assignments(
        pg_container=args.pg_container,
        db_user=args.db_user,
        db_name=args.db_name,
        split_map=split_map,
        dataset_version=args.dataset_version,
        logger=logger,
    )

    examples = {split: pick_examples(rows) for split, rows in split_rows.items()}
    write_json(args.dataset_root / "examples.json", examples, logger)

    object_target = build_rclone_target(args.rclone_remote, args.object_bucket, args.object_prefix)
    manifest = {
        "dataset_name": "roberta_stage1_boundary",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "db_name": args.db_name,
            "pg_container": args.pg_container,
            "source_type": args.source_type,
            "source_unit": "ami_segments_in_postgres",
            "label_source": "gold_topic_segments",
        },
        "params": {
            "dataset_version": args.dataset_version,
            "window_size": args.window_size,
            "transition_index": args.transition_index,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "clean_text_only": True,
            "drop_empty_clean_text": True,
            "min_utterance_chars": args.min_utterance_chars,
            "max_words_per_utterance": args.max_words_per_utterance,
        },
        "meetings": {
            "discovered": len(selected_meetings),
            "eligible": len(eligible_meetings),
            "train": len(split_map["train"]),
            "val": len(split_map["val"]),
            "test": len(split_map["test"]),
        },
        "rows": {
            "total": len(dataset_rows),
            "train": label_counts(split_rows["train"]),
            "val": label_counts(split_rows["val"]),
            "test": label_counts(split_rows["test"]),
        },
        "artifacts": {
            "dataset_root": str(args.dataset_root),
            "object_storage_target": object_target,
            "log_file": str(args.log_file),
        },
        "meeting_state": {
            "dataset_version_written_to_meetings": args.dataset_version,
            "dataset_split_written_to_meetings": True,
            "note": "AMI meetings are stamped at v1 snapshot creation time. The later v2 augmentation reuses the same meeting split without changing per-meeting membership.",
        },
    }
    write_json(args.dataset_root / "manifest.json", manifest, logger)

    uploaded_target = upload_dataset(
        dataset_root=args.dataset_root,
        rclone_remote=args.rclone_remote,
        object_bucket=args.object_bucket,
        object_prefix=args.object_prefix,
        logger=logger,
    )

    logger.info("SUCCESS | Stage 1 dataset build completed")
    logger.info("Artifacts on block storage: %s", args.dataset_root)
    logger.info("Artifacts in object storage: %s", uploaded_target)
    logger.info("Log file: %s", args.log_file)


if __name__ == "__main__":
    main()
