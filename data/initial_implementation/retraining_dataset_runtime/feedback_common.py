from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json


def env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_local_tmp_root() -> Path:
    return Path(os.getenv("LOCAL_TMP_ROOT", "/mnt/block/staging/feedback_loop"))


def get_log_root() -> Path:
    log_root = os.getenv("LOG_ROOT")
    if log_root:
        return Path(log_root)
    return Path(os.getenv("LOCAL_TMP_ROOT", "/mnt/block/staging/feedback_loop")) / "logs"


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    log_path = get_log_root() / f"{name}.log"
    ensure_dir(log_path.parent)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_conn():
    return psycopg.connect(env("DATABASE_URL"), row_factory=dict_row)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_object_uri(object_key: str) -> str:
    return f"{env('RCLONE_REMOTE')}:{env('BUCKET')}/{object_key}"


def run_command(cmd: list[str], logger: logging.Logger, label: str) -> None:
    logger.info("START | %s", label)
    logger.info("CMD   | %s", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        logger.error("STDOUT:\n%s", result.stdout)
        logger.error("STDERR:\n%s", result.stderr)
        raise RuntimeError(f"Command failed: {label}")
    logger.info("DONE  | %s", label)


def upload_file(local_path: Path, object_key: str, logger: logging.Logger) -> None:
    cmd = ["rclone", "copyto", str(local_path), build_object_uri(object_key), "-P"]
    run_command(cmd, logger, f"upload file {local_path.name}")


def upload_dir(local_dir: Path, object_prefix: str, logger: logging.Logger) -> None:
    cmd = ["rclone", "copy", str(local_dir), build_object_uri(object_prefix), "-P"]
    run_command(cmd, logger, f"upload dir {local_dir}")


def stable_split_70_15_15(meeting_ids: list[str]) -> dict[str, list[str]]:
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []

    for meeting_id in sorted(set(meeting_ids)):
        bucket = int(hashlib.md5(meeting_id.encode()).hexdigest(), 16) % 20
        if bucket < 14:
            train.append(meeting_id)
        elif bucket < 17:
            val.append(meeting_id)
        else:
            test.append(meeting_id)

    return {"train": train, "val": val, "test": test}


def insert_dataset_version(
    conn,
    dataset_name: str,
    stage: str,
    source_type: str,
    object_key: str,
    manifest_json: dict[str, Any],
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO dataset_versions (
                dataset_name, stage, source_type, object_key, manifest_json
            )
            VALUES (%s, %s, %s, %s, %s)
            """,
            (dataset_name, stage, source_type, object_key, Json(manifest_json)),
        )
    conn.commit()


def fetch_source_utterances(conn, meeting_ids: list[str]) -> list[dict]:
    if not meeting_ids:
        return []

    with conn.cursor() as cur:
        cur.execute(
            """
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
            WHERE u.meeting_id = ANY(%s)
            ORDER BY u.meeting_id, u.utterance_index
            """,
            (meeting_ids,),
        )
        return cur.fetchall()


def fetch_meeting_utterance_lookup(conn, meeting_id: str) -> dict[int, dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                u.utterance_id,
                u.utterance_index,
                u.start_time_sec,
                u.end_time_sec,
                COALESCE(u.clean_text, '') AS clean_text,
                ms.speaker_label
            FROM utterances u
            JOIN meeting_speakers ms
              ON ms.meeting_speaker_id = u.meeting_speaker_id
            WHERE u.meeting_id = %s
            ORDER BY u.utterance_index
            """,
            (meeting_id,),
        )
        rows = cur.fetchall()

    return {row["utterance_index"]: row for row in rows}


def fetch_topic_segments(conn, meeting_ids: list[str], segment_type: str) -> list[dict]:
    if not meeting_ids:
        return []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                topic_segment_id,
                meeting_id,
                start_time_sec,
                end_time_sec,
                topic_label
            FROM topic_segments
            WHERE meeting_id = ANY(%s)
              AND segment_type = %s
            ORDER BY meeting_id, start_time_sec, end_time_sec, topic_segment_id
            """,
            (meeting_ids, segment_type),
        )
        return cur.fetchall()


def normalize_segments_by_next_start(
    segments: list[dict],
    utterance_lookup: dict[int, dict],
) -> list[dict]:
    if not segments or not utterance_lookup:
        return []

    ordered = sorted(
        segments,
        key=lambda seg: (
            seg["segment_index"],
            seg["start_utterance_index"],
            seg["end_utterance_index"],
        ),
    )

    max_utterance_index = max(utterance_lookup)
    normalized: list[dict] = []

    for idx, seg in enumerate(ordered):
        start_idx = seg["start_utterance_index"]
        if start_idx not in utterance_lookup:
            continue

        if idx + 1 < len(ordered):
            next_start_idx = ordered[idx + 1]["start_utterance_index"]
            end_idx = next_start_idx - 1
            if end_idx < start_idx:
                end_idx = start_idx
            end_idx = min(end_idx, max_utterance_index)
            t_end = utterance_lookup.get(next_start_idx, utterance_lookup[end_idx])["start_time_sec"]
        else:
            end_idx = min(seg["end_utterance_index"], max_utterance_index)
            if end_idx < start_idx:
                end_idx = start_idx
            t_end = utterance_lookup[end_idx]["end_time_sec"]

        normalized.append(
            {
                **seg,
                "segment_index": idx + 1,
                "start_utterance_index": start_idx,
                "end_utterance_index": end_idx,
                "t_start": utterance_lookup[start_idx]["start_time_sec"],
                "t_end": t_end,
            }
        )

    return normalized


def build_emulated_summary_bullets(topic_label: str, utterances: list[dict]) -> list[str]:
    speakers: list[str] = []
    for utt in utterances:
        speaker = utt["speaker"]
        if speaker not in speakers:
            speakers.append(speaker)

    bullets = [f"The discussion focused on {topic_label}."]

    if speakers:
        if len(speakers) == 1:
            bullets.append(f"Speaker {speakers[0]} led this portion of the conversation.")
        elif len(speakers) == 2:
            bullets.append(f"Speakers {speakers[0]} and {speakers[1]} both contributed to this segment.")
        else:
            bullets.append(
                f"Multiple speakers contributed to this segment, including {', '.join(speakers[:3])}."
            )
    else:
        bullets.append("The segment contains concise discussion without many participant turns.")

    if len(utterances) >= 8:
        bullets.append("The team explored the topic through an extended multi-turn exchange.")
    elif len(utterances) >= 4:
        bullets.append("The group covered the topic through a short focused exchange.")
    else:
        bullets.append("The topic was discussed briefly before the meeting moved forward.")

    return bullets[:3]


def split_source_utterance(row: dict, max_words: int, min_chars: int) -> list[dict]:
    text = (row["clean_text"] or "").strip()
    if not text:
        return []

    words = text.split()
    if not words:
        return []

    total_words = len(words)
    total_parts = (total_words + max_words - 1) // max_words
    duration = max(row["end_time_sec"] - row["start_time_sec"], 0.0)

    chunks: list[dict] = []
    words_before = 0

    for part_idx, start in enumerate(range(0, total_words, max_words), start=1):
        chunk_words = words[start:start + max_words]
        chunk_text = " ".join(chunk_words).strip()
        words_after = words_before + len(chunk_words)

        if duration > 0 and total_words > 0:
            chunk_start = row["start_time_sec"] + (words_before / total_words) * duration
            chunk_end = row["start_time_sec"] + (words_after / total_words) * duration
        else:
            chunk_start = row["start_time_sec"]
            chunk_end = row["end_time_sec"]

        words_before = words_after

        if len(chunk_text) < min_chars:
            continue

        chunks.append(
            {
                "meeting_id": row["meeting_id"],
                "model_utterance_id": f"{row['utterance_id']}_part{part_idx}",
                "source_utterance_id": row["utterance_id"],
                "source_utterance_index": row["utterance_index"],
                "speaker_label": row["speaker_label"],
                "start_time_sec": round(chunk_start, 3),
                "end_time_sec": round(chunk_end, 3),
                "text": chunk_text,
                "split_part": part_idx,
                "split_parts_total": total_parts,
            }
        )

    return chunks


def build_model_utterances_by_meeting(
    source_rows: list[dict],
    max_words: int,
    min_chars: int,
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in source_rows:
        grouped[row["meeting_id"]].append(row)

    result: dict[str, list[dict]] = {}

    for meeting_id, rows in grouped.items():
        ordered = sorted(
            rows,
            key=lambda r: (
                r["start_time_sec"],
                r["end_time_sec"],
                r["utterance_index"],
                r["utterance_id"],
            ),
        )

        derived: list[dict] = []
        for row in ordered:
            chunks = split_source_utterance(row, max_words=max_words, min_chars=min_chars)
            derived.extend(chunks)

        derived.sort(
            key=lambda r: (
                r["start_time_sec"],
                r["end_time_sec"],
                r["source_utterance_index"],
                r["split_part"],
                r["source_utterance_id"],
            )
        )

        for idx, row in enumerate(derived):
            row["model_index"] = idx

        result[meeting_id] = derived

    return result


def assign_topic_segment_id(utterance: dict, topic_segments: list[dict]) -> int:
    midpoint = (utterance["start_time_sec"] + utterance["end_time_sec"]) / 2.0

    for segment in topic_segments:
        if segment["start_time_sec"] <= midpoint <= segment["end_time_sec"]:
            return segment["topic_segment_id"]

    nearest = min(
        topic_segments,
        key=lambda s: abs(s["start_time_sec"] - midpoint),
    )
    return nearest["topic_segment_id"]


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


def make_window_entry(position: int, utterance: dict) -> dict:
    return {
        "position": position,
        "speaker": utterance["speaker_label"],
        "start_time_sec": utterance["start_time_sec"],
        "end_time_sec": utterance["end_time_sec"],
        "text": utterance["text"],
        "source_utterance_id": utterance["source_utterance_id"],
        "model_utterance_id": utterance["model_utterance_id"],
        "is_padding": False,
    }


def build_stage1_rows(
    model_utterances_by_meeting: dict[str, list[dict]],
    topic_segments_by_meeting: dict[str, list[dict]],
    window_size: int,
    transition_index: int,
) -> list[dict]:
    rows: list[dict] = []

    for meeting_id, utterances in model_utterances_by_meeting.items():
        if len(utterances) < window_size:
            continue

        topic_segments = topic_segments_by_meeting.get(meeting_id, [])
        if not topic_segments:
            continue

        topic_id_by_model_index = {
            utt["model_index"]: assign_topic_segment_id(utt, topic_segments)
            for utt in utterances
        }

        for left_idx in range(len(utterances) - 1):
            right_idx = left_idx + 1
            left = utterances[left_idx]
            right = utterances[right_idx]

            start_index = left_idx - transition_index
            window: list[dict] = []

            for pos in range(window_size):
                idx = start_index + pos
                if 0 <= idx < len(utterances):
                    window.append(make_window_entry(pos, utterances[idx]))
                else:
                    window.append(make_padding(pos))

            first_real_start = next(
                (item["start_time_sec"] for item in window if item["start_time_sec"] is not None),
                0.0,
            )

            left_segment_id = topic_id_by_model_index[left["model_index"]]
            right_segment_id = topic_id_by_model_index[right["model_index"]]
            label = 1 if left_segment_id != right_segment_id else 0

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
                        "left_model_index": left["model_index"],
                        "right_model_index": right["model_index"],
                        "left_model_utterance_id": left["model_utterance_id"],
                        "right_model_utterance_id": right["model_utterance_id"],
                        "left_source_utterance_id": left["source_utterance_id"],
                        "right_source_utterance_id": right["source_utterance_id"],
                        "left_topic_segment_id": left_segment_id,
                        "right_topic_segment_id": right_segment_id,
                    },
                }
            )

    return rows


def label_counts(rows: list[dict]) -> dict[str, int]:
    positives = sum(row["output"]["label"] for row in rows)
    return {
        "rows": len(rows),
        "positive": positives,
        "negative": len(rows) - positives,
    }


def pick_stage1_examples(rows: list[dict]) -> dict[str, dict | None]:
    positive = next((row for row in rows if row["output"]["label"] == 1), None)
    negative = next((row for row in rows if row["output"]["label"] == 0), None)
    return {"positive": positive, "negative": negative}
