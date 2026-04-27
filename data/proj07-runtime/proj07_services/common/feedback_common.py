from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json


DATASET_VERSION_OBJECT_KEY_RE = re.compile(r"(?:^|/)v(?P<version>\d+)(?:/|$)")


def env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def object_bucket() -> str:
    bucket = os.getenv("OBJECT_BUCKET", "").strip() or os.getenv("BUCKET", "").strip()
    if not bucket:
        raise RuntimeError("Missing required environment variable: OBJECT_BUCKET or BUCKET")
    return bucket


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
    return f"{env('RCLONE_REMOTE')}:{object_bucket()}/{object_key}"


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


def download_dir(object_prefix: str, local_dir: Path, logger: logging.Logger) -> None:
    ensure_dir(local_dir)
    cmd = ["rclone", "copy", build_object_uri(object_prefix), str(local_dir), "-P"]
    run_command(cmd, logger, f"download dir {object_prefix}")


def parse_dataset_version_from_manifest(manifest_json: dict[str, Any] | None) -> int | None:
    if not isinstance(manifest_json, dict):
        return None

    ongoing = manifest_json.get("ongoing_version")
    if isinstance(ongoing, dict):
        for key in ("snapshot_version", "feedback_pool_version", "version"):
            value = ongoing.get(key)
            if value not in (None, ""):
                return int(value)

    for key in ("dataset_version", "snapshot_version", "feedback_pool_version", "version"):
        value = manifest_json.get(key)
        if value not in (None, ""):
            return int(value)
    return None


def parse_dataset_version_from_object_key(object_key: str) -> int | None:
    normalized = str(object_key or "").strip().rstrip("/")
    if not normalized:
        return None
    match = DATASET_VERSION_OBJECT_KEY_RE.search(normalized)
    if not match:
        return None
    return int(match.group("version"))


def parse_dataset_version_value(object_key: str, manifest_json: dict[str, Any] | None = None) -> int | None:
    version = parse_dataset_version_from_object_key(object_key)
    if version is not None:
        return version
    return parse_dataset_version_from_manifest(manifest_json)


def dataset_object_prefix_from_key(object_key: str) -> str:
    normalized = str(object_key or "").strip().rstrip("/")
    if not normalized:
        raise RuntimeError("Dataset object_key cannot be empty")
    tail = normalized.rsplit("/", 1)[-1]
    if "." in tail:
        return normalized.rsplit("/", 1)[0]
    return normalized


def list_dataset_version_records(
    conn,
    *,
    dataset_name: str,
    stage: str | None = None,
) -> list[dict[str, Any]]:
    query = """
        SELECT dataset_version_id, dataset_name, stage, source_type, object_key, manifest_json, created_at
        FROM dataset_versions
        WHERE dataset_name = %s
    """
    params: list[Any] = [dataset_name]
    if stage is not None:
        query += " AND stage = %s"
        params.append(stage)
    query += " ORDER BY dataset_version_id ASC"

    with conn.cursor() as cur:
        cur.execute(query, tuple(params))
        rows = cur.fetchall()

    records: list[dict[str, Any]] = []
    for row in rows:
        version = parse_dataset_version_value(
            str(row["object_key"] or ""),
            row.get("manifest_json"),
        )
        if version is None:
            raise RuntimeError(
                "dataset_versions row is missing a parseable version "
                f"(dataset_version_id={row['dataset_version_id']} object_key={row['object_key']!r})"
            )
        records.append({**row, "version": version})

    records.sort(key=lambda row: (int(row["version"]), int(row["dataset_version_id"])))
    return records


def latest_dataset_version_record(
    conn,
    *,
    dataset_name: str,
    stage: str | None = None,
) -> dict[str, Any] | None:
    records = list_dataset_version_records(conn, dataset_name=dataset_name, stage=stage)
    return records[-1] if records else None


def next_dataset_version_number(
    conn,
    *,
    dataset_name: str,
    stage: str | None = None,
) -> int:
    latest_record = latest_dataset_version_record(conn, dataset_name=dataset_name, stage=stage)
    if latest_record is None:
        return 1
    return int(latest_record["version"]) + 1


def upsert_meeting_artifact(
    conn,
    meeting_id: str,
    artifact_type: str,
    object_key: str,
    content_type: str,
    artifact_version: int,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO meeting_artifacts (
                meeting_id, artifact_type, object_key, content_type, artifact_version
            )
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (meeting_id, artifact_type, artifact_version)
            DO UPDATE
            SET object_key = EXCLUDED.object_key,
                content_type = EXCLUDED.content_type,
                created_at = NOW()
            """,
            (meeting_id, artifact_type, object_key, content_type, artifact_version),
        )


def stable_split_70_15_15(meeting_ids: list[str]) -> dict[str, list[str]]:
    """
    Deterministically slice sorted meeting IDs into approximate 70/15/15
    train/val/test partitions for the current batch.

    This uses batch-relative percentage cutoffs rather than hashing each
    meeting independently, so small retraining batches are more likely to
    contribute examples to validation and test.
    """
    ordered_meeting_ids = sorted(set(meeting_ids))
    total = len(ordered_meeting_ids)
    if total == 0:
        return {"train": [], "val": [], "test": []}

    train_cutoff = max(1, (total * 70) // 100)
    val_cutoff = max(train_cutoff, (total * 85) // 100)

    train = ordered_meeting_ids[:train_cutoff]
    val = ordered_meeting_ids[train_cutoff:val_cutoff]
    test = ordered_meeting_ids[val_cutoff:]
    return {"train": train, "val": val, "test": test}


def ensure_dataset_version_record(
    conn,
    dataset_name: str,
    stage: str,
    source_type: str,
    object_key: str,
    manifest_json: dict[str, Any],
) -> None:
    target_version = parse_dataset_version_value(object_key, manifest_json)
    if target_version is None:
        raise RuntimeError(
            "Cannot upsert dataset_versions row without a parseable version in object_key or manifest_json"
        )

    matching_row_id: int | None = None
    for row in list_dataset_version_records(conn, dataset_name=dataset_name, stage=stage):
        if int(row["version"]) == target_version:
            matching_row_id = int(row["dataset_version_id"])

    with conn.cursor() as cur:
        if matching_row_id is not None:
            cur.execute(
                """
                UPDATE dataset_versions
                SET source_type = %s,
                    object_key = %s,
                    manifest_json = %s
                WHERE dataset_version_id = %s
                """,
                (source_type, object_key, Json(manifest_json), matching_row_id),
            )
            conn.commit()
            return

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


def insert_dataset_version(
    conn,
    dataset_name: str,
    stage: str,
    source_type: str,
    object_key: str,
    manifest_json: dict[str, Any],
) -> None:
    ensure_dataset_version_record(
        conn=conn,
        dataset_name=dataset_name,
        stage=stage,
        source_type=source_type,
        object_key=object_key,
        manifest_json=manifest_json,
    )


def insert_dataset_quality_report(
    conn,
    *,
    dataset_name: str,
    report_scope: str,
    report_status: str,
    details_json: dict[str, Any],
    dataset_version: str | None = None,
    reference_dataset_name: str | None = None,
    reference_dataset_version: str | None = None,
    report_path: str | None = None,
    share_drifted_features: float | None = None,
    drifted_feature_count: int | None = None,
    total_feature_count: int | None = None,
    window_started_at: str | None = None,
    window_ended_at: str | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO dataset_quality_reports (
                dataset_name,
                report_scope,
                report_status,
                dataset_version,
                reference_dataset_name,
                reference_dataset_version,
                report_path,
                share_drifted_features,
                drifted_feature_count,
                total_feature_count,
                window_started_at,
                window_ended_at,
                details_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                dataset_name,
                report_scope,
                report_status,
                dataset_version,
                reference_dataset_name,
                reference_dataset_version,
                report_path,
                share_drifted_features,
                drifted_feature_count,
                total_feature_count,
                window_started_at,
                window_ended_at,
                Json(details_json),
            ),
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
