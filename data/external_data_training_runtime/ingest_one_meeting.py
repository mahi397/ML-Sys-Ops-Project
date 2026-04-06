#!/usr/bin/env python3
"""
Parse one AMI meeting from staged raw XML files, write selected local processed
artifacts, upload those durable artifacts to object storage, load normalized
rows into PostgreSQL, then delete the uploaded local artifacts.

Expected raw layout (already staged by run_ingest.py):
  <raw_root>/
    corpusResources/meetings.xml
    corpusResources/participants.xml
    ontologies/default-topics.xml
    words/<MEETING>.<SPEAKER>.words.xml
    segments/<MEETING>.<SPEAKER>.segments.xml
    topics/<MEETING>.topic.xml
    abstractive/<MEETING>.abssumm.xml

Temporary processed artifacts:
  <processed_root>/
    artifacts/parsed_transcript/<MEETING>.json
    artifacts/summaries/<MEETING>.json
    manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


NITE_ID = "{http://nite.sourceforge.net/}id"
WORD_REF_RE = re.compile(r"#id\((?P<start>[^)]+)\)(?:\.\.id\((?P<end>[^)]+)\))?")


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Token:
    token_id: str
    kind: str
    text: str
    start: float
    end: float
    punc: bool = False
    extra_type: str | None = None


# -----------------------------
# Logging
# -----------------------------
def setup_logger(meeting_id: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(f"ingest.{meeting_id}")
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

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meeting", required=True, help="AMI meeting id, e.g. ES2002a")
    parser.add_argument("--raw-root", type=Path, required=True, help="Root of staged raw input files")
    parser.add_argument("--processed-root", type=Path, required=True, help="Root of processed output files for this meeting")
    parser.add_argument("--pg-container", default="postgres", help="Docker container name for PostgreSQL")
    parser.add_argument("--db-user", default="proj07_user", help="PostgreSQL user")
    parser.add_argument("--db-name", default="proj07_sql_db", help="PostgreSQL database")
    parser.add_argument("--rclone-remote", default="rclone_s3", help="rclone remote name")
    parser.add_argument("--bucket", default="objstore-proj07", help="Object storage bucket/container name")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional log file path")
    parser.add_argument(
        "--no-cleanup-local-artifacts",
        action="store_true",
        help="Keep local uploaded artifact files instead of deleting them",
    )
    return parser.parse_args()


# -----------------------------
# Files / XML helpers
# -----------------------------
def parse_xml(path: Path) -> ET.Element:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return ET.parse(path).getroot()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def load_topic_names(path: Path) -> dict[str, str]:
    root = parse_xml(path)
    mapping: dict[str, str] = {}
    for elem in root.iter():
        topic_id = elem.attrib.get(NITE_ID)
        topic_name = elem.attrib.get("name")
        if topic_id and topic_name:
            mapping[topic_id] = topic_name
    return mapping


def extract_word_ids_from_href(href: str, token_order: list[str]) -> list[str]:
    match = WORD_REF_RE.search(href)
    if not match:
        raise ValueError(f"Could not parse href: {href}")
    start_id = match.group("start")
    end_id = match.group("end") or start_id
    start_idx = token_order.index(start_id)
    end_idx = token_order.index(end_id)
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    return token_order[start_idx:end_idx + 1]


# -----------------------------
# Token / utterance parsing
# -----------------------------
def load_tokens(
    raw_root: Path,
    meeting_id: str,
    speaker_label: str,
    logger: logging.Logger,
) -> tuple[list[Token], dict[str, Token], list[str]]:
    path = raw_root / "words" / f"{meeting_id}.{speaker_label}.words.xml"
    root = parse_xml(path)
    tokens: list[Token] = []
    by_id: dict[str, Token] = {}
    order: list[str] = []

    for child in root:
        token_id = child.attrib.get(NITE_ID)
        if not token_id:
            continue

        start_s = child.attrib.get("starttime")
        end_s = child.attrib.get("endtime")

        if start_s is None and end_s is None:
            logger.warning(
                "Token %s in %s missing both starttime and endtime; defaulting to 0.0",
                token_id,
                path.name,
            )
            start = 0.0
            end = 0.0
        elif start_s is None:
            end = float(end_s)
            start = end
            logger.warning(
                "Token %s in %s missing starttime; using endtime=%s",
                token_id,
                path.name,
                end_s,
            )
        elif end_s is None:
            start = float(start_s)
            end = start
            logger.warning(
                "Token %s in %s missing endtime; using starttime=%s",
                token_id,
                path.name,
                start_s,
            )
        else:
            start = float(start_s)
            end = float(end_s)

        if end < start:
            logger.warning(
                "Token %s in %s has end < start (%.3f < %.3f); normalizing",
                token_id,
                path.name,
                end,
                start,
            )
            start, end = min(start, end), max(start, end)

        kind = child.tag.split("}", 1)[-1]
        text = (child.text or "").strip()

        token = Token(
            token_id=token_id,
            kind=kind,
            text=text,
            start=start,
            end=end,
            punc=child.attrib.get("punc") == "true",
            extra_type=child.attrib.get("type"),
        )
        tokens.append(token)
        by_id[token_id] = token
        order.append(token_id)

    return tokens, by_id, order


def token_to_text_piece(token: Token) -> str:
    if token.kind == "w":
        return token.text
    if token.kind == "vocalsound":
        return f"[{token.extra_type or 'sound'}]"
    if token.kind == "gap":
        return "[gap]"
    if token.kind == "disfmarker":
        return ""
    return token.text


def join_token_text(tokens: Iterable[Token]) -> str:
    parts: list[str] = []
    for token in tokens:
        piece = token_to_text_piece(token)
        if not piece:
            continue

        if token.kind == "w" and token.punc:
            if parts:
                parts[-1] = parts[-1] + piece
            else:
                parts.append(piece)
            continue

        parts.append(piece)

    text = " ".join(parts)
    return re.sub(r"\s+", " ", text).strip()


def make_clean_text(raw_text: str) -> str:
    lowered = raw_text.lower()
    lowered = re.sub(r"\[[^\]]+\]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


# -----------------------------
# AMI metadata parsing
# -----------------------------
def load_meeting_metadata(raw_root: Path, meeting_id: str) -> tuple[dict, list[dict]]:
    meetings_root = parse_xml(raw_root / "corpusResources" / "meetings.xml")

    meeting_elem = None
    for meeting in meetings_root.findall("meeting"):
        if meeting.attrib.get("observation") == meeting_id:
            meeting_elem = meeting
            break

    if meeting_elem is None:
        raise ValueError(f"Meeting {meeting_id} not found in meetings.xml")

    meeting_row = {
        "meeting_id": meeting_id,
        "source_type": "ami",
        "source_name": meeting_elem.attrib.get("name", "AMI meeting"),
        "started_at": None,
        "ended_at": None,
        "raw_folder_prefix": "ami_public_manual_1.6.2/",
    }

    speaker_rows = []
    for speaker in meeting_elem.findall("speaker"):
        speaker_rows.append(
            {
                "meeting_id": meeting_id,
                "user_id": None,
                "speaker_label": speaker.attrib["nxt_agent"],
                "display_name": speaker.attrib["global_name"],
                "role": speaker.attrib.get("role"),
            }
        )

    return meeting_row, speaker_rows


def load_utterances(
    raw_root: Path,
    meeting_id: str,
    speaker_rows: list[dict],
    logger: logging.Logger,
) -> list[dict]:
    speaker_labels = sorted(row["speaker_label"] for row in speaker_rows)
    utterances: list[dict] = []

    for speaker_label in speaker_labels:
        _, token_by_id, token_order = load_tokens(raw_root, meeting_id, speaker_label, logger)
        seg_root = parse_xml(raw_root / "segments" / f"{meeting_id}.{speaker_label}.segments.xml")

        for seg in seg_root.findall("segment"):
            seg_id = seg.attrib[NITE_ID]
            word_ids: list[str] = []

            for child in seg.findall("{http://nite.sourceforge.net/}child"):
                word_ids.extend(extract_word_ids_from_href(child.attrib["href"], token_order))

            segment_tokens = [token_by_id[word_id] for word_id in word_ids]
            raw_text = join_token_text(segment_tokens)

            start_s = seg.attrib.get("transcriber_start")
            end_s = seg.attrib.get("transcriber_end")

            if start_s is None and end_s is None:
                if segment_tokens:
                    seg_start = min(token.start for token in segment_tokens)
                    seg_end = max(token.end for token in segment_tokens)
                    logger.warning(
                        "Segment %s missing both transcriber_start and transcriber_end; inferred from tokens",
                        seg_id,
                    )
                else:
                    seg_start = 0.0
                    seg_end = 0.0
                    logger.warning(
                        "Segment %s missing both transcriber_start and transcriber_end with no tokens; defaulting to 0.0",
                        seg_id,
                    )
            elif start_s is None:
                seg_end = float(end_s)
                seg_start = seg_end
                logger.warning("Segment %s missing transcriber_start; using end value", seg_id)
            elif end_s is None:
                seg_start = float(start_s)
                seg_end = seg_start
                logger.warning("Segment %s missing transcriber_end; using start value", seg_id)
            else:
                seg_start = float(start_s)
                seg_end = float(end_s)

            if seg_end < seg_start:
                logger.warning(
                    "Segment %s has end < start (%.3f < %.3f); normalizing",
                    seg_id,
                    seg_end,
                    seg_start,
                )
                seg_start, seg_end = min(seg_start, seg_end), max(seg_start, seg_end)

            utterances.append(
                {
                    "meeting_id": meeting_id,
                    "speaker_label": speaker_label,
                    "utterance_index": None,
                    "start_time_sec": seg_start,
                    "end_time_sec": seg_end,
                    "raw_text": raw_text,
                    "clean_text": make_clean_text(raw_text),
                    "source_segment_id": seg_id,
                    "word_ids": word_ids,
                }
            )

    utterances.sort(
        key=lambda row: (
            row["start_time_sec"],
            row["end_time_sec"],
            row["speaker_label"],
            row["source_segment_id"],
        )
    )

    for idx, row in enumerate(utterances):
        row["utterance_index"] = idx

    return utterances


def load_topic_segments(
    raw_root: Path,
    meeting_id: str,
    utterances: list[dict],
    logger: logging.Logger,
) -> list[dict]:
    topic_path = raw_root / "topics" / f"{meeting_id}.topic.xml"
    if not topic_path.exists():
        return []

    topic_name_map = load_topic_names(raw_root / "ontologies" / "default-topics.xml")

    token_orders: dict[str, list[str]] = {}
    for speaker_label in {row["speaker_label"] for row in utterances}:
        _, _, order = load_tokens(raw_root, meeting_id, speaker_label, logger)
        token_orders[speaker_label] = order

    utterance_by_index = {row["utterance_index"]: row for row in utterances}
    utterance_index_by_word_id: dict[str, int] = {}
    for row in utterances:
        for word_id in row["word_ids"]:
            utterance_index_by_word_id[word_id] = row["utterance_index"]

    topic_root = parse_xml(topic_path)
    topic_rows: list[dict] = []

    for segment_index, topic in enumerate(topic_root.findall("topic"), start=1):
        word_ids: list[str] = []

        for child in topic.findall("{http://nite.sourceforge.net/}child"):
            href = child.attrib["href"]
            speaker_label = href.split(".", 2)[1]
            word_ids.extend(extract_word_ids_from_href(href, token_orders[speaker_label]))

        utterance_indices = sorted({
            utterance_index_by_word_id[word_id]
            for word_id in word_ids
            if word_id in utterance_index_by_word_id
        })
        if not utterance_indices:
            continue

        pointer = topic.find("{http://nite.sourceforge.net/}pointer")
        topic_pointer_id = None
        if pointer is not None:
            pointer_href = pointer.attrib.get("href", "")
            pointer_match = re.search(r"#id\(([^)]+)\)", pointer_href)
            if pointer_match:
                topic_pointer_id = pointer_match.group(1)

        topic_label = topic.attrib.get("other_description") or topic_name_map.get(
            topic_pointer_id or "", topic_pointer_id or "unknown"
        )

        start_idx = utterance_indices[0]
        end_idx = utterance_indices[-1]
        start_utterance = utterance_by_index[start_idx]
        end_utterance = utterance_by_index[end_idx]

        topic_rows.append(
            {
                "meeting_id": meeting_id,
                "segment_type": "gold",
                "segment_index": segment_index,
                "start_utterance_index": start_idx,
                "end_utterance_index": end_idx,
                "start_time_sec": start_utterance["start_time_sec"],
                "end_time_sec": end_utterance["end_time_sec"],
                "topic_label": topic_label,
            }
        )

    return topic_rows


def build_transition_rows(meeting_id: str, utterances: list[dict], topic_segments: list[dict]) -> list[dict]:
    topic_by_utterance_index: dict[int, int] = {}

    for topic in topic_segments:
        for idx in range(topic["start_utterance_index"], topic["end_utterance_index"] + 1):
            topic_by_utterance_index[idx] = topic["segment_index"]

    transitions: list[dict] = []
    for idx in range(len(utterances) - 1):
        transitions.append(
            {
                "meeting_id": meeting_id,
                "left_utterance_index": idx,
                "right_utterance_index": idx + 1,
                "transition_index": idx,
                "gold_boundary_label": (
                    topic_by_utterance_index.get(idx) != topic_by_utterance_index.get(idx + 1)
                ),
                "pred_boundary_prob": None,
                "pred_boundary_label": None,
            }
        )

    return transitions


def parse_summary_sections(raw_root: Path, meeting_id: str) -> dict:
    summary_path = raw_root / "abstractive" / f"{meeting_id}.abssumm.xml"
    if not summary_path.exists():
        return {"meeting_id": meeting_id, "sections": {}, "full_summary": ""}

    summary_root = parse_xml(summary_path)
    sections: dict[str, list[str]] = {}

    for child in summary_root:
        section_name = child.tag.split("}", 1)[-1]
        sections[section_name] = [
            (sentence.text or "").strip()
            for sentence in child.findall("sentence")
            if (sentence.text or "").strip()
        ]

    return {
        "meeting_id": meeting_id,
        "sections": sections,
        "full_summary": " ".join(sections.get("abstract", [])),
    }


# -----------------------------
# Object storage upload helpers
# -----------------------------
def upload_file_to_object_storage(
    local_path: Path,
    remote: str,
    bucket: str,
    object_key: str,
    logger: logging.Logger,
) -> None:
    dest = f"{remote}:{bucket}/{object_key}"
    logger.info("Uploading file to object storage: %s -> %s", local_path, dest)

    start = time.time()
    cmd = ["rclone", "copyto", str(local_path), dest, "-P"]
    result = subprocess.run(cmd, text=True, capture_output=True)

    if result.returncode != 0:
        logger.error("Upload failed for %s", local_path)
        logger.error("rclone stdout:\n%s", result.stdout)
        logger.error("rclone stderr:\n%s", result.stderr)
        raise RuntimeError(f"Upload failed for {local_path} -> {dest}")

    logger.info("Upload complete in %.2fs: %s", time.time() - start, dest)


# -----------------------------
# PostgreSQL helpers
# -----------------------------
def sql_literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def run_psql_script(
    pg_container: str,
    db_user: str,
    db_name: str,
    sql: str,
    logger: logging.Logger,
    label: str,
) -> None:
    logger.info("Running Postgres step: %s", label)
    start = time.time()

    cmd = [
        "docker", "exec", "-i", pg_container,
        "psql",
        "-U", db_user,
        "-d", db_name,
        "-v", "ON_ERROR_STOP=1",
        "-f", "-",
    ]
    result = subprocess.run(cmd, input=sql, text=True, capture_output=True)
    if result.returncode != 0:
        logger.error("Postgres step failed: %s", label)
        logger.error("psql stdout:\n%s", result.stdout)
        logger.error("psql stderr:\n%s", result.stderr)
        raise RuntimeError(f"psql script failed for step: {label}")

    logger.info("Completed Postgres step in %.2fs: %s", time.time() - start, label)


def run_psql_query(
    pg_container: str,
    db_user: str,
    db_name: str,
    sql: str,
    logger: logging.Logger,
    label: str,
) -> str:
    logger.info("Running Postgres query: %s", label)

    cmd = [
        "docker", "exec", pg_container,
        "psql",
        "-U", db_user,
        "-d", db_name,
        "-At",
        "-F", "\t",
        "-c", sql,
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        logger.error("Postgres query failed: %s", label)
        logger.error("psql stdout:\n%s", result.stdout)
        logger.error("psql stderr:\n%s", result.stderr)
        raise RuntimeError(f"psql query failed for step: {label}")

    return result.stdout


def load_rows_to_postgres(
    meeting_row: dict,
    speaker_rows: list[dict],
    utterances: list[dict],
    topic_segments: list[dict],
    transitions: list[dict],
    parsed_transcript_object_key: str,
    summary_object_key: str,
    pg_container: str,
    db_user: str,
    db_name: str,
    logger: logging.Logger,
) -> dict:
    meeting_id = meeting_row["meeting_id"]

    logger.info("Deleting existing rows for meeting=%s if present", meeting_id)
    delete_sql = f"""
    BEGIN;
    DELETE FROM meetings WHERE meeting_id = {sql_literal(meeting_id)};
    COMMIT;
    """
    run_psql_script(pg_container, db_user, db_name, delete_sql, logger, "delete existing meeting rows")

    logger.info("Inserting meeting row")
    meeting_sql = f"""
    BEGIN;
    INSERT INTO meetings (
        meeting_id, source_type, source_name, started_at, ended_at, raw_folder_prefix
    ) VALUES (
        {sql_literal(meeting_row["meeting_id"])},
        {sql_literal(meeting_row["source_type"])},
        {sql_literal(meeting_row["source_name"])},
        {sql_literal(meeting_row["started_at"])},
        {sql_literal(meeting_row["ended_at"])},
        {sql_literal(meeting_row["raw_folder_prefix"])}
    );
    COMMIT;
    """
    run_psql_script(pg_container, db_user, db_name, meeting_sql, logger, "insert meeting")

    if speaker_rows:
        logger.info("Inserting %d speaker rows", len(speaker_rows))
        values = []
        for row in speaker_rows:
            values.append(
                "("
                f"{sql_literal(row['meeting_id'])}, "
                f"{sql_literal(row['user_id'])}, "
                f"{sql_literal(row['speaker_label'])}, "
                f"{sql_literal(row['display_name'])}, "
                f"{sql_literal(row['role'])}"
                ")"
            )
        speaker_sql = f"""
        BEGIN;
        INSERT INTO meeting_speakers (
            meeting_id, user_id, speaker_label, display_name, role
        ) VALUES
        {",\n".join(values)};
        COMMIT;
        """
        run_psql_script(pg_container, db_user, db_name, speaker_sql, logger, "insert meeting_speakers")

    speaker_map_sql = f"""
    SELECT speaker_label, meeting_speaker_id
    FROM meeting_speakers
    WHERE meeting_id = {sql_literal(meeting_id)}
    ORDER BY meeting_speaker_id;
    """
    speaker_map_raw = run_psql_query(pg_container, db_user, db_name, speaker_map_sql, logger, "resolve meeting_speaker_ids")
    meeting_speaker_id_by_label: dict[str, int] = {}
    for line in speaker_map_raw.splitlines():
        if not line.strip():
            continue
        speaker_label, meeting_speaker_id = line.split("\t")
        meeting_speaker_id_by_label[speaker_label] = int(meeting_speaker_id)

    if utterances:
        logger.info("Inserting %d utterance rows", len(utterances))
        values = []
        for row in utterances:
            values.append(
                "("
                f"{sql_literal(row['meeting_id'])}, "
                f"{sql_literal(meeting_speaker_id_by_label[row['speaker_label']])}, "
                f"{sql_literal(row['utterance_index'])}, "
                f"{sql_literal(row['start_time_sec'])}, "
                f"{sql_literal(row['end_time_sec'])}, "
                f"{sql_literal(row['raw_text'])}, "
                f"{sql_literal(row['clean_text'])}, "
                f"{sql_literal(row['source_segment_id'])}"
                ")"
            )

        utterance_sql = f"""
        BEGIN;
        INSERT INTO utterances (
            meeting_id, meeting_speaker_id, utterance_index,
            start_time_sec, end_time_sec, raw_text, clean_text, source_segment_id
        ) VALUES
        {",\n".join(values)};
        COMMIT;
        """
        run_psql_script(pg_container, db_user, db_name, utterance_sql, logger, "insert utterances")

    utterance_map_sql = f"""
    SELECT utterance_index, utterance_id
    FROM utterances
    WHERE meeting_id = {sql_literal(meeting_id)}
    ORDER BY utterance_index;
    """
    utterance_map_raw = run_psql_query(pg_container, db_user, db_name, utterance_map_sql, logger, "resolve utterance_ids")
    utterance_id_by_index: dict[int, int] = {}
    for line in utterance_map_raw.splitlines():
        if not line.strip():
            continue
        utterance_index, utterance_id = line.split("\t")
        utterance_id_by_index[int(utterance_index)] = int(utterance_id)

    if topic_segments:
        logger.info("Inserting %d topic segment rows", len(topic_segments))
        values = []
        for row in topic_segments:
            values.append(
                "("
                f"{sql_literal(row['meeting_id'])}, "
                f"{sql_literal(row['segment_type'])}, "
                f"{sql_literal(row['segment_index'])}, "
                f"{sql_literal(utterance_id_by_index[row['start_utterance_index']])}, "
                f"{sql_literal(utterance_id_by_index[row['end_utterance_index']])}, "
                f"{sql_literal(row['start_time_sec'])}, "
                f"{sql_literal(row['end_time_sec'])}, "
                f"{sql_literal(row['topic_label'])}"
                ")"
            )

        topic_sql = f"""
        BEGIN;
        INSERT INTO topic_segments (
            meeting_id, segment_type, segment_index,
            start_utterance_id, end_utterance_id,
            start_time_sec, end_time_sec, topic_label
        ) VALUES
        {",\n".join(values)};
        COMMIT;
        """
        run_psql_script(pg_container, db_user, db_name, topic_sql, logger, "insert topic_segments")

    if transitions:
        logger.info("Inserting %d transition rows", len(transitions))
        values = []
        for row in transitions:
            values.append(
                "("
                f"{sql_literal(row['meeting_id'])}, "
                f"{sql_literal(utterance_id_by_index[row['left_utterance_index']])}, "
                f"{sql_literal(utterance_id_by_index[row['right_utterance_index']])}, "
                f"{sql_literal(row['transition_index'])}, "
                f"{sql_literal(row['gold_boundary_label'])}, "
                f"{sql_literal(row['pred_boundary_prob'])}, "
                f"{sql_literal(row['pred_boundary_label'])}"
                ")"
            )

        transition_sql = f"""
        BEGIN;
        INSERT INTO utterance_transitions (
            meeting_id, left_utterance_id, right_utterance_id,
            transition_index, gold_boundary_label, pred_boundary_prob, pred_boundary_label
        ) VALUES
        {",\n".join(values)};
        COMMIT;
        """
        run_psql_script(pg_container, db_user, db_name, transition_sql, logger, "insert utterance_transitions")

    artifact_rows = [
        {
            "meeting_id": meeting_id,
            "artifact_type": "parsed_transcript",
            "object_key": parsed_transcript_object_key,
            "content_type": "application/json",
            "artifact_version": 1,
        },
        {
            "meeting_id": meeting_id,
            "artifact_type": "summary_json",
            "object_key": summary_object_key,
            "content_type": "application/json",
            "artifact_version": 1,
        },
    ]

    logger.info("Inserting %d meeting_artifact rows", len(artifact_rows))
    values = []
    for row in artifact_rows:
        values.append(
            "("
            f"{sql_literal(row['meeting_id'])}, "
            f"{sql_literal(row['artifact_type'])}, "
            f"{sql_literal(row['object_key'])}, "
            f"{sql_literal(row['content_type'])}, "
            f"{sql_literal(row['artifact_version'])}"
            ")"
        )

    artifact_sql = f"""
    BEGIN;
    INSERT INTO meeting_artifacts (
        meeting_id, artifact_type, object_key, content_type, artifact_version
    ) VALUES
    {",\n".join(values)};
    COMMIT;
    """
    run_psql_script(pg_container, db_user, db_name, artifact_sql, logger, "insert meeting_artifacts")

    summary_rows = [
        {
            "meeting_id": meeting_id,
            "summary_type": "ami_gold",
            "summary_object_key": summary_object_key,
            "created_by_user_id": None,
            "version": 1,
        }
    ]

    logger.info("Inserting %d summary rows", len(summary_rows))
    values = []
    for row in summary_rows:
        values.append(
            "("
            f"{sql_literal(row['meeting_id'])}, "
            f"{sql_literal(row['summary_type'])}, "
            f"{sql_literal(row['summary_object_key'])}, "
            f"{sql_literal(row['created_by_user_id'])}, "
            f"{sql_literal(row['version'])}"
            ")"
        )

    summary_sql = f"""
    BEGIN;
    INSERT INTO summaries (
        meeting_id, summary_type, summary_object_key, created_by_user_id, version
    ) VALUES
    {",\n".join(values)};
    COMMIT;
    """
    run_psql_script(pg_container, db_user, db_name, summary_sql, logger, "insert summaries")

    return {
        "meeting_artifacts": artifact_rows,
        "summaries": summary_rows,
    }


def delete_uploaded_local_files(
    parsed_transcript_local: Path,
    summary_local: Path,
    manifest_local: Path,
    logger: logging.Logger,
) -> None:
    logger.info("Deleting uploaded local artifact files from block storage")
    parsed_transcript_local.unlink(missing_ok=True)
    summary_local.unlink(missing_ok=True)
    manifest_local.unlink(missing_ok=True)

    for parent in [
        parsed_transcript_local.parent,
        summary_local.parent,
        parsed_transcript_local.parent.parent,
    ]:
        try:
            parent.rmdir()
        except OSError:
            pass

    logger.info("Local artifact cleanup complete")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    default_log_file = args.log_file or (Path("/mnt/block/ingest_logs") / f"{args.meeting}.log")
    logger = setup_logger(args.meeting, default_log_file)

    try:
        meeting_id = args.meeting
        raw_root = args.raw_root.resolve()
        processed_root = args.processed_root.resolve()
        processed_root.mkdir(parents=True, exist_ok=True)

        logger.info("Starting ingestion for meeting=%s", meeting_id)
        logger.info("Raw root=%s", raw_root)
        logger.info("Processed root=%s", processed_root)
        logger.info("Postgres container=%s db=%s user=%s", args.pg_container, args.db_name, args.db_user)
        logger.info("Object storage target=%s:%s", args.rclone_remote, args.bucket)

        parsed_transcript_object_key = f"processed/ami/v1/transcripts/{meeting_id}.json"
        summary_object_key = f"processed/ami/v1/summaries/{meeting_id}.json"
        manifest_object_key = f"processed/ami/v1/manifests/{meeting_id}.json"

        logger.info("Parsing meeting metadata")
        meeting_row, speaker_rows = load_meeting_metadata(raw_root, meeting_id)
        logger.info("Parsed meeting metadata: meeting_id=%s speakers=%d", meeting_id, len(speaker_rows))

        logger.info("Parsing utterances")
        utterances = load_utterances(raw_root, meeting_id, speaker_rows, logger)
        logger.info("Parsed utterances: count=%d", len(utterances))

        logger.info("Parsing topic segments")
        topic_segments = load_topic_segments(raw_root, meeting_id, utterances, logger)
        logger.info("Parsed topic segments: count=%d", len(topic_segments))

        logger.info("Building transition rows")
        transitions = build_transition_rows(meeting_id, utterances, topic_segments)
        logger.info("Built transition rows: count=%d", len(transitions))

        logger.info("Parsing abstractive summary")
        summary_payload = parse_summary_sections(raw_root, meeting_id)
        logger.info(
            "Parsed summary sections: sections=%s full_summary_chars=%d",
            list(summary_payload.get("sections", {}).keys()),
            len(summary_payload.get("full_summary", "")),
        )

        transcript_view_rows = [
            {
                "utterance_index": row["utterance_index"],
                "speaker_label": row["speaker_label"],
                "start_time_sec": row["start_time_sec"],
                "end_time_sec": row["end_time_sec"],
                "raw_text": row["raw_text"],
                "source_segment_id": row["source_segment_id"],
            }
            for row in utterances
        ]
        parsed_transcript_payload = {
            "meeting_id": meeting_id,
            "utterance_unit": "segment",
            "utterance_count": len(transcript_view_rows),
            "utterances": transcript_view_rows,
        }

        parsed_transcript_local = processed_root / "artifacts" / "parsed_transcript" / f"{meeting_id}.json"
        summary_local = processed_root / "artifacts" / "summaries" / f"{meeting_id}.json"
        manifest_local = processed_root / "manifest.json"

        logger.info("Writing temporary local processed artifacts")
        write_json(parsed_transcript_local, parsed_transcript_payload)
        write_json(summary_local, summary_payload)

        manifest = {
            "meeting_id": meeting_id,
            "counts": {
                "meetings": 1,
                "meeting_speakers": len(speaker_rows),
                "meeting_artifacts": 2,
                "utterances": len(utterances),
                "utterance_transitions": len(transitions),
                "topic_segments": len(topic_segments),
                "summaries": 1,
            },
            "artifact_keys": {
                "parsed_transcript_object_key": parsed_transcript_object_key,
                "summary_object_key": summary_object_key,
                "manifest_object_key": manifest_object_key,
            },
            "first_rows": {
                "meetings": meeting_row,
                "meeting_speakers": speaker_rows[0] if speaker_rows else None,
                "utterances": {
                    "meeting_id": utterances[0]["meeting_id"],
                    "speaker_label": utterances[0]["speaker_label"],
                    "utterance_index": utterances[0]["utterance_index"],
                    "start_time_sec": utterances[0]["start_time_sec"],
                    "end_time_sec": utterances[0]["end_time_sec"],
                    "raw_text": utterances[0]["raw_text"],
                    "source_segment_id": utterances[0]["source_segment_id"],
                } if utterances else None,
                "utterance_transitions": transitions[0] if transitions else None,
                "topic_segments": topic_segments[0] if topic_segments else None,
            },
        }
        write_json(manifest_local, manifest)

        logger.info("Uploading durable artifacts to object storage")
        upload_file_to_object_storage(
            parsed_transcript_local,
            remote=args.rclone_remote,
            bucket=args.bucket,
            object_key=parsed_transcript_object_key,
            logger=logger,
        )
        upload_file_to_object_storage(
            summary_local,
            remote=args.rclone_remote,
            bucket=args.bucket,
            object_key=summary_object_key,
            logger=logger,
        )
        upload_file_to_object_storage(
            manifest_local,
            remote=args.rclone_remote,
            bucket=args.bucket,
            object_key=manifest_object_key,
            logger=logger,
        )

        logger.info("Loading normalized rows into PostgreSQL")
        load_rows_to_postgres(
            meeting_row=meeting_row,
            speaker_rows=speaker_rows,
            utterances=utterances,
            topic_segments=topic_segments,
            transitions=transitions,
            parsed_transcript_object_key=parsed_transcript_object_key,
            summary_object_key=summary_object_key,
            pg_container=args.pg_container,
            db_user=args.db_user,
            db_name=args.db_name,
            logger=logger,
        )

        if not args.no_cleanup_local_artifacts:
            delete_uploaded_local_files(
                parsed_transcript_local=parsed_transcript_local,
                summary_local=summary_local,
                manifest_local=manifest_local,
                logger=logger,
            )

        logger.info("Successfully finished ingestion for meeting=%s", meeting_id)
        logger.info("Uploaded durable artifacts:")
        logger.info("  %s", parsed_transcript_object_key)
        logger.info("  %s", summary_object_key)
        logger.info("  %s", manifest_object_key)

    except Exception:
        logger.exception("Ingestion failed for meeting=%s", args.meeting)
        raise


if __name__ == "__main__":
    main()
