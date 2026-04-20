#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row


APP_NAME = "ingest_ami_meeting"
NITE_ID = "{http://nite.sourceforge.net/}id"
WORD_REF_RE = re.compile(r"#id\((?P<start>[^)]+)\)(?:\.\.id\((?P<end>[^)]+)\))?")


@dataclass
class Token:
    token_id: str
    kind: str
    text: str
    start: float
    end: float
    punc: bool = False
    extra_type: str | None = None


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_conn():
    return psycopg.connect(require_env("DATABASE_URL"), row_factory=dict_row)


def setup_logger(log_file: Path | None = None, *, name: str = APP_NAME) -> logging.Logger:
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

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meeting", required=True, help="AMI meeting id, e.g. ES2002a")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument(
        "--raw-folder-prefix",
        default=os.getenv("AMI_OBJECT_PREFIX", "ami_public_manual_1.6.2"),
        help="Object-store prefix that contains the raw AMI corpus.",
    )
    parser.add_argument(
        "--processed-object-prefix",
        default="processed/ami/v1",
        help="Object-store prefix for processed AMI artifacts.",
    )
    parser.add_argument("--artifact-version", type=int, default=1)
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--no-cleanup-local-artifacts", action="store_true")
    return parser.parse_args()


def parse_xml(path: Path) -> ET.Element:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return ET.parse(path).getroot()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def upload_file_to_object_storage(
    local_path: Path,
    *,
    remote: str,
    bucket: str,
    object_key: str,
    logger: logging.Logger,
) -> None:
    dest = f"{remote}:{bucket}/{object_key}"
    cmd = ["rclone", "copyto", str(local_path), dest, "-P"]
    logger.info("Uploading %s -> %s", local_path, dest)
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        if result.stdout:
            logger.error("rclone stdout:\n%s", result.stdout)
        if result.stderr:
            logger.error("rclone stderr:\n%s", result.stderr)
        raise RuntimeError(f"Upload failed for {local_path} -> {dest}")


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
            start = 0.0
            end = 0.0
            logger.warning("Token %s in %s missing timestamps; defaulting to 0.0", token_id, path.name)
        elif start_s is None:
            end = float(end_s)
            start = end
            logger.warning("Token %s in %s missing starttime; using endtime", token_id, path.name)
        elif end_s is None:
            start = float(start_s)
            end = start
            logger.warning("Token %s in %s missing endtime; using starttime", token_id, path.name)
        else:
            start = float(start_s)
            end = float(end_s)

        if end < start:
            start, end = min(start, end), max(start, end)

        token = Token(
            token_id=token_id,
            kind=child.tag.split("}", 1)[-1],
            text=(child.text or "").strip(),
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

    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def make_clean_text(raw_text: str) -> str:
    lowered = raw_text.lower()
    lowered = re.sub(r"\[[^\]]+\]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def build_ami_user_id(meeting_id: str, speaker_label: str) -> str:
    normalized_meeting = re.sub(r"[^a-zA-Z0-9]+", "_", meeting_id).strip("_").lower()
    normalized_speaker = re.sub(r"[^a-zA-Z0-9]+", "_", speaker_label).strip("_").lower()
    return f"ami_{normalized_meeting}_{normalized_speaker}"


def load_meeting_metadata(
    raw_root: Path,
    meeting_id: str,
    *,
    raw_folder_prefix: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    meetings_root = parse_xml(raw_root / "corpusResources" / "meetings.xml")

    meeting_elem = None
    for meeting in meetings_root.findall("meeting"):
        if meeting.attrib.get("observation") == meeting_id:
            meeting_elem = meeting
            break

    if meeting_elem is None:
        raise ValueError(f"Meeting {meeting_id} not found in meetings.xml")

    normalized_raw_prefix = raw_folder_prefix.strip("/")
    meeting_row = {
        "meeting_id": meeting_id,
        "source_type": "ami",
        "source_name": meeting_elem.attrib.get("name", "AMI meeting"),
        "started_at": None,
        "ended_at": None,
        "raw_folder_prefix": f"{normalized_raw_prefix}/" if normalized_raw_prefix else None,
        "is_valid": True,
    }

    user_rows: list[dict[str, Any]] = []
    participant_rows: list[dict[str, Any]] = []
    speaker_rows: list[dict[str, Any]] = []

    for speaker in meeting_elem.findall("speaker"):
        speaker_label = speaker.attrib["nxt_agent"]
        display_name = speaker.attrib.get("global_name", speaker_label).strip() or speaker_label
        user_id = build_ami_user_id(meeting_id, speaker_label)

        user_rows.append(
            {
                "user_id": user_id,
                "display_name": display_name,
                "email": None,
            }
        )
        participant_rows.append(
            {
                "meeting_id": meeting_id,
                "user_id": user_id,
                "role": "participant",
                "can_view_summary": True,
                "can_edit_summary": False,
                "joined_at": None,
                "left_at": None,
            }
        )
        speaker_rows.append(
            {
                "meeting_id": meeting_id,
                "user_id": user_id,
                "speaker_label": speaker_label,
                "display_name": display_name,
                "role": speaker.attrib.get("role"),
            }
        )

    return meeting_row, user_rows, participant_rows, speaker_rows


def load_utterances(
    raw_root: Path,
    meeting_id: str,
    speaker_rows: list[dict[str, Any]],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    speaker_labels = sorted(row["speaker_label"] for row in speaker_rows)
    utterances: list[dict[str, Any]] = []

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
                else:
                    seg_start = 0.0
                    seg_end = 0.0
            elif start_s is None:
                seg_end = float(end_s)
                seg_start = seg_end
            elif end_s is None:
                seg_start = float(start_s)
                seg_end = seg_start
            else:
                seg_start = float(start_s)
                seg_end = float(end_s)

            if seg_end < seg_start:
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
    utterances: list[dict[str, Any]],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
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
    topic_rows: list[dict[str, Any]] = []

    for segment_index, topic in enumerate(topic_root.findall("topic"), start=1):
        word_ids: list[str] = []

        for child in topic.findall("{http://nite.sourceforge.net/}child"):
            href = child.attrib["href"]
            speaker_label = href.split(".", 2)[1]
            word_ids.extend(extract_word_ids_from_href(href, token_orders[speaker_label]))

        utterance_indices = sorted(
            {
                utterance_index_by_word_id[word_id]
                for word_id in word_ids
                if word_id in utterance_index_by_word_id
            }
        )
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
            topic_pointer_id or "",
            topic_pointer_id or "unknown",
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


def build_transition_rows(
    meeting_id: str,
    utterances: list[dict[str, Any]],
    topic_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    topic_by_utterance_index: dict[int, int] = {}
    for topic in topic_segments:
        for idx in range(topic["start_utterance_index"], topic["end_utterance_index"] + 1):
            topic_by_utterance_index[idx] = topic["segment_index"]

    transitions: list[dict[str, Any]] = []
    for idx in range(len(utterances) - 1):
        transitions.append(
            {
                "meeting_id": meeting_id,
                "left_utterance_index": idx,
                "right_utterance_index": idx + 1,
                "transition_index": idx,
                "gold_boundary_label": topic_by_utterance_index.get(idx) != topic_by_utterance_index.get(idx + 1),
                "pred_boundary_prob": None,
                "pred_boundary_label": None,
            }
        )

    return transitions


def parse_summary_sections(raw_root: Path, meeting_id: str) -> dict[str, Any]:
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


def get_ami_meeting_ingest_state(conn, meeting_id: str) -> tuple[bool, bool]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                EXISTS (
                    SELECT 1
                    FROM meetings m
                    WHERE m.meeting_id = %s
                      AND m.source_type = 'ami'
                ) AS meeting_exists,
                EXISTS (
                    SELECT 1
                    FROM meetings m
                    WHERE m.meeting_id = %s
                      AND m.source_type = 'ami'
                      AND EXISTS (
                          SELECT 1
                          FROM utterances u
                          WHERE u.meeting_id = m.meeting_id
                      )
                      AND EXISTS (
                          SELECT 1
                          FROM topic_segments ts
                          WHERE ts.meeting_id = m.meeting_id
                            AND ts.segment_type = 'gold'
                      )
                      AND EXISTS (
                          SELECT 1
                          FROM meeting_artifacts ma
                          WHERE ma.meeting_id = m.meeting_id
                            AND ma.artifact_type = 'parsed_transcript'
                      )
                      AND EXISTS (
                          SELECT 1
                          FROM meeting_artifacts ma
                          WHERE ma.meeting_id = m.meeting_id
                            AND ma.artifact_type = 'summary_json'
                      )
                      AND EXISTS (
                          SELECT 1
                          FROM summaries s
                          WHERE s.meeting_id = m.meeting_id
                            AND s.summary_type = 'ami_gold'
                      )
                ) AS fully_ingested
            """,
            (meeting_id, meeting_id),
        )
        row = cur.fetchone()
    return bool(row["meeting_exists"]), bool(row["fully_ingested"])


def count_fully_ingested_meetings(conn, meeting_ids: list[str]) -> int:
    if not meeting_ids:
        return 0

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) AS meeting_count
            FROM meetings m
            WHERE m.meeting_id = ANY(%s)
              AND m.source_type = 'ami'
              AND EXISTS (
                  SELECT 1
                  FROM utterances u
                  WHERE u.meeting_id = m.meeting_id
              )
              AND EXISTS (
                  SELECT 1
                  FROM topic_segments ts
                  WHERE ts.meeting_id = m.meeting_id
                    AND ts.segment_type = 'gold'
              )
              AND EXISTS (
                  SELECT 1
                  FROM meeting_artifacts ma
                  WHERE ma.meeting_id = m.meeting_id
                    AND ma.artifact_type = 'parsed_transcript'
              )
              AND EXISTS (
                  SELECT 1
                  FROM meeting_artifacts ma
                  WHERE ma.meeting_id = m.meeting_id
                    AND ma.artifact_type = 'summary_json'
              )
              AND EXISTS (
                  SELECT 1
                  FROM summaries s
                  WHERE s.meeting_id = m.meeting_id
                    AND s.summary_type = 'ami_gold'
              )
            """,
            (meeting_ids,),
        )
        row = cur.fetchone()
    return int(row["meeting_count"])


def fetch_returning_id(cur, key: str) -> int:
    row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"Expected RETURNING {key} row, but query returned none")
    return int(row[key])


def upsert_users(cur, user_rows: list[dict[str, Any]]) -> None:
    for user in user_rows:
        cur.execute(
            """
            INSERT INTO users (user_id, display_name, email)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id)
            DO UPDATE
            SET display_name = EXCLUDED.display_name
            """,
            (user["user_id"], user["display_name"], user["email"]),
        )


def insert_rows(
    *,
    conn,
    payload: dict[str, Any],
    replace_existing: bool,
    artifact_version: int,
    logger: logging.Logger,
) -> str:
    meeting = payload["meeting"]
    users = payload["users"]
    participants = payload["meeting_participants"]
    speakers = payload["meeting_speakers"]
    artifacts = payload["meeting_artifacts"]
    utterances = payload["utterances"]
    topic_segments = payload["topic_segments"]
    transitions = payload["utterance_transitions"]
    summary = payload["summary"]

    meeting_id = meeting["meeting_id"]

    with conn.cursor() as cur:
        meeting_exists, fully_ingested = get_ami_meeting_ingest_state(conn, meeting_id)

        if fully_ingested and not replace_existing:
            logger.info("Meeting already fully ingested, skipping: %s", meeting_id)
            conn.rollback()
            return "already_ingested"

        if meeting_exists and replace_existing:
            logger.info("Replacing existing AMI meeting rows: %s", meeting_id)
            cur.execute("DELETE FROM meetings WHERE meeting_id = %s", (meeting_id,))

        upsert_users(cur, users)

        cur.execute(
            """
            INSERT INTO meetings (
                meeting_id,
                source_type,
                source_name,
                started_at,
                ended_at,
                raw_folder_prefix,
                is_valid
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                meeting["meeting_id"],
                meeting["source_type"],
                meeting["source_name"],
                meeting["started_at"],
                meeting["ended_at"],
                meeting["raw_folder_prefix"],
                meeting["is_valid"],
            ),
        )

        for participant in participants:
            cur.execute(
                """
                INSERT INTO meeting_participants (
                    meeting_id,
                    user_id,
                    role,
                    can_view_summary,
                    can_edit_summary,
                    joined_at,
                    left_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    participant["meeting_id"],
                    participant["user_id"],
                    participant["role"],
                    participant["can_view_summary"],
                    participant["can_edit_summary"],
                    participant["joined_at"],
                    participant["left_at"],
                ),
            )

        speaker_id_by_label: dict[str, int] = {}
        for speaker in speakers:
            cur.execute(
                """
                INSERT INTO meeting_speakers (
                    meeting_id,
                    user_id,
                    speaker_label,
                    display_name,
                    role
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING meeting_speaker_id
                """,
                (
                    speaker["meeting_id"],
                    speaker["user_id"],
                    speaker["speaker_label"],
                    speaker["display_name"],
                    speaker["role"],
                ),
            )
            speaker_id_by_label[speaker["speaker_label"]] = fetch_returning_id(cur, "meeting_speaker_id")

        for artifact in artifacts:
            cur.execute(
                """
                INSERT INTO meeting_artifacts (
                    meeting_id,
                    artifact_type,
                    object_key,
                    content_type,
                    artifact_version
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    artifact["meeting_id"],
                    artifact["artifact_type"],
                    artifact["object_key"],
                    artifact["content_type"],
                    artifact["artifact_version"],
                ),
            )

        utterance_id_by_index: dict[int, int] = {}
        for utterance in utterances:
            cur.execute(
                """
                INSERT INTO utterances (
                    meeting_id,
                    meeting_speaker_id,
                    utterance_index,
                    start_time_sec,
                    end_time_sec,
                    raw_text,
                    clean_text,
                    source_segment_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING utterance_id
                """,
                (
                    meeting_id,
                    speaker_id_by_label[utterance["speaker_label"]],
                    utterance["utterance_index"],
                    utterance["start_time_sec"],
                    utterance["end_time_sec"],
                    utterance["raw_text"],
                    utterance["clean_text"],
                    utterance["source_segment_id"],
                ),
            )
            utterance_id_by_index[utterance["utterance_index"]] = fetch_returning_id(cur, "utterance_id")

        for topic in topic_segments:
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
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    meeting_id,
                    topic["segment_type"],
                    topic["segment_index"],
                    utterance_id_by_index[topic["start_utterance_index"]],
                    utterance_id_by_index[topic["end_utterance_index"]],
                    topic["start_time_sec"],
                    topic["end_time_sec"],
                    topic["topic_label"],
                ),
            )

        for transition in transitions:
            cur.execute(
                """
                INSERT INTO utterance_transitions (
                    meeting_id,
                    left_utterance_id,
                    right_utterance_id,
                    transition_index,
                    gold_boundary_label,
                    pred_boundary_prob,
                    pred_boundary_label
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    meeting_id,
                    utterance_id_by_index[transition["left_utterance_index"]],
                    utterance_id_by_index[transition["right_utterance_index"]],
                    transition["transition_index"],
                    transition["gold_boundary_label"],
                    transition["pred_boundary_prob"],
                    transition["pred_boundary_label"],
                ),
            )

        cur.execute(
            """
            INSERT INTO summaries (
                meeting_id,
                summary_type,
                summary_object_key,
                created_by_user_id,
                version
            )
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                meeting_id,
                summary["summary_type"],
                summary["summary_object_key"],
                summary["created_by_user_id"],
                artifact_version,
            ),
        )

    conn.commit()
    return "ingested"


def delete_uploaded_local_files(
    parsed_transcript_local: Path,
    summary_local: Path,
    manifest_local: Path,
    logger: logging.Logger,
) -> None:
    logger.info("Deleting uploaded local artifact files")
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


def ingest_meeting(
    *,
    meeting_id: str,
    raw_root: Path,
    processed_root: Path,
    raw_folder_prefix: str,
    processed_object_prefix: str,
    artifact_version: int,
    replace_existing: bool,
    cleanup_local_artifacts: bool,
    rclone_remote: str,
    bucket: str,
    logger: logging.Logger,
) -> str:
    processed_root.mkdir(parents=True, exist_ok=True)

    parsed_transcript_object_key = f"{processed_object_prefix.strip('/')}/transcripts/{meeting_id}.json"
    summary_object_key = f"{processed_object_prefix.strip('/')}/summaries/{meeting_id}.json"
    manifest_object_key = f"{processed_object_prefix.strip('/')}/manifests/{meeting_id}.json"

    meeting_row, user_rows, participant_rows, speaker_rows = load_meeting_metadata(
        raw_root,
        meeting_id,
        raw_folder_prefix=raw_folder_prefix,
    )
    utterances = load_utterances(raw_root, meeting_id, speaker_rows, logger)
    topic_segments = load_topic_segments(raw_root, meeting_id, utterances, logger)
    transitions = build_transition_rows(meeting_id, utterances, topic_segments)
    summary_payload = parse_summary_sections(raw_root, meeting_id)

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

    write_json(parsed_transcript_local, parsed_transcript_payload)
    write_json(summary_local, summary_payload)
    write_json(
        manifest_local,
        {
            "meeting_id": meeting_id,
            "counts": {
                "users": len(user_rows),
                "meeting_participants": len(participant_rows),
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
                "meeting": meeting_row,
                "user": user_rows[0] if user_rows else None,
                "meeting_participant": participant_rows[0] if participant_rows else None,
                "meeting_speaker": speaker_rows[0] if speaker_rows else None,
                "utterance": {
                    "meeting_id": utterances[0]["meeting_id"],
                    "speaker_label": utterances[0]["speaker_label"],
                    "utterance_index": utterances[0]["utterance_index"],
                    "start_time_sec": utterances[0]["start_time_sec"],
                    "end_time_sec": utterances[0]["end_time_sec"],
                    "raw_text": utterances[0]["raw_text"],
                    "source_segment_id": utterances[0]["source_segment_id"],
                }
                if utterances
                else None,
                "topic_segment": topic_segments[0] if topic_segments else None,
            },
        },
    )

    upload_file_to_object_storage(
        parsed_transcript_local,
        remote=rclone_remote,
        bucket=bucket,
        object_key=parsed_transcript_object_key,
        logger=logger,
    )
    upload_file_to_object_storage(
        summary_local,
        remote=rclone_remote,
        bucket=bucket,
        object_key=summary_object_key,
        logger=logger,
    )
    upload_file_to_object_storage(
        manifest_local,
        remote=rclone_remote,
        bucket=bucket,
        object_key=manifest_object_key,
        logger=logger,
    )

    payload = {
        "meeting": meeting_row,
        "users": user_rows,
        "meeting_participants": participant_rows,
        "meeting_speakers": speaker_rows,
        "meeting_artifacts": [
            {
                "meeting_id": meeting_id,
                "artifact_type": "parsed_transcript",
                "object_key": parsed_transcript_object_key,
                "content_type": "application/json",
                "artifact_version": artifact_version,
            },
            {
                "meeting_id": meeting_id,
                "artifact_type": "summary_json",
                "object_key": summary_object_key,
                "content_type": "application/json",
                "artifact_version": artifact_version,
            },
        ],
        "utterances": utterances,
        "topic_segments": topic_segments,
        "utterance_transitions": transitions,
        "summary": {
            "summary_type": "ami_gold",
            "summary_object_key": summary_object_key,
            "created_by_user_id": None,
        },
    }

    conn = get_conn()
    try:
        status = insert_rows(
            conn=conn,
            payload=payload,
            replace_existing=replace_existing,
            artifact_version=artifact_version,
            logger=logger,
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    if cleanup_local_artifacts:
        delete_uploaded_local_files(parsed_transcript_local, summary_local, manifest_local, logger)

    return status


def main() -> int:
    args = parse_args()
    logger = setup_logger(
        args.log_file or Path("/mnt/block/ingest_logs") / f"{APP_NAME}_{args.meeting}.log"
    )

    rclone_remote = os.getenv("RCLONE_REMOTE", "rclone_s3").strip() or "rclone_s3"
    bucket = (
        os.getenv("OBJECT_BUCKET", "").strip()
        or os.getenv("BUCKET", "").strip()
        or "objstore-proj07"
    )

    try:
        exists = False
        fully_ingested = False
        conn = get_conn()
        try:
            exists, fully_ingested = get_ami_meeting_ingest_state(conn, args.meeting)
        finally:
            conn.close()

        replace_existing = args.replace_existing or exists
        if fully_ingested and not args.replace_existing:
            logger.info("Meeting already fully ingested, skipping: %s", args.meeting)
            return 0

        status = ingest_meeting(
            meeting_id=args.meeting,
            raw_root=args.raw_root.resolve(),
            processed_root=args.processed_root.resolve(),
            raw_folder_prefix=args.raw_folder_prefix,
            processed_object_prefix=args.processed_object_prefix,
            artifact_version=args.artifact_version,
            replace_existing=replace_existing,
            cleanup_local_artifacts=not args.no_cleanup_local_artifacts,
            rclone_remote=rclone_remote,
            bucket=bucket,
            logger=logger,
        )
    except Exception:
        logger.exception("AMI ingest failed for %s", args.meeting)
        return 1

    logger.info("AMI ingest finished | meeting_id=%s | status=%s", args.meeting, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
