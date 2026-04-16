#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import string
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

# Optional project helper. If not available, fallback to DATABASE_URL.
try:
    from feedback_common import get_conn as project_get_conn  # type: ignore
except Exception:
    project_get_conn = None


APP_NAME = "ingest_saved_jitsi_transcript"

HEADER_RE = re.compile(
    r"^Transcript of conference held at (?P<date>.+?) in room (?P<room>.+)$"
)
START_RE = re.compile(r"^Transcript, started at (?P<time>.+):$")
END_RE = re.compile(r"^End of transcript at (?P<date>.+, \d{4}), (?P<time>.+)$")
EVENT_RE = re.compile(r"^<(?P<time>[^>]+)> (?P<body>.+)$")
SPOKEN_RE = re.compile(r"^(?P<speaker>[^:]+): (?P<text>.*)$")

# IMPORTANT:
# meeting_id is derived from the ORIGINAL uploaded Jitsi filename,
# not from the saved UUID-prefixed receiver filename.
TRANSCRIPT_FILENAME_RE = re.compile(
    r"^transcript_(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.\d+)?Z_(?P<uuid8>[0-9a-fA-F]{8})[0-9a-fA-F-]*\.txt$"
)


@dataclass
class ParsedUtterance:
    speaker_name: str
    started_at: datetime
    raw_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript-path", type=Path, required=True)
    parser.add_argument("--original-filename", required=True)
    parser.add_argument("--host-external-key", required=True)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--timezone", default="America/New_York")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument(
        "--local-output-root",
        type=Path,
        default=Path("/mnt/block/user-behaviour/parsed_transcripts"),
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Delete existing rows for the same meeting_id before inserting.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def setup_logger(log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(APP_NAME)
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def get_db_conn():
    if project_get_conn is not None:
        return project_get_conn()

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError(
            "No DB connection helper found and DATABASE_URL is not set."
        )

    try:
        import psycopg2  # type: ignore

        return psycopg2.connect(database_url)
    except Exception:
        try:
            import psycopg  # type: ignore

            return psycopg.connect(database_url)
        except Exception as exc:
            raise RuntimeError(
                "Failed to connect to Postgres using DATABASE_URL."
            ) from exc


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def make_clean_text(raw_text: str) -> str:
    lowered = raw_text.lower()
    lowered = re.sub(r"\[[^\]]+\]", " ", lowered)
    return normalize_whitespace(lowered)


def parse_date(date_str: str) -> date:
    return datetime.strptime(date_str.strip(), "%b %d, %Y").date()


def parse_time(time_str: str) -> time:
    return datetime.strptime(time_str.strip(), "%I:%M:%S %p").time()


def combine_local_datetime(day: date, time_str: str, tz: ZoneInfo) -> datetime:
    return datetime.combine(day, parse_time(time_str), tzinfo=tz)


def speaker_label_for_index(index: int) -> str:
    letters = string.ascii_uppercase
    label = ""
    value = index
    while True:
        label = letters[value % 26] + label
        value = value // 26 - 1
        if value < 0:
            return label


def derive_meeting_id_from_original_filename(original_filename: str) -> str:
    base_name = Path(original_filename).name
    match = TRANSCRIPT_FILENAME_RE.match(base_name)
    if not match:
        raise ValueError(
            "Original filename must follow: "
            "transcript_YYYY-MM-DDTHH:MM:SS(.fraction)Z_<uuid>.txt"
        )

    normalized_ts = match.group("ts").replace("-", "").replace(":", "")
    return f"jitsi_{normalized_ts}Z_{match.group('uuid8').lower()}"


def flush_current(
    current: dict[str, Any] | None,
    utterances: list[ParsedUtterance],
) -> None:
    if not current:
        return

    raw_text = normalize_whitespace(" ".join(current["text_parts"]))
    if not raw_text:
        return

    utterances.append(
        ParsedUtterance(
            speaker_name=current["speaker_name"],
            started_at=current["started_at"],
            raw_text=raw_text,
        )
    )


def parse_transcript(path: Path, tz_name: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    tz = ZoneInfo(tz_name)
    meeting_date: date | None = None
    meeting_room: str | None = None
    meeting_start: datetime | None = None
    meeting_end: datetime | None = None

    initial_people: list[str] = []
    seen_people: set[str] = set()
    utterances: list[ParsedUtterance] = []
    current: dict[str, Any] | None = None
    previous_event_dt: datetime | None = None
    in_initial_people = False

    for raw_line in lines:
        line = raw_line.rstrip()

        header_match = HEADER_RE.match(line)
        if header_match:
            meeting_date = parse_date(header_match.group("date"))
            meeting_room = header_match.group("room").strip()
            continue

        if line.startswith("Initial people present at "):
            in_initial_people = True
            continue

        start_match = START_RE.match(line)
        if start_match:
            if meeting_date is None:
                raise ValueError("Transcript start found before meeting date header")
            meeting_start = combine_local_datetime(
                meeting_date,
                start_match.group("time"),
                tz,
            )
            previous_event_dt = meeting_start
            in_initial_people = False
            continue

        end_match = END_RE.match(line)
        if end_match:
            meeting_end = combine_local_datetime(
                parse_date(end_match.group("date")),
                end_match.group("time"),
                tz,
            )
            continue

        if in_initial_people:
            person = line.strip()
            if person and not person.startswith("_"):
                if person not in seen_people:
                    initial_people.append(person)
                    seen_people.add(person)
            continue

        if not line or line.startswith("_"):
            continue

        event_match = EVENT_RE.match(line)
        if event_match:
            flush_current(current, utterances)
            current = None

            if meeting_date is None or previous_event_dt is None:
                raise ValueError("Encountered transcript event before meeting start")

            event_dt = combine_local_datetime(
                previous_event_dt.date(),
                event_match.group("time"),
                tz,
            )
            if event_dt < previous_event_dt:
                event_dt = event_dt + timedelta(days=1)
            previous_event_dt = event_dt

            body = event_match.group("body").strip()
            spoken_match = SPOKEN_RE.match(body)
            if spoken_match:
                speaker_name = spoken_match.group("speaker").strip()
                current = {
                    "speaker_name": speaker_name,
                    "started_at": event_dt,
                    "text_parts": [spoken_match.group("text").strip()],
                }
                if speaker_name not in seen_people:
                    initial_people.append(speaker_name)
                    seen_people.add(speaker_name)
            else:
                joined_name = None
                if body.endswith(" joined the conference"):
                    joined_name = body[: -len(" joined the conference")].strip()
                elif body.endswith(" left the conference"):
                    joined_name = body[: -len(" left the conference")].strip()

                if joined_name and joined_name not in seen_people:
                    initial_people.append(joined_name)
                    seen_people.add(joined_name)
            continue

        if current and raw_line.startswith(" "):
            current["text_parts"].append(line.strip())

    flush_current(current, utterances)

    if meeting_room is None or meeting_start is None or meeting_end is None:
        raise ValueError("Transcript is missing required header/start/end metadata")

    if not utterances:
        raise ValueError("No spoken utterances were found in transcript")

    return {
        "meeting_room": meeting_room,
        "meeting_start": meeting_start,
        "meeting_end": meeting_end,
        "participants_in_transcript": initial_people,
        "utterances": utterances,
    }


def build_parsed_payload(
    *,
    meeting_id: str,
    transcript_path: Path,
    original_filename: str,
    host_external_key: str,
    parsed: dict[str, Any],
    version: int,
) -> dict[str, Any]:
    utterances: list[ParsedUtterance] = parsed["utterances"]
    start_dt: datetime = parsed["meeting_start"]
    end_dt: datetime = parsed["meeting_end"]

    speaker_order: list[str] = []
    for utterance in utterances:
        if utterance.speaker_name not in speaker_order:
            speaker_order.append(utterance.speaker_name)

    speaker_labels = {
        name: speaker_label_for_index(idx)
        for idx, name in enumerate(speaker_order)
    }

    utterance_rows: list[dict[str, Any]] = []
    for idx, utterance in enumerate(utterances):
        next_start_dt = (
            utterances[idx + 1].started_at if idx + 1 < len(utterances) else end_dt
        )
        end_seconds = max(
            (next_start_dt - start_dt).total_seconds() - 0.001,
            (utterance.started_at - start_dt).total_seconds(),
        )
        start_seconds = round((utterance.started_at - start_dt).total_seconds(), 3)

        utterance_rows.append(
            {
                "utterance_index": idx,
                "speaker_label": speaker_labels[utterance.speaker_name],
                "display_name": utterance.speaker_name,
                "start_time_sec": start_seconds,
                "end_time_sec": round(end_seconds, 3),
                "raw_text": utterance.raw_text,
                "clean_text": make_clean_text(utterance.raw_text),
            }
        )

    transitions = [
        {
            "transition_index": idx,
            "left_utterance_index": idx,
            "right_utterance_index": idx + 1,
            "gold_boundary_label": None,
            "pred_boundary_prob": None,
            "pred_boundary_label": None,
        }
        for idx in range(len(utterance_rows) - 1)
    ]

    return {
        "meeting": {
            "meeting_id": meeting_id,
            "source_type": "jitsi",
            "source_name": parsed["meeting_room"],
            "started_at": start_dt.isoformat(),
            "ended_at": end_dt.isoformat(),
            "raw_folder_prefix": str(transcript_path.parent),
        },
        "host": {
            "user_id": host_external_key,
            "display_name": host_external_key,
        },
        "meeting_participants": [
            {
                "meeting_id": meeting_id,
                "user_id": host_external_key,
                "role": "host",
                "can_view_summary": True,
                "can_edit_summary": True,
                "joined_at": start_dt.isoformat(),
                "left_at": end_dt.isoformat(),
            }
        ],
        "meeting_speakers": [
            {
                "meeting_id": meeting_id,
                "speaker_label": speaker_labels[name],
                "display_name": name,
                "role": None,
            }
            for name in speaker_order
        ],
        "meeting_artifacts": [
            {
                "meeting_id": meeting_id,
                "artifact_type": "raw_transcript",
                "object_key": f"local://{transcript_path.resolve()}",
                "content_type": "text/plain",
                "artifact_version": version,
            },
            {
                "meeting_id": meeting_id,
                "artifact_type": "parsed_transcript",
                "object_key": "",  # filled later after parsed JSON is written
                "content_type": "application/json",
                "artifact_version": version,
            },
        ],
        "ingest_notes": [
            f"meeting_id derived from original filename: {original_filename}",
            "meeting_participants currently includes only the host uploader identity.",
            "meeting_speakers contains transcript-level spoken identities only.",
            "meeting_speakers.user_id is left NULL for now.",
            "Utterance end time is assumed to be just before the next utterance start time.",
        ],
        "participants_in_transcript": parsed["participants_in_transcript"],
        "utterances": utterance_rows,
        "transition_placeholders": transitions,
    }


def upsert_host_user(cur, host_external_key: str) -> None:
    # Assumes updated schema:
    # users.user_id TEXT PRIMARY KEY
    cur.execute(
        """
        INSERT INTO users (user_id, display_name, email)
        VALUES (%s, %s, NULL)
        ON CONFLICT (user_id)
        DO UPDATE SET display_name = EXCLUDED.display_name
        """,
        (host_external_key, host_external_key),
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
    host = payload["host"]
    participants = payload["meeting_participants"]
    speakers = payload["meeting_speakers"]
    artifacts = payload["meeting_artifacts"]
    utterances = payload["utterances"]
    transitions = payload["transition_placeholders"]

    meeting_id = meeting["meeting_id"]

    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM meetings WHERE meeting_id = %s", (meeting_id,))
        already_exists = cur.fetchone() is not None

        if already_exists and not replace_existing:
            logger.info("Meeting already exists, skipping ingest: %s", meeting_id)
            conn.rollback()
            return "already_ingested"

        if already_exists and replace_existing:
            logger.info("Replacing existing meeting: %s", meeting_id)
            cur.execute("DELETE FROM meetings WHERE meeting_id = %s", (meeting_id,))

        upsert_host_user(cur, host["user_id"])

        cur.execute(
            """
            INSERT INTO meetings (
                meeting_id, source_type, source_name, started_at, ended_at, raw_folder_prefix
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                meeting["meeting_id"],
                meeting["source_type"],
                meeting["source_name"],
                meeting["started_at"],
                meeting["ended_at"],
                meeting["raw_folder_prefix"],
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
                VALUES (%s, NULL, %s, %s, %s)
                RETURNING meeting_speaker_id
                """,
                (
                    speaker["meeting_id"],
                    speaker["speaker_label"],
                    speaker["display_name"],
                    speaker["role"],
                ),
            )
            speaker_id_by_label[speaker["speaker_label"]] = cur.fetchone()[0]

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
                VALUES (%s, %s, %s, %s, %s, %s, %s, NULL)
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
                ),
            )
            utterance_id_by_index[utterance["utterance_index"]] = cur.fetchone()[0]

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

    conn.commit()
    return "ingested"


def update_metadata_sidecar(
    *,
    metadata_path: Path | None,
    meeting_id: str,
    parsed_json_path: Path,
    ingest_status: str,
    logger: logging.Logger,
) -> None:
    if metadata_path is None:
        return

    try:
        payload: dict[str, Any] = {}
        if metadata_path.exists():
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))

        payload["meeting_id"] = meeting_id
        payload["parsed_json_path"] = str(parsed_json_path)
        payload["ingest_status"] = ingest_status

        metadata_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        logger.exception("Failed updating metadata sidecar: %s", metadata_path)


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_file)

    transcript_path = args.transcript_path
    original_filename = Path(args.original_filename).name
    host_external_key = args.host_external_key.strip()

    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    if not host_external_key:
        raise ValueError("host_external_key cannot be empty")

    meeting_id = derive_meeting_id_from_original_filename(original_filename)
    logger.info(
        "Derived meeting_id=%s from original filename=%s",
        meeting_id,
        original_filename,
    )

    parsed = parse_transcript(transcript_path, args.timezone)

    payload = build_parsed_payload(
        meeting_id=meeting_id,
        transcript_path=transcript_path,
        original_filename=original_filename,
        host_external_key=host_external_key,
        parsed=parsed,
        version=args.version,
    )

    local_root = args.local_output_root / meeting_id
    ensure_dir(local_root)
    parsed_path = local_root / "parsed_transcript.json"

    # Fill parsed artifact local URI now that we know the parsed file path.
    payload["meeting_artifacts"][1]["object_key"] = f"local://{parsed_path.resolve()}"

    write_json(parsed_path, payload)

    conn = get_db_conn()
    try:
        ingest_status = insert_rows(
            conn=conn,
            payload=payload,
            replace_existing=args.replace_existing,
            artifact_version=args.version,
            logger=logger,
        )
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass

    update_metadata_sidecar(
        metadata_path=args.metadata_path,
        meeting_id=meeting_id,
        parsed_json_path=parsed_path,
        ingest_status=ingest_status,
        logger=logger,
    )

    logger.info(
        "Finished Jitsi ingest | meeting_id=%s | status=%s | utterances=%d | transitions=%d",
        meeting_id,
        ingest_status,
        len(payload["utterances"]),
        len(payload["transition_placeholders"]),
    )

    print(
        json.dumps(
            {
                "meeting_id": meeting_id,
                "status": ingest_status,
                "parsed_json_path": str(parsed_path),
                "utterances": len(payload["utterances"]),
                "transitions": len(payload["transition_placeholders"]),
            }
        )
    )


if __name__ == "__main__":
    main()