#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import string
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from feedback_common import (  # noqa: E402
    ensure_dir,
    get_conn,
    upload_file,
    write_json,
)


HEADER_RE = re.compile(r"^Transcript of conference held at (?P<date>.+?) in room (?P<room>.+)$")
START_RE = re.compile(r"^Transcript, started at (?P<time>.+):$")
END_RE = re.compile(r"^End of transcript at (?P<date>.+, \d{4}), (?P<time>.+)$")
EVENT_RE = re.compile(r"^<(?P<time>[^>]+)> (?P<body>.+)$")
SPOKEN_RE = re.compile(r"^(?P<speaker>[^:]+): (?P<text>.*)$")
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
    parser.add_argument("--timezone", default="America/New_York")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument(
        "--local-output-root",
        type=Path,
        default=Path("/mnt/block/user-behaviour/parsed_transcripts"),
    )
    parser.add_argument(
        "--raw-object-prefix",
        default="production/jitsi/raw_transcripts",
    )
    parser.add_argument(
        "--parsed-object-prefix",
        default="production/jitsi/parsed_transcripts",
    )
    parser.add_argument(
        "--skip-upload-artifacts",
        action="store_true",
        help="Do not upload raw/parsed artifacts to object storage; register local:// paths instead.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Delete any existing meeting with the same meeting_id before inserting fresh rows.",
    )
    parser.add_argument(
        "--build-stage1-after-ingest",
        action="store_true",
        help="After DB ingest, also create Stage 1 online inference request artifacts for this meeting.",
    )
    parser.add_argument(
        "--stage1-output-root",
        type=Path,
        default=Path("/mnt/block/user-behaviour/online_inference/stage1"),
    )
    parser.add_argument("--stage1-window-size", type=int, default=7)
    parser.add_argument("--stage1-transition-index", type=int, default=3)
    parser.add_argument("--stage1-min-utterance-chars", type=int, default=1)
    parser.add_argument("--stage1-max-words-per-utterance", type=int, default=50)
    parser.add_argument(
        "--upload-stage1-artifacts",
        action="store_true",
        help="Upload generated Stage 1 request artifacts to object storage.",
    )
    parser.add_argument(
        "--stage1-object-prefix",
        default="production/inference_requests/stage1",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional local log file path.",
    )
    return parser.parse_args()


def setup_logger(log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("ingest_jitsi_transcript")
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


def derive_meeting_id_from_filename(transcript_path: Path) -> str:
    match = TRANSCRIPT_FILENAME_RE.match(transcript_path.name)
    if not match:
        raise ValueError(
            "Transcript filename must follow the Jitsi export pattern "
            "'transcript_YYYY-MM-DDTHH:MM:SS(.fraction)Z_<uuid>.txt' so meeting_id "
            "can be derived automatically."
        )

    normalized_ts = match.group("ts").replace("-", "").replace(":", "")
    return f"jitsi_{normalized_ts}Z_{match.group('uuid8').lower()}"


def flush_current(current: dict | None, utterances: list[ParsedUtterance]) -> None:
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


def parse_transcript(path: Path, tz_name: str) -> dict:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    tz = ZoneInfo(tz_name)
    meeting_date: date | None = None
    meeting_room: str | None = None
    meeting_start: datetime | None = None
    meeting_end: datetime | None = None

    initial_people: list[str] = []
    seen_people: set[str] = set()
    utterances: list[ParsedUtterance] = []
    current: dict | None = None
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
            meeting_start = combine_local_datetime(meeting_date, start_match.group("time"), tz)
            previous_event_dt = meeting_start
            in_initial_people = False
            continue

        end_match = END_RE.match(line)
        if end_match:
            meeting_end = combine_local_datetime(parse_date(end_match.group("date")), end_match.group("time"), tz)
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

            event_dt = combine_local_datetime(previous_event_dt.date(), event_match.group("time"), tz)
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
        "participants": initial_people,
        "utterances": utterances,
    }


def build_ingest_payload(
    meeting_id: str,
    transcript_path: Path,
    parsed: dict,
    version: int,
    raw_object_prefix: str,
) -> dict:
    utterances: list[ParsedUtterance] = parsed["utterances"]
    participants: list[str] = parsed["participants"]
    start_dt: datetime = parsed["meeting_start"]
    end_dt: datetime = parsed["meeting_end"]

    speaker_order: list[str] = []
    for utterance in utterances:
        if utterance.speaker_name not in speaker_order:
            speaker_order.append(utterance.speaker_name)
    for participant in participants:
        if participant not in speaker_order:
            speaker_order.append(participant)

    speaker_labels = {
        name: speaker_label_for_index(idx)
        for idx, name in enumerate(speaker_order)
    }

    utterance_rows: list[dict] = []
    for idx, utterance in enumerate(utterances):
        next_start_dt = utterances[idx + 1].started_at if idx + 1 < len(utterances) else end_dt
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
                "t_start": start_seconds,
                "t_end": round(end_seconds, 3),
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
            "raw_folder_prefix": f"{raw_object_prefix.strip('/')}/{meeting_id}/",
        },
        "meeting_speakers": [
            {
                "meeting_id": meeting_id,
                "speaker_label": speaker_labels[name],
                "display_name": name,
                "role": None,
            }
            for name in speaker_order
        ],
        "meeting_artifact": {
            "meeting_id": meeting_id,
            "artifact_type": "raw_transcript",
            "object_key": transcript_path.name,
            "content_type": "text/plain",
            "artifact_version": version,
        },
        "ingest_notes": [
            "Join and leave lines are treated as system events and are not inserted into utterances.",
            "Utterance end times are assumed to be just before the next utterance start time.",
        ],
        "utterances": utterance_rows,
        "transition_placeholders": transitions,
    }


def upload_or_local_uri(local_path: Path, object_key: str, skip_upload: bool, logger) -> str:
    if skip_upload:
        return f"local://{local_path.resolve()}"
    upload_file(local_path, object_key, logger)
    return object_key


def insert_rows(
    conn,
    payload: dict,
    raw_object_key: str,
    parsed_object_key: str,
    replace_existing: bool,
    artifact_version: int,
) -> None:
    meeting = payload["meeting"]
    speakers = payload["meeting_speakers"]
    utterances = payload["utterances"]
    transitions = payload["transition_placeholders"]

    with conn.cursor() as cur:
        if replace_existing:
            cur.execute("DELETE FROM meetings WHERE meeting_id = %s", (meeting["meeting_id"],))

        cur.execute("SELECT 1 FROM meetings WHERE meeting_id = %s", (meeting["meeting_id"],))
        if cur.fetchone():
            raise RuntimeError(
                f"Meeting {meeting['meeting_id']} already exists. Use --replace-existing to overwrite it."
            )

        cur.execute(
            """
            INSERT INTO meetings (
                meeting_id, source_type, source_name, started_at, ended_at, raw_folder_prefix, is_valid
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
                False,
            ),
        )

        speaker_id_by_label: dict[str, int] = {}
        for speaker in speakers:
            cur.execute(
                """
                INSERT INTO meeting_speakers (
                    meeting_id, user_id, speaker_label, display_name, role
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
            speaker_id_by_label[speaker["speaker_label"]] = cur.fetchone()["meeting_speaker_id"]

        cur.execute(
            """
            INSERT INTO meeting_artifacts (
                meeting_id, artifact_type, object_key, content_type, artifact_version
            )
            VALUES (%s, 'raw_transcript', %s, 'text/plain', %s)
            """,
            (meeting["meeting_id"], raw_object_key, artifact_version),
        )
        cur.execute(
            """
            INSERT INTO meeting_artifacts (
                meeting_id, artifact_type, object_key, content_type, artifact_version
            )
            VALUES (%s, 'parsed_transcript', %s, 'application/json', %s)
            """,
            (meeting["meeting_id"], parsed_object_key, artifact_version),
        )

        utterance_id_by_index: dict[int, int] = {}
        for utterance in utterances:
            cur.execute(
                """
                INSERT INTO utterances (
                    meeting_id, meeting_speaker_id, utterance_index,
                    start_time_sec, end_time_sec, raw_text, clean_text, source_segment_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, NULL)
                RETURNING utterance_id
                """,
                (
                    meeting["meeting_id"],
                    speaker_id_by_label[utterance["speaker_label"]],
                    utterance["utterance_index"],
                    utterance["t_start"],
                    utterance["t_end"],
                    utterance["raw_text"],
                    utterance["clean_text"],
                ),
            )
            utterance_id_by_index[utterance["utterance_index"]] = cur.fetchone()["utterance_id"]

        for transition in transitions:
            cur.execute(
                """
                INSERT INTO utterance_transitions (
                    meeting_id, left_utterance_id, right_utterance_id,
                    transition_index, gold_boundary_label, pred_boundary_prob, pred_boundary_label
                )
                VALUES (%s, %s, %s, %s, NULL, NULL, NULL)
                """,
                (
                    meeting["meeting_id"],
                    utterance_id_by_index[transition["left_utterance_index"]],
                    utterance_id_by_index[transition["right_utterance_index"]],
                    transition["transition_index"],
                ),
            )

    conn.commit()


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_file)
    transcript_path = args.transcript_path

    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    meeting_id = derive_meeting_id_from_filename(transcript_path)
    logger.info("Derived meeting_id=%s from transcript filename=%s", meeting_id, transcript_path.name)

    parsed = parse_transcript(transcript_path, args.timezone)
    payload = build_ingest_payload(
        meeting_id,
        transcript_path,
        parsed,
        args.version,
        args.raw_object_prefix,
    )

    local_root = args.local_output_root / meeting_id
    ensure_dir(local_root)
    parsed_path = local_root / "parsed_transcript.json"
    write_json(parsed_path, payload)

    raw_object_key = (
        f"{args.raw_object_prefix.strip('/')}/{meeting_id}/"
        f"v{args.version}_{transcript_path.name}"
    )
    parsed_object_key = (
        f"{args.parsed_object_prefix.strip('/')}/{meeting_id}/v{args.version}.json"
    )

    raw_uri = upload_or_local_uri(
        transcript_path,
        raw_object_key,
        skip_upload=args.skip_upload_artifacts,
        logger=logger,
    )
    parsed_uri = upload_or_local_uri(
        parsed_path,
        parsed_object_key,
        skip_upload=args.skip_upload_artifacts,
        logger=logger,
    )

    conn = get_conn()
    insert_rows(
        conn=conn,
        payload=payload,
        raw_object_key=raw_uri,
        parsed_object_key=parsed_uri,
        replace_existing=args.replace_existing,
        artifact_version=args.version,
    )

    logger.info(
        "Ingested Jitsi transcript meeting_id=%s utterances=%d transitions=%d",
        meeting_id,
        len(payload["utterances"]),
        len(payload["transition_placeholders"]),
    )

    if args.build_stage1_after_ingest:
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "build_online_inference_payloads.py"),
            "--meeting-id",
            meeting_id,
            "--window-size",
            str(args.stage1_window_size),
            "--transition-index",
            str(args.stage1_transition_index),
            "--min-utterance-chars",
            str(args.stage1_min_utterance_chars),
            "--max-words-per-utterance",
            str(args.stage1_max_words_per_utterance),
            "--output-root",
            str(args.stage1_output_root),
            "--version",
            str(args.version),
        ]
        if args.upload_stage1_artifacts:
            cmd.extend(["--upload-artifacts", "--stage1-object-prefix", args.stage1_object_prefix])

        logger.info("Building Stage 1 request artifacts immediately after ingest")
        result = subprocess.run(cmd, text=True, capture_output=True)
        if result.returncode != 0:
            if result.stdout:
                logger.error("Stage 1 builder stdout:\n%s", result.stdout)
            if result.stderr:
                logger.error("Stage 1 builder stderr:\n%s", result.stderr)
            raise RuntimeError("Stage 1 request build failed after ingest")
        if result.stdout:
            logger.info("Stage 1 builder output:\n%s", result.stdout.strip())


if __name__ == "__main__":
    main()
