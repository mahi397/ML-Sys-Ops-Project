#!/usr/bin/env python3
"""
emulate_production.py - Production traffic emulator for the NeuralOps demo

Emulates completed Jitsi transcript uploads by driving the same public ingest
API the production stack uses:

  /ingest/jitsi-transcript -> ingest + downstream processing

Usage (standalone):
  python data/emulate_production.py

Usage (docker compose):
  docker compose --profile emulated-traffic up -d traffic-generator

Environment variables:
  INGEST_URL                  Full ingest endpoint URL
                              (default: http://jitsi_transcript_receiver:9000/ingest/jitsi-transcript)
  MEETING_COUNT               Meetings per batch (default: 5, 0 = run forever)
  DELAY_SECONDS               Pause between meetings (default: 10)
  INGEST_TOKEN                Bearer token for ingest, if enabled (default: none)
  HOST_EXTERNAL_KEY           Stable uploader identity
                              (default: emulated-traffic-generator)
  MEETING_SOURCE_MODE         synthetic | archived | mixed
                              (default: mixed)
  ARCHIVED_TRANSCRIPT_ROOT    Root folder for archived Jitsi mock transcripts
                              (default: ./initial_implementation/mock_jitsi_meet)
  IDENTITY_SOURCE             Identity source sent with synthetic participants
                              (default: emulate_production)
  MEETING_NAME_PREFIX         Prefix for synthetic meeting names
                              (default: synthetic)
  REQUEST_TIMEOUT_SECONDS     Timeout for individual HTTP calls (default: 120)
  SEED                        Random seed (default: 42)
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import textwrap
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("emulate_production")

# --- config -----------------------------------------------------------------

INGEST_URL = os.environ.get(
    "INGEST_URL",
    "http://http://192.5.86.182:9000/ingest/jitsi-transcript",
).rstrip("/")
MEETING_COUNT = int(os.environ.get("MEETING_COUNT", "5"))
DELAY_SECONDS = float(os.environ.get("DELAY_SECONDS", "10"))
INGEST_TOKEN = os.environ.get("INGEST_TOKEN", "").strip()
HOST_EXTERNAL_KEY = os.environ.get(
    "HOST_EXTERNAL_KEY",
    os.environ.get("JITSI_HOST_EXTERNAL_KEY", "emulated-traffic-generator"),
).strip()
MEETING_SOURCE_MODE = os.environ.get("MEETING_SOURCE_MODE", "mixed").strip().lower() or "mixed"
ARCHIVED_TRANSCRIPT_ROOT = Path(
    os.environ.get(
        "ARCHIVED_TRANSCRIPT_ROOT",
        str(Path(__file__).resolve().parent / "initial_implementation" / "mock_jitsi_meet"),
    )
).expanduser()
IDENTITY_SOURCE = os.environ.get("IDENTITY_SOURCE", "emulate_production").strip()
MEETING_NAME_PREFIX = os.environ.get("MEETING_NAME_PREFIX", "synthetic").strip() or "synthetic"
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "120"))
SEED = int(os.environ.get("SEED", "42"))

TRANSCRIPT_FILENAME_RE = re.compile(
    r"^transcript_(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.\d+)?Z_(?P<uuid8>[0-9a-fA-F]{8})[0-9a-fA-F-]*\.txt$"
)
TRANSCRIPT_SEPARATOR = "_" * 80
HEADER_RE = re.compile(r"^Transcript of conference held at (?P<date>.+?) in room (?P<room>.+)$")
EVENT_RE = re.compile(r"^<(?P<time>[^>]+)> (?P<body>.+)$")
SPOKEN_RE = re.compile(r"^(?P<speaker>[^:]+): (?P<text>.+)$")

# --- realistic meeting content ----------------------------------------------

TOPIC_BLOCKS = [
    {
        "topic": "project status review",
        "utterances": [
            ("Alice", "Let's kick off with a quick status check on the ML pipeline."),
            ("Bob", "The serving container has been stable since last Tuesday with no restarts."),
            ("Alice", "Good. What about the retraining job? Did the last run pass quality gates?"),
            ("Bob", "Yes, Pk came in at 0.21 and F1 at 0.23, both within threshold."),
            ("Carol", "I verified the candidate alias is set in MLflow and ready for promotion review."),
            ("Alice", "Let's schedule that for tomorrow once the team has reviewed the model card."),
        ],
    },
    {
        "topic": "feedback data quality",
        "utterances": [
            ("Bob", "We have 82 new boundary corrections from last week's meetings."),
            ("Carol", "That is enough to rebuild the feedback pool but still below the production threshold."),
            ("Alice", "Are any of the corrections concentrated in single speaker meetings?"),
            ("Bob", "About a third of them, yes. That slice has historically lower Pk."),
            ("Carol", "We should check the slice metrics in MLflow after the next retrain."),
            ("Alice", "Agreed. I'll add a note to the model card review checklist."),
        ],
    },
    {
        "topic": "infrastructure costs",
        "utterances": [
            ("Dave", "The GPU node has been running continuously since April and block storage is at 40 percent."),
            ("Alice", "Are the MinIO ray checkpoints growing unbounded?"),
            ("Dave", "Ray keeps the last two checkpoints per run, so it stays bounded."),
            ("Carol", "Good. What about MLflow artifact storage on chi.tacc?"),
            ("Dave", "Each model is around 400 MB. We have five versions so far, about 2 GB total."),
            ("Alice", "That is fine for now. Let's revisit when we approach the bucket limit."),
        ],
    },
    {
        "topic": "serving latency discussion",
        "utterances": [
            ("Bob", "Segmentation p95 is 173 ms at concurrency 5, well inside the 2 second SLA."),
            ("Carol", "Summarization is slower, but the async pipeline means users do not feel it."),
            ("Dave", "I noticed the GPU memory metric in Grafana shows zero. Is that a bug?"),
            ("Bob", "Yes, it is a known issue because MetricsDeployment runs in a separate Ray worker process."),
            ("Carol", "The actual usage visible in nvidia-smi is around 6 GB, which is expected."),
            ("Alice", "We should switch to pynvml for system wide GPU queries before the final demo."),
        ],
    },
    {
        "topic": "data pipeline health",
        "utterances": [
            ("Carol", "Stage 1 forward service processed 200 requests overnight without errors."),
            ("Dave", "The drift monitor ran at 3 AM and found no significant feature drift."),
            ("Alice", "Good. How many meetings were marked valid in the last 24 hours?"),
            ("Carol", "Fourteen Jitsi meetings. All passed the utterance and stage artifact checks."),
            ("Bob", "That brings the unconsumed valid meeting count to 32, below the retraining threshold."),
            ("Dave", "We will hit the threshold after the next busy meeting day."),
        ],
    },
    {
        "topic": "model promotion process",
        "utterances": [
            ("Alice", "The promotion workflow is retrain passes gates and the candidate alias is set automatically."),
            ("Bob", "Then a team member reviews the model card in MLflow and manually sets the production alias."),
            ("Carol", "The serving layer polls the registry every 5 minutes and hot reloads without a restart."),
            ("Dave", "What triggers rollback if the promoted model degrades in production?"),
            ("Alice", "Online correction rate. If it exceeds 15 percent, we roll back to the fallback alias."),
            ("Bob", "The retrain log and audit log both capture the watermark and gate results for each run."),
        ],
    },
    {
        "topic": "fairness evaluation review",
        "utterances": [
            ("Carol", "The fairness gate checks six slices: short, medium, long meetings, and speaker count."),
            ("Dave", "Single speaker meetings have the worst Pk because there is no speaker change signal."),
            ("Alice", "That is expected. The slice gate threshold is 0.40, higher than the aggregate 0.25."),
            ("Bob", "Speaker relabeling invariance also passed. Pk did not degrade when we renamed speakers."),
            ("Carol", "That confirms the model is reading content and not just latching onto speaker identity tokens."),
            ("Alice", "Good. All of this is documented in the model card logged to MLflow."),
        ],
    },
    {
        "topic": "next sprint planning",
        "utterances": [
            ("Dave", "For the next sprint I want to add pynvml based GPU metrics to the Grafana dashboard."),
            ("Alice", "I'd like to increase the retrain threshold back to 500 after the demo."),
            ("Bob", "We should also look at adding a second feedback type like thumbs up on segments."),
            ("Carol", "That could lower the correction rate metric and help distinguish good boundaries."),
            ("Dave", "I can wire it up on the serving side once we agree on the event schema."),
            ("Alice", "Let's draft the schema in the contracts folder and review it async."),
        ],
    },
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def structured_log(level: str, event: str, **fields: Any) -> None:
    record = {
        "ts": utc_now_iso(),
        "level": level.upper(),
        "event": event,
        **fields,
    }
    getattr(log, level)(json.dumps(record, sort_keys=True, default=str))


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "mock-user"


def make_mock_user(display_name: str) -> dict[str, str]:
    slug = slugify(display_name)
    return {
        "user_id": f"speaker_{slug}",
        "display_name": display_name,
        "email": f"{slug}.mock@example.com",
        "identity_source": IDENTITY_SOURCE,
    }


def select_mock_user(speaker_names: list[str], rng: random.Random) -> dict[str, str]:
    return make_mock_user(rng.choice(speaker_names))


def build_room_participants(speaker_names: list[str]) -> list[dict[str, str]]:
    return [make_mock_user(speaker_name) for speaker_name in speaker_names]


def meeting_log_fields(payload: dict[str, Any]) -> dict[str, Any]:
    mock_user = payload["mock_user"]
    return {
        "meeting_id": payload["meeting_id"],
        "meeting_name": payload["meeting_name"],
        "meeting_room": payload["meeting_room"],
        "original_filename": payload["original_filename"],
        "meeting_source": payload["meeting_source"],
        "mock_user_id": mock_user["user_id"],
        "mock_user_display_name": mock_user["display_name"],
        "mock_user_email": mock_user["email"],
        "host_external_key": HOST_EXTERNAL_KEY,
    }


def derive_meeting_id_from_original_filename(original_filename: str) -> str:
    match = TRANSCRIPT_FILENAME_RE.match(original_filename)
    if not match:
        raise ValueError(
            "Original filename must follow transcript_YYYY-MM-DDTHH:MM:SS(.fraction)Z_<uuid>.txt"
        )
    normalized_ts = match.group("ts").replace("-", "").replace(":", "")
    return f"jitsi_{normalized_ts}Z_{match.group('uuid8').lower()}"


def format_meeting_date(dt: datetime) -> str:
    return dt.strftime("%b %d, %Y")


def format_clock(dt: datetime) -> str:
    return dt.strftime("%I:%M:%S %p").lstrip("0")


def make_original_filename(meeting_start: datetime) -> str:
    return f"transcript_{meeting_start.strftime('%Y-%m-%dT%H:%M:%S.%fZ')}_{uuid.uuid4()}.txt"


def make_meeting_name(meeting_id: str) -> str:
    return f"{MEETING_NAME_PREFIX}-{meeting_id}"


def list_archived_transcripts(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.glob("transcript_*.txt") if path.is_file())


def append_unique(items: list[str], value: str | None) -> None:
    normalized = (value or "").strip()
    if normalized and normalized not in items:
        items.append(normalized)


def summarize_archived_transcript_text(text: str) -> dict[str, Any]:
    meeting_room = ""
    participants: list[str] = []
    speaker_names: list[str] = []
    utterance_count = 0
    in_initial_people = False
    current_speaker: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        header_match = HEADER_RE.match(line)
        if header_match:
            meeting_room = header_match.group("room").strip()
            continue

        if line.startswith("Initial people present at "):
            in_initial_people = True
            current_speaker = None
            continue

        if line.startswith("Transcript, started at "):
            in_initial_people = False
            current_speaker = None
            continue

        if in_initial_people:
            append_unique(participants, line.strip())
            continue

        if not line or line.startswith("_"):
            continue

        event_match = EVENT_RE.match(line)
        if event_match:
            current_speaker = None
            body = event_match.group("body").strip()
            spoken_match = SPOKEN_RE.match(body)
            if spoken_match:
                speaker_name = spoken_match.group("speaker").strip()
                append_unique(speaker_names, speaker_name)
                append_unique(participants, speaker_name)
                utterance_count += 1
                current_speaker = speaker_name
            else:
                for suffix in (" joined the conference", " left the conference"):
                    if body.endswith(suffix):
                        append_unique(participants, body[: -len(suffix)].strip())
                        break
            continue

        if raw_line.startswith(" ") and current_speaker:
            continue

    if not speaker_names:
        speaker_names = list(participants)

    return {
        "meeting_room": meeting_room,
        "participants": participants,
        "speaker_names": speaker_names,
        "utterance_count": utterance_count,
    }


def build_utterances(rng: random.Random) -> list[dict[str, Any]]:
    """Pick 2-3 topic blocks and expand them into a timed utterance list."""
    blocks = rng.sample(TOPIC_BLOCKS, k=rng.randint(2, 3))
    utterances: list[dict[str, Any]] = []
    t = 0.0
    for block in blocks:
        for speaker, text in block["utterances"]:
            duration = rng.uniform(3.0, 8.0)
            utterances.append(
                {
                    "speaker": speaker,
                    "text": text,
                    "t_start": round(t, 1),
                    "t_end": round(t + duration, 1),
                }
            )
            t += duration + rng.uniform(0.5, 2.0)
    return utterances


def wrap_spoken_event(prefix: str, text: str) -> list[str]:
    width = max(24, 88 - len(prefix))
    wrapped = textwrap.wrap(text, width=width) or [""]
    continuation_prefix = " " * len(prefix)
    lines = [f"{prefix}{wrapped[0]}"]
    lines.extend(f"{continuation_prefix}{part}" for part in wrapped[1:])
    return lines


def render_transcript(
    meeting_start: datetime,
    meeting_room: str,
    utterances: list[dict[str, Any]],
) -> str:
    unique_speakers = list(dict.fromkeys(utterance["speaker"] for utterance in utterances))
    join_buffer_seconds = len(unique_speakers) * 2 + 3
    transcript_end = meeting_start + timedelta(
        seconds=join_buffer_seconds + utterances[-1]["t_end"] + 10
    )

    lines: list[str] = [
        f"Transcript of conference held at {format_meeting_date(meeting_start)} in room {meeting_room}",
        f"Initial people present at {format_clock(meeting_start)}:",
    ]
    lines.extend(f"\t{speaker}" for speaker in unique_speakers)
    lines.extend(["", f"Transcript, started at {format_clock(meeting_start)}:", TRANSCRIPT_SEPARATOR])

    for index, speaker in enumerate(unique_speakers):
        joined_at = meeting_start + timedelta(seconds=index * 2)
        lines.append(f"<{format_clock(joined_at)}> {speaker} joined the conference")

    for utterance in utterances:
        spoken_at = meeting_start + timedelta(seconds=join_buffer_seconds + utterance["t_start"])
        prefix = f"<{format_clock(spoken_at)}> {utterance['speaker']}: "
        lines.extend(wrap_spoken_event(prefix, utterance["text"]))

    lines.extend(
        [
            TRANSCRIPT_SEPARATOR,
            "",
            "",
            f"End of transcript at {format_meeting_date(transcript_end)}, {format_clock(transcript_end)}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_synthetic_meeting_payload(rng: random.Random) -> dict[str, Any]:
    meeting_start = datetime.now(timezone.utc).replace(microsecond=0)
    original_filename = make_original_filename(meeting_start)
    meeting_id = derive_meeting_id_from_original_filename(original_filename)
    meeting_name = make_meeting_name(meeting_id)
    meeting_room = meeting_name
    utterances = build_utterances(rng)
    speaker_names = list(dict.fromkeys(utterance["speaker"] for utterance in utterances))
    mock_user = select_mock_user(speaker_names, rng)
    room_participants = build_room_participants(speaker_names)
    transcript_text = render_transcript(meeting_start, meeting_room, utterances)
    return {
        "meeting_id": meeting_id,
        "meeting_name": meeting_name,
        "meeting_room": meeting_room,
        "original_filename": original_filename,
        "utterances": utterances,
        "speaker_names": speaker_names,
        "mock_user": mock_user,
        "room_participants": room_participants,
        "transcript_text": transcript_text,
        "utterance_count": len(utterances),
        "meeting_source": "synthetic",
    }


def build_archived_meeting_payload(path: Path, rng: random.Random) -> dict[str, Any]:
    transcript_text = path.read_text(encoding="utf-8", errors="replace")
    metadata = summarize_archived_transcript_text(transcript_text)
    original_filename = path.name
    meeting_id = derive_meeting_id_from_original_filename(original_filename)
    meeting_room = metadata["meeting_room"] or make_meeting_name(meeting_id)
    speaker_names = metadata["speaker_names"] or ["Archived Speaker"]
    participant_names = metadata["participants"] or speaker_names
    mock_user = select_mock_user(speaker_names, rng)
    room_participants = build_room_participants(participant_names)
    return {
        "meeting_id": meeting_id,
        "meeting_name": meeting_room,
        "meeting_room": meeting_room,
        "original_filename": original_filename,
        "speaker_names": speaker_names,
        "mock_user": mock_user,
        "room_participants": room_participants,
        "transcript_text": transcript_text,
        "utterance_count": int(metadata["utterance_count"]),
        "meeting_source": "archived",
        "archived_transcript_path": str(path),
    }


def choose_meeting_source(meeting_number: int, archived_paths: list[Path]) -> str:
    if MEETING_SOURCE_MODE == "synthetic":
        return "synthetic"
    if MEETING_SOURCE_MODE == "archived":
        return "archived"
    if MEETING_SOURCE_MODE == "mixed":
        if archived_paths and meeting_number % 2 == 0:
            return "archived"
        return "synthetic"
    raise ValueError(f"Unsupported MEETING_SOURCE_MODE: {MEETING_SOURCE_MODE}")


def build_meeting_payload(
    rng: random.Random,
    *,
    meeting_number: int,
    archived_paths: list[Path],
    archived_index: int,
) -> tuple[dict[str, Any], int]:
    source = choose_meeting_source(meeting_number, archived_paths)
    if source == "archived":
        if not archived_paths:
            raise RuntimeError("MEETING_SOURCE_MODE requires archived transcripts, but none were found")
        path = archived_paths[archived_index % len(archived_paths)]
        return build_archived_meeting_payload(path, rng), archived_index + 1
    return build_synthetic_meeting_payload(rng), archived_index


def post_transcript(payload: dict[str, Any]) -> dict[str, Any] | None:
    mock_user = payload["mock_user"]
    headers: dict[str, str] = {}
    if INGEST_TOKEN:
        headers["Authorization"] = f"Bearer {INGEST_TOKEN}"

    data = {
        "host_external_key": HOST_EXTERNAL_KEY,
        "host_user_id": mock_user["user_id"],
        "host_display_name": mock_user["display_name"],
        "host_email": mock_user["email"],
        "identity_source": mock_user["identity_source"],
        "meeting_room": payload["meeting_room"],
        "room_participants_json": json.dumps(payload["room_participants"]),
    }

    files = {
        "transcript": (
            payload["original_filename"],
            payload["transcript_text"].encode("utf-8"),
            "text/plain",
        )
    }
    try:
        response = requests.post(
            INGEST_URL,
            data=data,
            files=files,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        response_payload = response.json()
        structured_log(
            "info",
            "ingest_post_succeeded",
            **meeting_log_fields(payload),
            ingest_status=response_payload.get("ingest_status", "unknown"),
            saved_as=response_payload.get("saved_as"),
            stage1_build_mode=response_payload.get("stage1_build_mode"),
            stage1_build_status=response_payload.get("stage1_build_status"),
        )
        return response_payload
    except requests.exceptions.RequestException as exc:
        structured_log(
            "error",
            "ingest_post_failed",
            **meeting_log_fields(payload),
            error=str(exc),
        )
        return None


def run_batch(
    rng: random.Random,
    batch_size: int,
    *,
    start_meeting_number: int,
    archived_paths: list[Path],
    archived_index: int,
) -> tuple[int, int]:
    for index in range(batch_size):
        meeting_number = start_meeting_number + index
        payload, archived_index = build_meeting_payload(
            rng,
            meeting_number=meeting_number,
            archived_paths=archived_paths,
            archived_index=archived_index,
        )
        structured_log(
            "info",
            "meeting_generated",
            **meeting_log_fields(payload),
            meeting_number=meeting_number,
            batch_index=index + 1,
            batch_size=batch_size,
            utterance_count=payload["utterance_count"],
            speaker_names=payload["speaker_names"],
            archived_transcript_path=payload.get("archived_transcript_path"),
        )

        ingest_response = post_transcript(payload)
        if ingest_response is None:
            structured_log(
                "warning",
                "meeting_ingest_failed",
                **meeting_log_fields(payload),
                batch_index=index + 1,
            )

        if index < batch_size - 1:
            time.sleep(DELAY_SECONDS)

    structured_log(
        "info",
        "batch_complete",
        batch_size=batch_size,
    )
    return start_meeting_number + batch_size, archived_index


def main() -> None:
    archived_paths = list_archived_transcripts(ARCHIVED_TRANSCRIPT_ROOT)
    archived_count = len(archived_paths)

    if MEETING_SOURCE_MODE not in {"synthetic", "archived", "mixed"}:
        raise ValueError("MEETING_SOURCE_MODE must be one of: synthetic, archived, mixed")
    if MEETING_SOURCE_MODE == "archived" and not archived_paths:
        raise ValueError(
            f"MEETING_SOURCE_MODE=archived but no transcripts were found under {ARCHIVED_TRANSCRIPT_ROOT}"
        )

    structured_log(
        "info",
        "emulator_start",
        ingest_url=INGEST_URL,
        host_external_key=HOST_EXTERNAL_KEY,
        meeting_source_mode=MEETING_SOURCE_MODE,
        archived_transcript_root=str(ARCHIVED_TRANSCRIPT_ROOT),
        archived_transcript_count=archived_count,
        meeting_name_prefix=MEETING_NAME_PREFIX,
        meeting_count=MEETING_COUNT if MEETING_COUNT > 0 else "infinite",
        delay_seconds=DELAY_SECONDS,
        seed=SEED,
    )

    rng = random.Random(SEED)
    meeting_number = 1
    archived_index = 0
    if MEETING_COUNT > 0:
        meeting_number, archived_index = run_batch(
            rng,
            MEETING_COUNT,
            start_meeting_number=meeting_number,
            archived_paths=archived_paths,
            archived_index=archived_index,
        )
        return

    batch = 0
    while True:
        batch += 1
        structured_log("info", "continuous_batch_start", batch_number=batch)
        meeting_number, archived_index = run_batch(
            rng,
            5,
            start_meeting_number=meeting_number,
            archived_paths=archived_paths,
            archived_index=archived_index,
        )
        structured_log(
            "info",
            "continuous_batch_sleep",
            batch_number=batch,
            sleep_seconds=int(DELAY_SECONDS * 3),
        )
        time.sleep(DELAY_SECONDS * 3)


if __name__ == "__main__":
    main()
