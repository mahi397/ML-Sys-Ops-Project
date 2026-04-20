#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOM_RE = re.compile(r"[^a-zA-Z0-9_-]+")
MUC_ROOM_SUFFIX = "-muc-meet-jitsi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch local Jitsi transcript exports and upload completed transcript files to a VM ingest service."
    )
    parser.add_argument(
        "--transcript-root",
        type=Path,
        default=Path(os.getenv("JITSI_TRANSCRIPT_ROOT", "~/.jitsi-meet-cfg/transcripts")).expanduser(),
        help="Host path mounted into the transcriber container as /tmp/transcripts.",
    )
    parser.add_argument(
        "--ingest-url",
        default=os.getenv("JITSI_TRANSCRIPT_INGEST_URL"),
        help="Example: http://FLOATING_IP:9000/ingest/jitsi-transcript",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("INGEST_TOKEN", ""),
        help="Bearer token expected by the VM ingest service.",
    )
    parser.add_argument(
        "--host-external-key",
        default=os.getenv("JITSI_HOST_EXTERNAL_KEY", ""),
        help="Stable uploader identity for this host machine/container.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path(
            os.getenv(
                "JITSI_TRANSCRIPT_SENDER_STATE",
                "~/.jitsi-transcript-sender-state.json",
            )
        ).expanduser(),
    )
    parser.add_argument(
        "--room-context-root",
        type=Path,
        default=Path(
            os.getenv(
                "JITSI_ROOM_CONTEXT_ROOT",
                "~/.jitsi-meet-cfg/meeting-portal-app/room-contexts",
            )
        ).expanduser(),
        help="Directory of per-room host identity context files created by the Meeting Portal app.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=float(os.getenv("JITSI_TRANSCRIPT_POLL_SECONDS", "5")),
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=float(os.getenv("JITSI_TRANSCRIPT_SETTLE_SECONDS", "3")),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.getenv("JITSI_TRANSCRIPT_UPLOAD_TIMEOUT", "120")),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Scan once and exit instead of watching forever.",
    )
    return parser.parse_args()


def load_state(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    return set(payload.get("sent", []))


def save_state(path: Path, sent: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"sent": sorted(sent)}, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_room_name(room_name: str) -> str:
    cleaned = ROOM_RE.sub("-", room_name.strip()).strip("-_").lower()
    return cleaned[:80]


def find_transcripts(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("transcript_*.txt") if path.is_file())


def transcript_is_complete(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "End of transcript at " in text


def file_is_stable(path: Path, settle_seconds: float) -> bool:
    try:
        first = path.stat()
        time.sleep(settle_seconds)
        second = path.stat()
    except OSError:
        return False
    return (
        first.st_size == second.st_size
        and first.st_mtime_ns == second.st_mtime_ns
    )


def extract_room_name(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            first_line = handle.readline().strip()
    except OSError:
        return None

    marker = " in room "
    if marker not in first_line:
        return None

    room_name = first_line.split(marker, 1)[1].strip()
    normalized = normalize_room_name(room_name)
    return normalized or None


def load_room_context(room_context_root: Path, room_name: str | None) -> dict[str, str]:
    if not room_name:
        return {}

    candidate_paths: list[Path] = []
    normalized_room_name = room_name.strip()
    if normalized_room_name.endswith(MUC_ROOM_SUFFIX):
        base_room_name = normalized_room_name[:-len(MUC_ROOM_SUFFIX)]
        if base_room_name:
            candidate_paths.append(room_context_root / f"{base_room_name}.json")
    candidate_paths.append(room_context_root / f"{normalized_room_name}.json")

    room_context_path: Path | None = None
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            room_context_path = candidate_path
            break

    if room_context_path is None:
        return {}

    try:
        payload = json.loads(room_context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(payload, dict):
        return {}

    allowed_keys = (
        "host_user_id",
        "host_display_name",
        "host_email",
        "identity_source",
        "written_at",
    )
    context: dict[str, str] = {}
    for key in allowed_keys:
        value = payload.get(key)
        if value is None:
            continue
        text_value = str(value).strip()
        if text_value:
            context[key] = text_value

    participants = payload.get("participants")
    if isinstance(participants, list):
        normalized_participants = [
            participant
            for participant in participants
            if isinstance(participant, dict)
        ]
        if normalized_participants:
            context["room_participants_json"] = json.dumps(normalized_participants)
    return context


def multipart_body(
    file_field_name: str,
    file_path: Path,
    text_fields: dict[str, str],
) -> tuple[bytes, str]:
    boundary = f"----jitsi-transcript-{uuid.uuid4().hex}"
    filename = file_path.name
    body = bytearray()

    # Add text fields
    for field_name, field_value in text_fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n'
                f"{field_value}\r\n"
            ).encode("utf-8")
        )

    # Add transcript file
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        (
            f'Content-Disposition: form-data; name="{file_field_name}"; '
            f'filename="{filename}"\r\n'
            "Content-Type: text/plain\r\n\r\n"
        ).encode("utf-8")
    )
    body.extend(file_path.read_bytes())
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    return bytes(body), f"multipart/form-data; boundary={boundary}"


def upload_transcript(
    path: Path,
    ingest_url: str,
    token: str,
    timeout_seconds: float,
    host_external_key: str,
    room_context_root: Path,
) -> None:
    room_name = extract_room_name(path)
    text_fields = {
        "host_external_key": host_external_key,
    }
    if room_name:
        text_fields["meeting_room"] = room_name

    room_context = load_room_context(room_context_root, room_name)
    text_fields.update(room_context)

    body, content_type = multipart_body("transcript", path, text_fields)

    headers = {"Content-Type": content_type}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(ingest_url, data=body, headers=headers, method="POST")
    with urlopen(request, timeout=timeout_seconds) as response:
        response_text = response.read().decode("utf-8", errors="replace")
        participants_json = room_context.get("room_participants_json", "")
        print(
            f"Uploaded {path.name}: "
            f"room_name={room_name or ''} "
            f"host_external_key={host_external_key} "
            f"host_user_id={room_context.get('host_user_id', '')} "
            f"host_display_name={room_context.get('host_display_name', '')} "
            f"host_email={room_context.get('host_email', '')} "
            f"written_at={room_context.get('written_at', '')} "
            f"participants={participants_json} "
            f"HTTP {response.status} {response_text}"
        )


def process_ready_transcripts(args: argparse.Namespace, sent: set[str]) -> bool:
    uploaded_any = False

    for transcript_path in find_transcripts(args.transcript_root):
        resolved = str(transcript_path.resolve())

        if resolved in sent:
            continue
        if not transcript_is_complete(transcript_path):
            continue
        if not file_is_stable(transcript_path, args.settle_seconds):
            continue

        try:
            upload_transcript(
                transcript_path,
                args.ingest_url,
                args.token,
                args.timeout_seconds,
                args.host_external_key,
                args.room_context_root,
            )
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(
                f"Upload failed for {transcript_path.name}: "
                f"HTTP {exc.code} {detail}",
                file=sys.stderr,
            )
            continue
        except (OSError, URLError) as exc:
            print(
                f"Upload failed for {transcript_path.name}: {exc}",
                file=sys.stderr,
            )
            continue

        sent.add(resolved)
        save_state(args.state_file, sent)
        uploaded_any = True

    return uploaded_any


def main() -> None:
    args = parse_args()

    if not args.ingest_url:
        raise SystemExit("Missing --ingest-url or JITSI_TRANSCRIPT_INGEST_URL")

    if not args.host_external_key:
        raise SystemExit(
            "Missing --host-external-key or JITSI_HOST_EXTERNAL_KEY"
        )

    sent = load_state(args.state_file)

    print(f"Watching {args.transcript_root} for completed transcript_*.txt files")
    print(f"Uploading to {args.ingest_url}")
    print(f"Using host_external_key={args.host_external_key}")

    while True:
        process_ready_transcripts(args, sent)
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
