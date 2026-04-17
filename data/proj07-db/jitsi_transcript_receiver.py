#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile

APP_NAME = "jitsi_transcript_receiver"

INGEST_TOKEN = os.getenv("INGEST_TOKEN", "").strip()
SAVE_DIR = Path(
    os.getenv("JITSI_SAVE_DIR", "/mnt/block/user-behaviour/received_transcripts")
)
LOG_DIR = Path(
    os.getenv("JITSI_LOG_DIR", "/mnt/block/ingest_logs/jitsi_transcripts")
)
LOG_FILE = LOG_DIR / f"{APP_NAME}_logs.txt"

SCRIPT_DIR = Path(__file__).resolve().parent
INGEST_SCRIPT = Path(
    os.getenv(
        "JITSI_INGEST_SCRIPT",
        str(SCRIPT_DIR / "ingest_saved_jitsi_transcript.py"),
    )
)
INGEST_TIMEOUT_SECONDS = int(os.getenv("JITSI_INGEST_TIMEOUT_SECONDS", "300"))


def env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


BUILD_STAGE1_AFTER_INGEST = env_flag("JITSI_BUILD_STAGE1_AFTER_INGEST", False)
STAGE1_OUTPUT_ROOT = Path(
    os.getenv("JITSI_STAGE1_OUTPUT_ROOT", "/mnt/block/user-behaviour/inference_requests/stage1")
)
STAGE1_WINDOW_SIZE = int(os.getenv("JITSI_STAGE1_WINDOW_SIZE", "7"))
STAGE1_TRANSITION_INDEX = int(os.getenv("JITSI_STAGE1_TRANSITION_INDEX", "3"))
STAGE1_MIN_UTTERANCE_CHARS = int(os.getenv("JITSI_STAGE1_MIN_UTTERANCE_CHARS", "20"))
STAGE1_MAX_WORDS_PER_UTTERANCE = int(os.getenv("JITSI_STAGE1_MAX_WORDS_PER_UTTERANCE", "50"))
STAGE1_MIN_INFERENCE_UTTERANCES = int(
    os.getenv("JITSI_STAGE1_MIN_INFERENCE_UTTERANCES", "2")
)
STAGE1_SHORT_MEETING_MAX_UTTERANCES = int(
    os.getenv("JITSI_STAGE1_SHORT_MEETING_MAX_UTTERANCES", "6")
)
UPLOAD_STAGE1_ARTIFACTS = env_flag("JITSI_UPLOAD_STAGE1_ARTIFACTS", True)
STAGE1_OBJECT_PREFIX = os.getenv(
    "JITSI_STAGE1_OBJECT_PREFIX",
    "production/inference_requests/stage1",
).strip()

SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
if BUILD_STAGE1_AFTER_INGEST:
    STAGE1_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def build_logger() -> logging.Logger:
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        LOG_FILE,
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


def sanitize_host_key(value: str) -> str:
    """
    Keep only safe filename/path characters for local storage paths.
    Do not use this sanitized value as the DB identity.
    """
    cleaned = value.strip()
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", cleaned)
    return cleaned


def parse_ingester_summary(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def run_ingester(
    *,
    transcript_path: Path,
    original_filename: str,
    host_external_key: str,
    metadata_path: Path | None,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(INGEST_SCRIPT),
        "--transcript-path",
        str(transcript_path),
        "--original-filename",
        original_filename,
        "--host-external-key",
        host_external_key,
    ]

    if metadata_path is not None:
        cmd.extend(["--metadata-path", str(metadata_path)])

    if BUILD_STAGE1_AFTER_INGEST:
        cmd.extend(
            [
                "--build-stage1-after-ingest",
                "--stage1-output-root",
                str(STAGE1_OUTPUT_ROOT),
                "--stage1-window-size",
                str(STAGE1_WINDOW_SIZE),
                "--stage1-transition-index",
                str(STAGE1_TRANSITION_INDEX),
                "--stage1-min-utterance-chars",
                str(STAGE1_MIN_UTTERANCE_CHARS),
                "--stage1-max-words-per-utterance",
                str(STAGE1_MAX_WORDS_PER_UTTERANCE),
                "--stage1-min-inference-utterances",
                str(STAGE1_MIN_INFERENCE_UTTERANCES),
                "--stage1-short-meeting-max-utterances",
                str(STAGE1_SHORT_MEETING_MAX_UTTERANCES),
            ]
        )
        if UPLOAD_STAGE1_ARTIFACTS:
            cmd.extend(["--upload-stage1-artifacts", "--stage1-object-prefix", STAGE1_OBJECT_PREFIX])

    logger.info("Running ingester: %s", " ".join(cmd))

    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=INGEST_TIMEOUT_SECONDS,
    )


logger = build_logger()
app = FastAPI()


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Starting %s", APP_NAME)
    logger.info("Transcript save directory: %s", SAVE_DIR.resolve())
    logger.info("Log file: %s", LOG_FILE.resolve())
    logger.info("Bearer token enabled: %s", bool(INGEST_TOKEN))
    logger.info("Ingester script: %s", INGEST_SCRIPT.resolve())
    logger.info(
        "Stage 1 build mode: %s",
        "sync_ingest" if BUILD_STAGE1_AFTER_INGEST else "async_db_worker",
    )
    logger.info("Build Stage 1 after ingest: %s", BUILD_STAGE1_AFTER_INGEST)
    if BUILD_STAGE1_AFTER_INGEST and not (0 <= STAGE1_TRANSITION_INDEX < STAGE1_WINDOW_SIZE):
        raise RuntimeError(
            "Invalid Stage 1 configuration: transition index must be between 0 and window size - 1"
        )
    if BUILD_STAGE1_AFTER_INGEST:
        logger.info("Stage 1 output root: %s", STAGE1_OUTPUT_ROOT.resolve())
        logger.info(
            "Stage 1 config | window_size=%s transition_index=%s min_chars=%s max_words=%s upload=%s prefix=%s",
            STAGE1_WINDOW_SIZE,
            STAGE1_TRANSITION_INDEX,
            STAGE1_MIN_UTTERANCE_CHARS,
            STAGE1_MAX_WORDS_PER_UTTERANCE,
            UPLOAD_STAGE1_ARTIFACTS,
            STAGE1_OBJECT_PREFIX,
        )
        logger.info(
            "Stage 1 gating | min_inference_utterances=%s short_meeting_max_utterances=%s",
            STAGE1_MIN_INFERENCE_UTTERANCES,
            STAGE1_SHORT_MEETING_MAX_UTTERANCES,
        )

    if not INGEST_SCRIPT.exists():
        raise RuntimeError(f"Ingest script not found: {INGEST_SCRIPT}")


@app.get("/health")
async def health() -> dict:
    logger.info("Health check requested")
    return {"status": "ok", "service": APP_NAME}


@app.post("/ingest/jitsi-transcript")
async def ingest_jitsi_transcript(
    request: Request,
    transcript: UploadFile = File(...),
    host_external_key: str = Form(...),
    authorization: str | None = Header(default=None),
):
    client_host = request.client.host if request.client else "unknown"
    logger.info("Received upload request from %s", client_host)

    if INGEST_TOKEN:
        expected = f"Bearer {INGEST_TOKEN}"
        if authorization != expected:
            logger.warning("Unauthorized request from %s", client_host)
            raise HTTPException(status_code=401, detail="Unauthorized")

    raw_host_external_key = host_external_key.strip()
    safe_host_dir_key = sanitize_host_key(raw_host_external_key)

    if not raw_host_external_key:
        logger.warning("Request from %s missing host_external_key", client_host)
        raise HTTPException(status_code=400, detail="Missing host_external_key")

    if not safe_host_dir_key:
        logger.warning("Request from %s has invalid host_external_key", client_host)
        raise HTTPException(status_code=400, detail="Invalid host_external_key")

    if not transcript.filename:
        logger.warning("Request from %s missing filename", client_host)
        raise HTTPException(status_code=400, detail="Missing filename")

    original_filename = Path(transcript.filename).name
    if not original_filename.endswith(".txt"):
        logger.warning(
            "Rejected non-txt file from %s: %s",
            client_host,
            original_filename,
        )
        raise HTTPException(status_code=400, detail="Only .txt transcripts allowed")

    # Save under a host-specific directory so uploads from the same host stay grouped.
    # Use sanitized host key only for filesystem safety.
    host_dir = SAVE_DIR / safe_host_dir_key
    host_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{uuid.uuid4().hex}_{original_filename}"
    save_path = host_dir / safe_name

    logger.info(
        "Saving uploaded transcript | host_external_key=%s | path=%s",
        raw_host_external_key,
        save_path,
    )

    try:
        with save_path.open("wb") as f:
            shutil.copyfileobj(transcript.file, f)
    except Exception as exc:
        logger.exception("Failed saving file %s: %s", original_filename, exc)
        raise HTTPException(status_code=500, detail="Failed to save transcript")
    finally:
        transcript.file.close()

    try:
        text = save_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.exception("Failed reading saved transcript %s: %s", save_path, exc)
        raise HTTPException(status_code=500, detail="Failed to read saved transcript")

    if "End of transcript at " not in text:
        logger.warning(
            "Transcript appears incomplete | host_external_key=%s | file=%s",
            raw_host_external_key,
            save_path.name,
        )
        try:
            save_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed deleting incomplete transcript %s", save_path)
        raise HTTPException(
            status_code=400,
            detail="Transcript does not appear complete",
        )

    file_size = save_path.stat().st_size

    metadata_path = save_path.with_suffix(save_path.suffix + ".meta.json")
    metadata = {
        "host_external_key": raw_host_external_key,
        "client_host": client_host,
        "original_filename": original_filename,
        "saved_as": safe_name,
        "saved_path": str(save_path),
        "bytes": file_size,
    }

    try:
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        logger.exception("Failed writing metadata sidecar for %s", save_path.name)
        metadata_path = None

    try:
        result = run_ingester(
            transcript_path=save_path,
            original_filename=original_filename,
            host_external_key=raw_host_external_key,
            metadata_path=metadata_path,
        )
    except subprocess.TimeoutExpired:
        logger.exception("Ingester timed out for %s", save_path)
        raise HTTPException(
            status_code=500,
            detail="Transcript saved but ingest timed out",
        )
    except Exception as exc:
        logger.exception("Failed launching ingester for %s: %s", save_path, exc)
        raise HTTPException(
            status_code=500,
            detail="Transcript saved but ingest launch failed",
        )

    if result.returncode != 0:
        logger.error("Ingester failed for %s", save_path)

        if result.stdout:
            logger.error("Ingester stdout:\n%s", result.stdout.strip())
        if result.stderr:
            logger.error("Ingester stderr:\n%s", result.stderr.strip())

        raise HTTPException(
            status_code=500,
            detail="Transcript saved but DB ingest failed",
        )

    if result.stdout:
        logger.info("Ingester stdout:\n%s", result.stdout.strip())
    if result.stderr:
        logger.info("Ingester stderr:\n%s", result.stderr.strip())

    ingester_summary = parse_ingester_summary(result.stdout or "")
    if BUILD_STAGE1_AFTER_INGEST:
        stage1_build_status = ingester_summary.get("stage1_build_status", "unknown")
        stage1_build_error = ingester_summary.get("stage1_build_error")
    else:
        stage1_build_status = "deferred_to_db_worker"
        stage1_build_error = None

    logger.info(
        "Upload + ingest successful | client=%s | host_external_key=%s | original=%s | saved_as=%s | bytes=%s | stage1=%s",
        client_host,
        raw_host_external_key,
        original_filename,
        safe_name,
        file_size,
        stage1_build_status,
    )

    return {
        "status": "ok",
        "host_external_key": raw_host_external_key,
        "saved_as": safe_name,
        "saved_path": str(save_path),
        "bytes": file_size,
        "ingest_status": ingester_summary.get("status", "ok"),
        "stage1_build_enabled": BUILD_STAGE1_AFTER_INGEST,
        "stage1_build_mode": "sync_ingest" if BUILD_STAGE1_AFTER_INGEST else "async_db_worker",
        "stage1_build_status": stage1_build_status,
        "stage1_build_error": stage1_build_error,
    }
