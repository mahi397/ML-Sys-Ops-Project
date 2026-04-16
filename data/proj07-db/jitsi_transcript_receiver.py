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

SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


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

    logger.info(
        "Upload + ingest successful | client=%s | host_external_key=%s | original=%s | saved_as=%s | bytes=%s",
        client_host,
        raw_host_external_key,
        original_filename,
        safe_name,
        file_size,
    )

    return {
        "status": "ok",
        "host_external_key": raw_host_external_key,
        "saved_as": safe_name,
        "saved_path": str(save_path),
        "bytes": file_size,
        "ingest_status": "ok",
    }