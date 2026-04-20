from __future__ import annotations

import os
import re
from pathlib import Path

APP_NAME = "Meeting Portal"
APP_PREFIX = "/meeting-portal"
APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/data")
ROOM_CONTEXT_DIR = DATA_DIR / "room-contexts"
RECENT_ROOMS_PATH = DATA_DIR / "recent_rooms.json"
FEEDBACK_LOG_PATH = DATA_DIR / "recap-feedback.jsonl"

MAX_RECENT_ROOMS = 8
PASSWORD_ITERATIONS = 310_000
ROOM_RE = re.compile(r"[^a-z0-9_-]+")

def env(name: str, default: str | None = None) -> str:
    value = os.getenv(name)
    if value is not None:
        return value
    if default is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return default


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is not None:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is not None and raw_value.strip():
        return int(raw_value)
    return default


def get_public_url() -> str:
    return env("PUBLIC_URL", "https://localhost:8443").rstrip("/")


def get_db_dsn() -> str:
    return env("MEETING_PORTAL_DATABASE_URL")


def get_jwt_secret() -> str:
    return env("JWT_APP_SECRET")


def get_jwt_app_id() -> str:
    return env("JWT_APP_ID")


def get_jwt_audience() -> str:
    accepted = os.getenv("JWT_ACCEPTED_AUDIENCES", "").strip()
    if not accepted:
        return "jitsi"
    return accepted.split(",")[0].strip()


def get_xmpp_domain() -> str:
    return env("XMPP_DOMAIN", "meet.jitsi")


def get_rclone_remote() -> str:
    return env("MEETING_PORTAL_RCLONE_REMOTE", "rclone_s3").rstrip(":")


def get_rclone_bucket() -> str:
    return env("MEETING_PORTAL_RCLONE_BUCKET", "objstore-proj07").strip("/")


def get_rclone_timeout_seconds() -> int:
    return env_int("MEETING_PORTAL_RCLONE_TIMEOUT_SECONDS", 10)


def stage1_fallback_enabled() -> bool:
    return env_bool("MEETING_PORTAL_STAGE1_RCLONE_FALLBACK_ENABLED", True)


def get_session_secret() -> str:
    return env("MEETING_PORTAL_SESSION_SECRET")


def get_https_only() -> bool:
    return env_bool("MEETING_PORTAL_HTTPS_ONLY", True)


def get_token_ttl_seconds() -> int:
    return env_int("MEETING_PORTAL_TOKEN_TTL_SECONDS", 3600)


def build_rclone_object_uri(object_key: str) -> str:
    if re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*:.*$", object_key):
        return object_key
    return f"{get_rclone_remote()}:{get_rclone_bucket()}/{object_key.lstrip('/')}"
