from __future__ import annotations

import re
import secrets
from urllib.parse import quote, urlencode

from core.config import APP_PREFIX, ROOM_RE, get_public_url


def sanitize_room_name(raw_room_name: str | None) -> str:
    value = (raw_room_name or "").strip().lower()
    value = ROOM_RE.sub("-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-_")
    if value:
        return value[:64]
    return f"room-{secrets.token_hex(4)}"


def normalize_user_id(raw_user_id: str) -> str:
    value = (raw_user_id or "").strip().lower()
    value = ROOM_RE.sub("_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:64]


def safe_next_path(next_path: str | None, fallback: str) -> str:
    if not next_path:
        return fallback
    if not next_path.startswith("/") or next_path.startswith("//"):
        return fallback
    return next_path


def build_auth_landing_url(path: str, auth_mode: str | None = None) -> str:
    query: dict[str, str] = {}
    safe_path = safe_next_path(path, "/")
    if safe_path != "/":
        query["next"] = safe_path
    if auth_mode in {"login", "signup"}:
        query["auth"] = auth_mode
    if not query:
        return "/"
    return f"/?{urlencode(query)}"


def build_auth_redirect(path: str) -> str:
    return build_auth_landing_url(path, auth_mode="login")


def build_signup_redirect(path: str) -> str:
    return build_auth_landing_url(path, auth_mode="signup")


def build_host_launch_path(room_name: str | None = None) -> str:
    if room_name:
        return f"{APP_PREFIX}/host-launch/{quote(sanitize_room_name(room_name))}"
    return f"{APP_PREFIX}/host-launch"


def build_native_room_url(room_name: str, jwt_token: str | None = None) -> str:
    room = sanitize_room_name(room_name)
    base_url = f"{get_public_url()}/{quote(room)}"
    if not jwt_token:
        return base_url
    return f"{base_url}?{urlencode({'jwt': jwt_token})}"
