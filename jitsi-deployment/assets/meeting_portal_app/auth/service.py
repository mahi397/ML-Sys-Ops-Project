from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from fastapi import Request

from auth import repository
from core.config import APP_PREFIX, MAX_RECENT_ROOMS, RECENT_ROOMS_PATH, ROOM_CONTEXT_DIR, get_public_url
from core.urls import (
    build_auth_redirect,
    build_host_launch_path,
    build_signup_redirect,
    normalize_user_id,
    sanitize_room_name,
)


def fetch_authenticated_user(request: Request) -> dict[str, Any] | None:
    user_id = request.session.get("user_id")
    if not user_id:
        return None

    user = repository.fetch_user_by_id(user_id)
    if not user or not user.get("is_active"):
        request.session.clear()
        return None
    return user


def set_authenticated_user(request: Request, user: dict[str, Any]) -> None:
    request.session["user_id"] = user["user_id"]


def logout_user(request: Request) -> None:
    request.session.clear()


def authenticate_user(login_name: str, password: str) -> dict[str, Any] | None:
    from core.security import verify_password

    user = repository.fetch_user_for_login(login_name.strip())
    if not user or not user.get("is_active"):
        return None
    if not verify_password(password, user.get("password_salt"), user.get("password_hash")):
        return None
    return user


def register_user(
    user_id: str,
    display_name: str,
    email: str,
    password: str,
) -> tuple[dict[str, Any] | None, str | None]:
    normalized_user_id = normalize_user_id(user_id)
    normalized_display_name = display_name.strip()
    normalized_email = email.strip().lower()

    if (
        not normalized_user_id
        or not normalized_display_name
        or "@" not in normalized_email
        or len(password) < 8
    ):
        return None, "Use a username, display name, valid email, and a password with at least 8 characters."

    created, error = repository.create_user(
        user_id=normalized_user_id,
        display_name=normalized_display_name,
        email=normalized_email,
        password=password,
    )
    if not created:
        return None, error

    user = repository.fetch_user_by_id(normalized_user_id)
    return user, None


def build_session_payload(
    request: Request,
    user: dict[str, Any] | None = None,
    room_name: str | None = None,
) -> dict[str, Any]:
    resolved_room_name = sanitize_room_name(room_name) if room_name else None
    next_path = build_host_launch_path(resolved_room_name)

    if not user:
        user = fetch_authenticated_user(request)

    if not user:
        return {
            "authenticated": False,
            "login_url": build_auth_redirect(next_path),
            "signup_url": build_signup_redirect(next_path),
            "host_launch_url": build_host_launch_path(resolved_room_name),
            "guest_join_url": (
                f"{APP_PREFIX}/join/{quote(sanitize_room_name(resolved_room_name))}"
                if resolved_room_name
                else ""
            ),
        }

    return {
        "authenticated": True,
        "user_id": user["user_id"],
        "display_name": user["display_name"],
        "email": user["email"] or "",
        "login_url": build_auth_redirect(next_path),
        "signup_url": build_signup_redirect(next_path),
        "host_launch_url": build_host_launch_path(resolved_room_name),
        "guest_join_url": (
            f"{APP_PREFIX}/join/{quote(sanitize_room_name(resolved_room_name))}"
            if resolved_room_name
            else ""
        ),
    }


def write_room_context(room_name: str, user: dict[str, Any]) -> None:
    ROOM_CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "room_name": room_name,
        "host_user_id": user["user_id"],
        "host_display_name": user["display_name"],
        "host_email": user["email"],
        "identity_source": "meeting_portal",
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    (ROOM_CONTEXT_DIR / f"{room_name}.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def load_recent_rooms_store() -> dict[str, list[dict[str, str]]]:
    if not RECENT_ROOMS_PATH.exists():
        return {}

    try:
        payload = json.loads(RECENT_ROOMS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(payload, dict):
        return {}

    store: dict[str, list[dict[str, str]]] = {}
    for user_id, entries in payload.items():
        if not isinstance(user_id, str) or not isinstance(entries, list):
            continue

        clean_entries: list[dict[str, str]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue

            room_name = str(entry.get("room_name") or "").strip()
            if not room_name:
                continue

            clean_entries.append(
                {
                    "room_name": room_name,
                    "last_joined_at": str(entry.get("last_joined_at") or ""),
                }
            )

        if clean_entries:
            store[user_id] = clean_entries[:MAX_RECENT_ROOMS]
    return store


def save_recent_rooms_store(store: dict[str, list[dict[str, str]]]) -> None:
    RECENT_ROOMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECENT_ROOMS_PATH.write_text(
        json.dumps(store, indent=2) + "\n",
        encoding="utf-8",
    )


def record_recent_room_for_user(user_id: str, room_name: str) -> None:
    try:
        store = load_recent_rooms_store()
        recent_rooms = [
            entry
            for entry in store.get(user_id, [])
            if entry.get("room_name") != room_name
        ]
        recent_rooms.insert(
            0,
            {
                "room_name": room_name,
                "last_joined_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        store[user_id] = recent_rooms[:MAX_RECENT_ROOMS]
        save_recent_rooms_store(store)
    except OSError:
        return


def format_recent_room_time(raw_timestamp: str) -> str:
    if not raw_timestamp:
        return ""

    try:
        timestamp = datetime.fromisoformat(raw_timestamp)
    except ValueError:
        return ""

    return timestamp.astimezone().strftime("%b %d, %Y at %I:%M %p")


def list_recent_rooms_for_user(user_id: str) -> list[dict[str, str]]:
    recent_rooms = load_recent_rooms_store().get(user_id, [])
    items: list[dict[str, str]] = []

    for entry in recent_rooms:
        room_name = str(entry.get("room_name") or "").strip()
        if not room_name:
            continue

        encoded_room_name = quote(room_name)
        items.append(
            {
                "room_name": room_name,
                "last_joined_at": format_recent_room_time(str(entry.get("last_joined_at") or "")),
                "rejoin_url": f"{APP_PREFIX}/host-launch/{encoded_room_name}",
                "recap_url": f"{APP_PREFIX}/recaps/{encoded_room_name}",
                "raw_room_url": f"{get_public_url()}/{encoded_room_name}",
                "guest_join_url": f"{get_public_url()}{APP_PREFIX}/join/{encoded_room_name}",
            }
        )

    return items


def touch_room_for_user(user: dict[str, Any], room_name: str) -> str:
    room = sanitize_room_name(room_name)
    record_recent_room_for_user(user["user_id"], room)
    write_room_context(room, user)
    return room
