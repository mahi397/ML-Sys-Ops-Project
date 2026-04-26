from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from fastapi import Request

from auth import repository
from core.config import APP_PREFIX, MAX_RECENT_ROOMS, RECENT_ROOMS_PATH, ROOM_CONTEXT_DIR, get_public_url
from core.security import build_jitsi_token, decode_jitsi_token
from core.urls import (
    build_auth_redirect,
    build_host_launch_path,
    build_native_room_url,
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


def generate_internal_user_id(display_name: str) -> str:
    normalized_display_name = normalize_user_id(display_name)
    stem = normalized_display_name[:40].strip("_") or "user"
    return f"user_{stem}_{secrets.token_hex(4)}"


def register_user(
    display_name: str,
    email: str,
    password: str,
) -> tuple[dict[str, Any] | None, str | None]:
    normalized_display_name = display_name.strip()
    normalized_email = email.strip().lower()

    if (
        not normalized_display_name
        or "@" not in normalized_email
        or len(password) < 8
    ):
        return None, "Use a display name, valid email, and a password with at least 8 characters."

    for _ in range(5):
        generated_user_id = generate_internal_user_id(normalized_display_name)
        created, error = repository.create_user(
            user_id=generated_user_id,
            display_name=normalized_display_name,
            email=normalized_email,
            password=password,
        )
        if created:
            user = repository.fetch_user_by_id(generated_user_id)
            return user, None
        if error != "Generated user ID collision.":
            return None, error

    return None, "Could not generate a unique account ID. Please try again."


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


def build_room_auth_payload(request: Request, room_name: str) -> dict[str, Any]:
    room = sanitize_room_name(room_name)
    user = fetch_authenticated_user(request)
    if not user:
        return {
            "authenticated": False,
            "room_name": room,
        }

    touch_room_for_user(user, room, as_host=False)
    jwt_token = build_jitsi_token(user, room)
    return {
        "authenticated": True,
        "room_name": room,
        "room_url": build_native_room_url(room, jwt_token),
        "user_id": user["user_id"],
        "display_name": user["display_name"],
        "email": user["email"] or "",
    }


def write_room_context(room_name: str, user: dict[str, Any]) -> None:
    write_room_context_for_user(room_name, user, set_as_host=True)


def load_room_context(room_name: str) -> dict[str, Any]:
    room_context_path = ROOM_CONTEXT_DIR / f"{room_name}.json"
    if not room_context_path.exists():
        return {}

    try:
        payload = json.loads(room_context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    return payload if isinstance(payload, dict) else {}


def normalize_display_names(raw_display_names: Any, fallback_display_name: str = "") -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()

    def add_name(value: Any) -> None:
        text_value = str(value or "").strip()
        if not text_value:
            return

        key = text_value.casefold()
        if key in seen:
            return

        seen.add(key)
        normalized.append(text_value)

    if isinstance(raw_display_names, list):
        for display_name in raw_display_names:
            add_name(display_name)

    add_name(fallback_display_name)
    return normalized


def normalize_room_participants(raw_participants: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_participants, list):
        return []

    participants: list[dict[str, Any]] = []
    for raw_participant in raw_participants:
        if not isinstance(raw_participant, dict):
            continue

        user_id = str(raw_participant.get("user_id") or "").strip()
        display_name = str(raw_participant.get("display_name") or "").strip()
        display_names = normalize_display_names(
            raw_participant.get("display_names"),
            display_name,
        )
        email = str(raw_participant.get("email") or "").strip().lower()
        resolved_display_name = display_name or (display_names[-1] if display_names else "")
        if not user_id and not resolved_display_name and not email:
            continue

        participant: dict[str, Any] = {}
        if user_id:
            participant["user_id"] = user_id
        if resolved_display_name:
            participant["display_name"] = resolved_display_name
        if display_names:
            participant["display_names"] = display_names
        if email:
            participant["email"] = email

        identity_source = str(raw_participant.get("identity_source") or "").strip()
        if identity_source:
            participant["identity_source"] = identity_source

        written_at = str(
            raw_participant.get("written_at")
            or raw_participant.get("recorded_at")
            or ""
        ).strip()
        if written_at:
            participant["written_at"] = written_at

        participants.append(participant)

    return participants


def build_room_participant(
    user: dict[str, Any],
    written_at: str,
    *,
    current_display_name: str | None = None,
) -> dict[str, Any]:
    canonical_display_name = str(user.get("display_name") or user["user_id"]).strip()
    resolved_display_name = str(current_display_name or canonical_display_name).strip() or canonical_display_name
    display_names = normalize_display_names(
        [canonical_display_name, resolved_display_name],
        resolved_display_name,
    )
    participant = {
        "user_id": str(user["user_id"]).strip(),
        "display_name": resolved_display_name,
        "display_names": display_names,
        "identity_source": "meeting_portal",
        "written_at": written_at,
    }
    email = str(user.get("email") or "").strip().lower()
    if email:
        participant["email"] = email
    return participant


def merge_room_participants(
    participants: list[dict[str, Any]],
    current_participant: dict[str, Any],
) -> list[dict[str, Any]]:
    current_user_id = current_participant.get("user_id", "")
    current_email = current_participant.get("email", "")
    current_display_name = str(current_participant.get("display_name") or "").strip()
    current_display_names = normalize_display_names(
        current_participant.get("display_names"),
        current_display_name,
    )

    merged: list[dict[str, Any]] = []
    matched_existing = False
    for participant in participants:
        same_user_id = current_user_id and participant.get("user_id") == current_user_id
        same_email = current_email and participant.get("email") == current_email
        if not (same_user_id or same_email):
            merged.append(participant)
            continue

        merged_participant: dict[str, Any] = dict(participant)
        merged_display_names = normalize_display_names(
            list(participant.get("display_names") or []) + current_display_names,
            str(participant.get("display_name") or ""),
        )
        if merged_display_names:
            merged_participant["display_names"] = merged_display_names
            merged_participant["display_name"] = current_display_name or merged_display_names[-1]
        elif current_display_name:
            merged_participant["display_name"] = current_display_name

        if current_email:
            merged_participant["email"] = current_email

        current_identity_source = str(current_participant.get("identity_source") or "").strip()
        if current_identity_source:
            merged_participant["identity_source"] = current_identity_source

        current_written_at = str(current_participant.get("written_at") or "").strip()
        if current_written_at:
            merged_participant["written_at"] = current_written_at

        merged.append(merged_participant)
        matched_existing = True

    if not matched_existing:
        merged.append(current_participant)
    return merged


def write_room_context_for_user(
    room_name: str,
    user: dict[str, Any],
    *,
    set_as_host: bool,
    current_display_name: str | None = None,
) -> None:
    ROOM_CONTEXT_DIR.mkdir(parents=True, exist_ok=True)

    existing_payload = load_room_context(room_name)
    written_at = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = dict(existing_payload)
    payload["room_name"] = room_name
    payload["identity_source"] = "meeting_portal"
    payload["written_at"] = written_at

    existing_host_user_id = str(existing_payload.get("host_user_id") or "").strip()
    current_user_id = str(user["user_id"]).strip()
    resolved_display_name = str(current_display_name or user.get("display_name") or current_user_id).strip() or current_user_id
    if (not existing_host_user_id) or (existing_host_user_id == current_user_id and (set_as_host or resolved_display_name)):
        payload["host_user_id"] = user["user_id"]
        payload["host_display_name"] = resolved_display_name
        payload["host_email"] = user["email"]

    participants = normalize_room_participants(existing_payload.get("participants"))
    payload["participants"] = merge_room_participants(
        participants,
        build_room_participant(
            user,
            written_at,
            current_display_name=resolved_display_name,
        ),
    )

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


def touch_room_for_user(
    user: dict[str, Any],
    room_name: str,
    *,
    as_host: bool = False,
    current_display_name: str | None = None,
) -> str:
    room = sanitize_room_name(room_name)
    record_recent_room_for_user(user["user_id"], room)
    write_room_context_for_user(
        room,
        user,
        set_as_host=as_host,
        current_display_name=current_display_name,
    )
    return room


def record_room_presence_from_token(
    room_name: str,
    jwt_token: str,
    current_display_name: str | None = None,
) -> tuple[bool, str | None]:
    room = sanitize_room_name(room_name)
    payload = decode_jitsi_token(jwt_token)
    if not payload:
        return False, "Invalid Jitsi token."

    token_room = sanitize_room_name(str(payload.get("room") or ""))
    if token_room != room:
        return False, "Token room does not match the requested room."

    context = payload.get("context")
    if not isinstance(context, dict):
        return False, "Missing token context."

    token_user = context.get("user")
    if not isinstance(token_user, dict):
        return False, "Missing token user."

    user_id = str(token_user.get("id") or "").strip()
    if not user_id:
        return False, "Missing token user id."

    user = {
        "user_id": user_id,
        "display_name": str(token_user.get("name") or user_id).strip(),
        "email": str(token_user.get("email") or "").strip().lower(),
    }
    resolved_display_name = str(current_display_name or "").strip() or None
    touch_room_for_user(
        user,
        room,
        as_host=False,
        current_display_name=resolved_display_name,
    )
    return True, None
