from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from core.config import (
    PASSWORD_ITERATIONS,
    get_jwt_app_id,
    get_jwt_audience,
    get_jwt_secret,
    get_token_ttl_seconds,
    get_xmpp_domain,
)


def hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    salt_hex = salt_hex or secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        PASSWORD_ITERATIONS,
    ).hex()
    return salt_hex, password_hash


def verify_password(password: str, salt_hex: str | None, password_hash: str | None) -> bool:
    if not salt_hex or not password_hash:
        return False

    _, candidate_hash = hash_password(password, salt_hex)
    return hmac.compare_digest(candidate_hash, password_hash)


def build_jitsi_token(user: dict[str, Any], room_name: str) -> str:
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=get_token_ttl_seconds())
    payload = {
        "aud": get_jwt_audience(),
        "iss": get_jwt_app_id(),
        "sub": get_xmpp_domain(),
        "room": room_name,
        "iat": int(now.timestamp()),
        "nbf": int((now - timedelta(seconds=5)).timestamp()),
        "exp": int(expires_at.timestamp()),
        "context": {
            "user": {
                "id": user["user_id"],
                "name": user["display_name"],
                "email": user["email"] or "",
                "moderator": True,
            },
            "features": {
                "livestreaming": False,
                "recording": True,
                "transcription": True,
            },
        },
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm="HS256")
