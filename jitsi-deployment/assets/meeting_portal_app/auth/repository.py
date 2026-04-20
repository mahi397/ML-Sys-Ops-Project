from __future__ import annotations

from typing import Any

import psycopg

from core.db import get_conn
from core.security import hash_password


def fetch_user_by_id(user_id: str) -> dict[str, Any] | None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT user_id, display_name, email, is_active, created_at, updated_at
            FROM users
            WHERE user_id = %s
            """,
            (user_id,),
        )
        return cur.fetchone()


def fetch_user_for_login(login_name: str) -> dict[str, Any] | None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                user_id,
                display_name,
                email,
                password_salt,
                password_hash,
                is_active,
                created_at,
                updated_at
            FROM users
            WHERE user_id = %s OR email = %s
            """,
            (login_name, login_name),
        )
        return cur.fetchone()


def create_user(user_id: str, display_name: str, email: str, password: str) -> tuple[bool, str]:
    salt_hex, password_hash = hash_password(password)

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (
                    user_id,
                    display_name,
                    email,
                    password_salt,
                    password_hash,
                    is_active
                )
                VALUES (%s, %s, %s, %s, %s, TRUE)
                """,
                (user_id, display_name, email, salt_hex, password_hash),
            )
            conn.commit()
    except psycopg.Error as exc:
        if exc.sqlstate == "23505":
            return False, "That username or email is already in use."
        raise

    return True, ""
