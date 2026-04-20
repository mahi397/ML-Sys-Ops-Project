from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse

from auth.service import (
    authenticate_user,
    build_session_payload,
    logout_user,
    register_user,
    set_authenticated_user,
)
from core.config import APP_PREFIX

router = APIRouter(tags=["auth-api"])


@router.get(APP_PREFIX + "/api/session", response_model=None)
def session_status(request: Request) -> dict[str, Any]:
    room_name = request.query_params.get("room")
    return build_session_payload(request, room_name=room_name)


@router.post(APP_PREFIX + "/api/login", response_model=None)
def api_login(
    request: Request,
    login_name: str = Form(...),
    password: str = Form(...),
    room: str | None = Form(default=None),
) -> JSONResponse:
    user = authenticate_user(login_name, password)
    if not user:
        return JSONResponse({"ok": False, "error": "Invalid credentials."}, status_code=400)

    set_authenticated_user(request, user)
    payload = build_session_payload(request, user=user, room_name=room)
    payload["ok"] = True
    return JSONResponse(payload)


@router.post(APP_PREFIX + "/api/signup", response_model=None)
def api_signup(
    request: Request,
    user_id: str = Form(...),
    display_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    room: str | None = Form(default=None),
) -> JSONResponse:
    user, error = register_user(
        user_id=user_id,
        display_name=display_name,
        email=email,
        password=password,
    )
    if not user:
        return JSONResponse({"ok": False, "error": error}, status_code=400)

    set_authenticated_user(request, user)
    payload = build_session_payload(request, user=user, room_name=room)
    payload["ok"] = True
    return JSONResponse(payload)


@router.post(APP_PREFIX + "/api/logout", response_model=None)
def api_logout(request: Request, room: str | None = Form(default=None)) -> JSONResponse:
    logout_user(request)
    payload = build_session_payload(request, room_name=room)
    payload["ok"] = True
    return JSONResponse(payload)
