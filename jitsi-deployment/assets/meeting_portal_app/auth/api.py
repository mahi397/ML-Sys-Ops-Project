from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Form, Request
from fastapi.responses import JSONResponse

from auth.service import (
    authenticate_user,
    build_room_auth_payload,
    build_session_payload,
    logout_user,
    record_room_presence_from_token,
    register_user,
    set_authenticated_user,
)
from core.config import APP_PREFIX

router = APIRouter(tags=["auth-api"])


@router.get(APP_PREFIX + "/api/session", response_model=None)
def session_status(request: Request) -> dict[str, Any]:
    room_name = request.query_params.get("room")
    return build_session_payload(request, room_name=room_name)


@router.get(APP_PREFIX + "/api/room-auth-url", response_model=None)
def room_auth_url(request: Request) -> JSONResponse:
    room_name = str(request.query_params.get("room") or "").strip()
    if not room_name:
        return JSONResponse({"ok": False, "error": "Missing room."}, status_code=400)

    payload = build_room_auth_payload(request, room_name)
    payload["ok"] = True
    return JSONResponse(payload)


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


@router.post(APP_PREFIX + "/api/room-context/presence", response_model=None)
def api_room_presence(payload: dict[str, Any] = Body(...)) -> JSONResponse:
    room_name = str(payload.get("room_name") or "").strip()
    jwt_token = str(payload.get("jwt") or "").strip()
    display_name = str(payload.get("display_name") or "").strip() or None
    if not room_name or not jwt_token:
        return JSONResponse({"ok": False, "error": "Missing room_name or jwt."}, status_code=400)

    recorded, error = record_room_presence_from_token(
        room_name,
        jwt_token,
        current_display_name=display_name,
    )
    if not recorded:
        return JSONResponse({"ok": False, "error": error or "Could not record room presence."}, status_code=400)

    return JSONResponse({"ok": True})
