from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from auth.service import (
    fetch_authenticated_user,
    logout_user,
    touch_room_for_user,
)
from core.config import APP_PREFIX, get_public_url
from core.security import build_jitsi_token
from core.urls import (
    build_auth_redirect,
    build_host_launch_path,
    build_meet_path,
    build_native_room_url,
    sanitize_room_name,
)

router = APIRouter(tags=["auth-pages"])


def _launch_room_for_user(
    user: dict[str, Any],
    room_name: str | None,
    *,
    as_host: bool,
) -> RedirectResponse:
    room = touch_room_for_user(user, room_name or "", as_host=as_host)
    token = build_jitsi_token(user, room)
    return RedirectResponse(build_native_room_url(room, token), status_code=303)


@router.get(APP_PREFIX + "/", response_class=HTMLResponse, response_model=None)
def app_root_redirect() -> RedirectResponse:
    return RedirectResponse("/", status_code=303)


@router.get(APP_PREFIX + "/dashboard", response_class=HTMLResponse, response_model=None)
def dashboard_redirect() -> RedirectResponse:
    return RedirectResponse("/", status_code=303)


@router.get(APP_PREFIX + "/launch", response_model=None)
def launch_page() -> RedirectResponse:
    return RedirectResponse("/", status_code=303)


@router.get(APP_PREFIX + "/logout", response_model=None)
def logout(request: Request) -> RedirectResponse:
    logout_user(request)
    return RedirectResponse("/", status_code=303)


@router.post(APP_PREFIX + "/launch", response_model=None)
def launch(request: Request, room_name: str = Form(default="")) -> RedirectResponse:
    user = fetch_authenticated_user(request)
    room = sanitize_room_name(room_name)
    if not user:
        return RedirectResponse(build_auth_redirect(build_host_launch_path(room)), status_code=303)
    return _launch_room_for_user(user, room, as_host=True)


@router.get(APP_PREFIX + "/host-launch", response_model=None)
def host_launch_default(request: Request, room: str | None = None) -> RedirectResponse:
    user = fetch_authenticated_user(request)
    next_path = build_host_launch_path(room)
    if not user:
        return RedirectResponse(build_auth_redirect(next_path), status_code=303)
    return _launch_room_for_user(user, room, as_host=True)


@router.get(APP_PREFIX + "/host-launch/{room_name}", response_model=None)
def host_launch(request: Request, room_name: str) -> RedirectResponse:
    user = fetch_authenticated_user(request)
    next_path = build_host_launch_path(room_name)
    if not user:
        return RedirectResponse(build_auth_redirect(next_path), status_code=303)
    return _launch_room_for_user(user, room_name, as_host=True)


@router.get(APP_PREFIX + "/join/{room_name}", response_model=None)
def guest_join(room_name: str) -> RedirectResponse:
    room = sanitize_room_name(room_name)
    return RedirectResponse(f"{get_public_url()}/{room}", status_code=303)


@router.get(APP_PREFIX + "/meet/{room_name}", response_class=HTMLResponse, response_model=None)
def meet(request: Request, room_name: str) -> RedirectResponse:
    user = fetch_authenticated_user(request)
    room = sanitize_room_name(room_name)
    if not user:
        return RedirectResponse(build_auth_redirect(build_meet_path(room)), status_code=303)
    return _launch_room_for_user(user, room, as_host=False)
