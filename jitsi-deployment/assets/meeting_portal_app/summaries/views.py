from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from auth.service import fetch_authenticated_user
from core.config import APP_PREFIX
from core.templates import render_template
from core.urls import build_auth_redirect
from summaries.service import fetch_recaps_for_user

router = APIRouter(tags=["summary-pages"])


@router.get(APP_PREFIX + "/recaps", response_class=HTMLResponse, response_model=None)
def recaps_page(request: Request) -> HTMLResponse | RedirectResponse:
    user = fetch_authenticated_user(request)
    next_path = APP_PREFIX + "/recaps"
    if not user:
        return RedirectResponse(build_auth_redirect(next_path), status_code=303)

    recaps = fetch_recaps_for_user(user["user_id"])
    return render_template(
        request,
        "recaps.html",
        user,
        title="Meeting Recaps",
        recaps=recaps,
    )


@router.get(APP_PREFIX + "/recaps/{meeting_id}", response_class=HTMLResponse, response_model=None)
def recap_detail_page(request: Request, meeting_id: str) -> HTMLResponse | RedirectResponse:
    user = fetch_authenticated_user(request)
    next_path = f"{APP_PREFIX}/recaps/{quote(meeting_id)}"
    if not user:
        return RedirectResponse(build_auth_redirect(next_path), status_code=303)

    return render_template(
        request,
        "meeting_recap.html",
        user,
        title=f"Recap · {meeting_id}",
        meeting_id=meeting_id,
    )
