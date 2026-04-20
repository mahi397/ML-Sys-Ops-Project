from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from auth.service import fetch_authenticated_user
from core.config import APP_PREFIX
from summaries.service import (
    append_feedback_event,
    append_summary_edit_events,
    can_access_recap,
    can_edit_summary,
    fetch_recap_for_user,
    fetch_recaps_for_user,
)

router = APIRouter(tags=["summary-api"])


@router.get(APP_PREFIX + "/api/recaps", response_model=None)
def api_recaps(request: Request) -> JSONResponse:
    user = fetch_authenticated_user(request)
    if not user:
        return JSONResponse({"ok": False, "error": "Unauthorized"}, status_code=401)

    items = fetch_recaps_for_user(user["user_id"])
    return JSONResponse({"ok": True, "items": items})


@router.get(APP_PREFIX + "/api/recap/{meeting_id}", response_model=None)
def api_recap_detail(
    request: Request,
    meeting_id: str,
    source: str | None = None,
) -> JSONResponse:
    user = fetch_authenticated_user(request)
    if not user:
        return JSONResponse({"ok": False, "error": "Unauthorized"}, status_code=401)

    recap = fetch_recap_for_user(user["user_id"], meeting_id, summary_source=source)
    if recap is None:
        return JSONResponse(
            {"ok": False, "message": "No recap summary found."},
            status_code=404,
        )

    return JSONResponse({"ok": True, "recap": recap})


@router.post(APP_PREFIX + "/api/feedback", response_model=None)
def api_feedback(request: Request, payload: dict[str, Any]) -> JSONResponse:
    user = fetch_authenticated_user(request)
    if not user:
        return JSONResponse({"ok": False, "error": "Unauthorized"}, status_code=401)

    meeting_id = str(payload.get("meeting_id") or "").strip()
    if meeting_id and not can_access_recap(user["user_id"], meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")

    append_feedback_event(user["user_id"], payload)
    return JSONResponse({"ok": True})


@router.post(APP_PREFIX + "/api/summary-edits", response_model=None)
def api_summary_edits(request: Request, payload: dict[str, Any]) -> JSONResponse:
    user = fetch_authenticated_user(request)
    if not user:
        return JSONResponse({"ok": False, "error": "Unauthorized"}, status_code=401)

    meeting_id = str(payload.get("meeting_id") or "").strip()
    if not meeting_id:
        raise HTTPException(status_code=400, detail="meeting_id required")
    if not can_access_recap(user["user_id"], meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")
    if not can_edit_summary(user["user_id"], meeting_id):
        raise HTTPException(status_code=403, detail="You do not have permission to edit this summary")

    result = append_summary_edit_events(user["user_id"], payload)
    return JSONResponse({"ok": True, **result})
