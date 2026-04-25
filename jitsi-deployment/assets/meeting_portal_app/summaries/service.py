from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from fastapi import HTTPException

from core.config import APP_PREFIX
from summaries import repository


def format_display_datetime(value: datetime | str | None) -> str:
    if not value:
        return ""

    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value

    return value.astimezone().strftime("%b %d, %Y at %I:%M %p")


def format_datetime_iso(value: datetime | str | None) -> str:
    if not value:
        return ""

    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return ""

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def format_duration_label(total_seconds: float | int | None) -> str:
    if total_seconds is None:
        return ""

    seconds = max(int(round(float(total_seconds))), 0)
    if seconds <= 0:
        return ""

    hours, remainder = divmod(seconds, 3600)
    minutes, leftover_seconds = divmod(remainder, 60)

    if hours:
        return f"{hours} hr {minutes} min" if minutes else f"{hours} hr"
    if minutes:
        return f"{minutes} min"
    return f"{leftover_seconds} sec"


def normalize_summary_bullets(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        return normalize_summary_bullets(parsed)

    if isinstance(value, dict):
        return normalize_summary_bullets(list(value.values()))

    if isinstance(value, list):
        bullets: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = " ".join(
                    str(part).strip()
                    for part in item.values()
                    if str(part).strip()
                ).strip()
            else:
                text = str(item).strip()

            if text:
                bullets.append(text)
        return bullets

    text = str(value).strip()
    return [text] if text else []


def summarize_bullets(value: Any) -> str:
    bullets = normalize_summary_bullets(value)
    return ". ".join(bullets) if bullets else "No summary available."


def normalize_summary_source(value: str | None) -> str | None:
    normalized = (value or "").strip().lower()
    if normalized in {"llm", "llm_generated"}:
        return "llm_generated"
    if normalized in {"user", "edited", "user_edited"}:
        return "user_edited"
    return None


def summary_source_label(summary_type: str) -> str:
    if summary_type == "user_edited":
        return "User edited"
    return "LLM generated"


def can_edit_summary(user_id: str, meeting_id: str) -> bool:
    return repository.user_can_edit_summary(user_id, meeting_id)


def can_access_recap(user_id: str, meeting_id: str) -> bool:
    return repository.user_can_access_recap(user_id, meeting_id)


def fetch_recaps_for_user(user_id: str) -> list[dict[str, Any]]:
    rows = repository.fetch_recap_rows_for_user(user_id)
    items: list[dict[str, Any]] = []

    for row in rows:
        meeting_id = str(row["meeting_id"])
        items.append(
            {
                "meeting_id": meeting_id,
                "meeting_title": row["meeting_title"],
                "role": row["role"],
                "segment_count": int(row["segment_count"] or 0),
                "summary_id": row["summary_id"],
                "summary_version": row["summary_version"],
                "recap_status": "complete" if row["is_complete"] else "draft",
                "started_at": format_display_datetime(row.get("started_at")),
                "started_at_iso": format_datetime_iso(row.get("started_at")),
                "ended_at": format_display_datetime(row.get("ended_at")),
                "ended_at_iso": format_datetime_iso(row.get("ended_at")),
                "summary_created_at": format_display_datetime(row.get("summary_created_at")),
                "summary_created_at_iso": format_datetime_iso(row.get("summary_created_at")),
                "summary_updated_at": format_display_datetime(row.get("summary_updated_at")),
                "summary_updated_at_iso": format_datetime_iso(row.get("summary_updated_at")),
                "model_version": row.get("model_version") or "",
                "has_recap": True,
                "recap_url": f"{APP_PREFIX}/recaps/{quote(meeting_id)}",
            }
        )

    return items


def select_summary_variant(
    variants: list[dict[str, Any]],
    requested_source: str | None = None,
) -> dict[str, Any] | None:
    if not variants:
        return None

    normalized_source = normalize_summary_source(requested_source)
    if normalized_source:
        for variant in variants:
            if variant.get("summary_type") == normalized_source:
                return variant

    for preferred_source in ("user_edited", "llm_generated"):
        for variant in variants:
            if variant.get("summary_type") == preferred_source:
                return variant

    return variants[0]


def fetch_recap_for_user(
    user_id: str,
    meeting_id: str,
    summary_source: str | None = None,
) -> dict[str, Any] | None:
    variants = repository.fetch_summary_variants_for_user(user_id, meeting_id)
    meta = select_summary_variant(variants, requested_source=summary_source)
    if not meta:
        return None

    segments = repository.fetch_summary_segments(int(meta["summary_id"]))
    if not segments:
        return None

    utterances = repository.fetch_meeting_utterances(meeting_id)
    last_utterance_index = utterances[-1]["utterance_index"] if utterances else 0
    boundary_by_utterance = {
        int(utterance["utterance_index"]): utterance.get("boundary_confidence")
        for utterance in utterances
        if utterance.get("utterance_index") is not None
    }

    ui_segments: list[dict[str, Any]] = []
    for display_index, segment in enumerate(segments):
        start_utterance_index = segment.get("start_utterance_index")
        if start_utterance_index is None:
            start_utterance_index = 0 if display_index == 0 else ui_segments[-1]["end_utt"] + 1

        end_utterance_index = segment.get("end_utterance_index")
        if end_utterance_index is None:
            end_utterance_index = last_utterance_index

        boundary_confidence = boundary_by_utterance.get(int(end_utterance_index))
        ui_segments.append(
            {
                "segment_idx": display_index,
                "segment_summary_id": segment["segment_summary_id"],
                "start_utt": int(start_utterance_index),
                "end_utt": int(end_utterance_index),
                "t_start": float(segment.get("t_start") or 0),
                "t_end": float(segment.get("t_end") or 0),
                "topic_label": segment.get("topic_label") or "",
                "summary_bullets": normalize_summary_bullets(segment.get("summary_bullets")),
                "summary": summarize_bullets(segment.get("summary_bullets")),
                "status": segment.get("status") or "complete",
                "boundary_confidence": (
                    float(boundary_confidence)
                    if boundary_confidence is not None
                    else None
                ),
            }
        )

    available_sources: list[dict[str, Any]] = []
    for variant in variants:
        available_sources.append(
            {
                "summary_id": int(variant["summary_id"]),
                "summary_type": variant["summary_type"],
                "summary_type_label": summary_source_label(str(variant["summary_type"])),
                "summary_version": int(variant["summary_version"] or 0),
                "summary_created_at": format_display_datetime(variant.get("summary_created_at")),
                "summary_created_at_iso": format_datetime_iso(variant.get("summary_created_at")),
                "summary_updated_at": format_display_datetime(variant.get("summary_updated_at")),
                "summary_updated_at_iso": format_datetime_iso(variant.get("summary_updated_at")),
                "segment_count": int(variant.get("segment_count") or 0),
                "is_complete": bool(variant.get("is_complete")),
                "model_version": variant.get("model_version") or "",
                "created_by_user_id": variant.get("created_by_user_id"),
                "is_selected": int(variant["summary_id"]) == int(meta["summary_id"]),
            }
        )

    ui_utterances = [
        {
            "utterance_id": int(utterance["utterance_id"]),
            "utterance_idx": int(utterance["utterance_index"]),
            "speaker": utterance.get("speaker") or "Speaker",
            "text": utterance.get("text") or "",
            "t_start": float(utterance.get("start_time_sec") or 0),
            "t_end": float(utterance.get("end_time_sec") or 0),
        }
        for utterance in utterances
    ]

    duration_seconds: float | int | None = None
    if meta.get("started_at") and meta.get("ended_at"):
        duration_seconds = (meta["ended_at"] - meta["started_at"]).total_seconds()
    elif utterances:
        first_start = min(float(utterance["t_start"]) for utterance in ui_utterances)
        last_end = max(float(utterance["t_end"]) for utterance in ui_utterances)
        duration_seconds = max(last_end - first_start, 0)

    model_version = meta.get("model_version") or ""
    if not model_version:
        model_version = next(
            (
                segment.get("model_version") or segment.get("model_name") or ""
                for segment in segments
                if segment.get("model_version") or segment.get("model_name")
            ),
            "",
        )

    return {
        "summary_id": meta["summary_id"],
        "summary_type": meta["summary_type"],
        "summary_type_label": summary_source_label(str(meta["summary_type"])),
        "meeting_id": meta["meeting_id"],
        "meeting_title": meta["meeting_title"],
        "meeting_duration": format_duration_label(duration_seconds),
        "participant_count": int(meta.get("participant_count") or 0),
        "model_version": model_version or "llm_generated",
        "status": "complete" if all(segment.get("status") == "complete" for segment in segments) else "draft",
        "message": "",
        "can_edit_summary": bool(meta.get("can_edit_summary")),
        "available_sources": available_sources,
        "segments": ui_segments,
        "utterances": ui_utterances,
    }


def append_feedback_event(user_id: str, payload: dict[str, Any]) -> None:
    meeting_id = str(payload.get("meeting_id") or "").strip()
    action = str(payload.get("action") or "").strip()
    if not meeting_id:
        raise HTTPException(status_code=400, detail="meeting_id required")
    if not action:
        raise HTTPException(status_code=400, detail="action required")

    segment_summary_id_raw = payload.get("segment_summary_id")
    try:
        segment_summary_id = int(segment_summary_id_raw) if segment_summary_id_raw is not None else None
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="segment_summary_id must be an integer") from exc

    utterance_idx_raw = payload.get("utterance_idx")
    utterance_idx: int | None = None
    if utterance_idx_raw is not None:
        try:
            utterance_idx = int(utterance_idx_raw)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="utterance_idx must be an integer") from exc

    summary_id_raw = payload.get("summary_id")
    summary_id: int | None = None
    if summary_id_raw is not None:
        try:
            summary_id = int(summary_id_raw)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="summary_id must be an integer") from exc

    summary_source = normalize_summary_source(str(payload.get("summary_source") or ""))

    positive_actions = {"overall_good", "overall_positive"}
    negative_actions = {"overall_bad", "overall_negative", "overall_needs_work"}

    if action in positive_actions:
        event_type = "accept_summary"
        before_payload: dict[str, Any] = {}
        after_payload: dict[str, Any] = {"rating": action.replace("overall_", "")}
    elif action in negative_actions:
        event_type = "boundary_correction"
        before_payload = {}
        after_payload = {"rating": action.replace("overall_", "")}
    elif action == "remove_boundary":
        if utterance_idx is None:
            raise HTTPException(status_code=400, detail="utterance_idx required")
        event_type = "merge_segments"
        before_payload = {"utterance_idx": utterance_idx, "is_boundary": True}
        after_payload = {"utterance_idx": utterance_idx, "is_boundary": False}
    elif action == "add_boundary":
        if utterance_idx is None:
            raise HTTPException(status_code=400, detail="utterance_idx required")
        event_type = "split_segment"
        before_payload = {"utterance_idx": utterance_idx, "is_boundary": False}
        after_payload = {"utterance_idx": utterance_idx, "is_boundary": True}
    else:
        raise HTTPException(status_code=400, detail="Unsupported feedback action")

    if summary_id is None:
        summary_id = repository.fetch_latest_summary_id_for_meeting(
            meeting_id,
            summary_type=summary_source or "llm_generated",
        )

    if segment_summary_id is not None:
        segment_context = repository.fetch_segment_feedback_context(meeting_id, segment_summary_id)
        if not segment_context:
            raise HTTPException(status_code=404, detail="Segment summary not found")

        summary_id = int(segment_context["summary_id"])
        before_payload["segment"] = {
            "segment_summary_id": segment_context["segment_summary_id"],
            "segment_index": segment_context["segment_index"],
            "topic_label": segment_context.get("topic_label") or "",
            "summary_bullets": normalize_summary_bullets(segment_context.get("summary_bullets")),
            "start_utterance_index": segment_context.get("start_utterance_index"),
            "end_utterance_index": segment_context.get("end_utterance_index"),
        }

    repository.insert_feedback_event(
        meeting_id=meeting_id,
        summary_id=summary_id,
        segment_summary_id=segment_summary_id,
        event_type=event_type,
        before_payload=before_payload,
        after_payload=after_payload,
        user_id=user_id,
    )


def append_summary_edit_events(user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    meeting_id = str(payload.get("meeting_id") or "").strip()
    if not meeting_id:
        raise HTTPException(status_code=400, detail="meeting_id required")

    source_summary_type = normalize_summary_source(str(payload.get("source_summary_type") or ""))
    if source_summary_type not in {"llm_generated", "user_edited"}:
        raise HTTPException(status_code=400, detail="source_summary_type must be llm_generated or user_edited")

    source_summary_id_raw = payload.get("source_summary_id")
    try:
        source_summary_id = int(source_summary_id_raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="source_summary_id must be an integer") from exc

    edits = payload.get("edits")
    if not isinstance(edits, list) or not edits:
        raise HTTPException(status_code=400, detail="edits must be a non-empty list")

    source_variant = repository.fetch_summary_variant_for_user_by_id(user_id, meeting_id, source_summary_id)
    if not source_variant or str(source_variant.get("summary_type")) != source_summary_type:
        raise HTTPException(status_code=404, detail="Source summary not found")

    edit_session_id = str(payload.get("edit_session_id") or "").strip() or secrets.token_hex(12)
    operation_rows: list[tuple[int | None, str, dict[str, Any], dict[str, Any]]] = []

    for operation_index, raw_edit in enumerate(edits, start=1):
        if not isinstance(raw_edit, dict):
            raise HTTPException(status_code=400, detail="Each edit must be a JSON object")

        action = str(raw_edit.get("action") or "").strip()
        segment_summary_id_raw = raw_edit.get("segment_summary_id")
        segment_summary_id: int | None = None
        if segment_summary_id_raw is not None:
            try:
                segment_summary_id = int(segment_summary_id_raw)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail="segment_summary_id must be an integer") from exc

        utterance_idx_raw = raw_edit.get("utterance_idx")
        utterance_idx: int | None = None
        if utterance_idx_raw is not None:
            try:
                utterance_idx = int(utterance_idx_raw)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail="utterance_idx must be an integer") from exc

        raw_segment = raw_edit.get("segment")
        segment_payload: dict[str, Any] | None = None
        if raw_segment is not None:
            if not isinstance(raw_segment, dict):
                raise HTTPException(status_code=400, detail="segment must be a JSON object")
            start_utt = raw_segment.get("start_utt", raw_segment.get("start_utterance_index"))
            end_utt = raw_segment.get("end_utt", raw_segment.get("end_utterance_index"))
            try:
                start_utt_int = int(start_utt) if start_utt is not None else None
                end_utt_int = int(end_utt) if end_utt is not None else None
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail="segment utterance indexes must be integers") from exc

            segment_payload = {
                "segment_summary_id": segment_summary_id,
                "segment_index": raw_segment.get("segment_index"),
                "start_utterance_index": start_utt_int,
                "end_utterance_index": end_utt_int,
                "topic_label": str(raw_segment.get("topic_label") or "").strip(),
                "summary_bullets": normalize_summary_bullets(raw_segment.get("summary_bullets")),
            }

        before_payload: dict[str, Any] = {
            "base_summary_id": source_summary_id,
            "base_summary_type": source_summary_type,
            "edit_session_id": edit_session_id,
            "operation_index": operation_index,
        }
        after_payload: dict[str, Any] = {
            "base_summary_id": source_summary_id,
            "base_summary_type": source_summary_type,
            "edit_session_id": edit_session_id,
            "operation_index": operation_index,
        }

        if segment_summary_id is not None:
            segment_context = repository.fetch_segment_feedback_context(meeting_id, segment_summary_id)
            if not segment_context:
                raise HTTPException(status_code=404, detail="Segment summary not found")
            before_payload["segment"] = {
                "segment_summary_id": segment_context["segment_summary_id"],
                "segment_index": segment_context["segment_index"],
                "topic_label": segment_context.get("topic_label") or "",
                "summary_bullets": normalize_summary_bullets(segment_context.get("summary_bullets")),
                "start_utterance_index": segment_context.get("start_utterance_index"),
                "end_utterance_index": segment_context.get("end_utterance_index"),
            }

        if segment_payload is not None:
            after_payload["segment"] = segment_payload

        if action == "add_boundary":
            if utterance_idx is None:
                raise HTTPException(status_code=400, detail="utterance_idx required for add_boundary")
            before_payload["utterance_idx"] = utterance_idx
            before_payload["is_boundary"] = False
            after_payload["utterance_idx"] = utterance_idx
            after_payload["is_boundary"] = True
            event_type = "split_segment"
        elif action == "remove_boundary":
            if utterance_idx is None:
                raise HTTPException(status_code=400, detail="utterance_idx required for remove_boundary")
            before_payload["utterance_idx"] = utterance_idx
            before_payload["is_boundary"] = True
            after_payload["utterance_idx"] = utterance_idx
            after_payload["is_boundary"] = False
            event_type = "merge_segments"
        elif action == "edit_topic_label":
            topic_label = str(raw_edit.get("topic_label") or "").strip()
            if not topic_label:
                raise HTTPException(status_code=400, detail="topic_label required for edit_topic_label")
            after_payload["topic_label"] = topic_label
            event_type = "edit_topic_label"
        elif action == "edit_summary_bullets":
            summary_bullets = normalize_summary_bullets(raw_edit.get("summary_bullets"))
            if not summary_bullets:
                raise HTTPException(status_code=400, detail="summary_bullets required for edit_summary_bullets")
            after_payload["summary_bullets"] = summary_bullets
            event_type = "edit_summary_bullets"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported edit action: {action}")

        operation_rows.append(
            (
                segment_summary_id,
                event_type,
                before_payload,
                after_payload,
            )
        )

    text_only_user_summary_update = (
        source_summary_type == "user_edited"
        and all(
            event_type in {"edit_topic_label", "edit_summary_bullets"}
            for _segment_summary_id, event_type, _before_payload, _after_payload in operation_rows
        )
    )
    if text_only_user_summary_update:
        updated_count = repository.update_user_summary_text_edits(
            meeting_id,
            source_summary_id,
            operation_rows,
        )
        if updated_count <= 0:
            raise HTTPException(status_code=404, detail="No editable summary segments were updated")
        return {
            "edit_session_id": edit_session_id,
            "source_summary_id": source_summary_id,
            "source_summary_type": source_summary_type,
            "operation_count": len(operation_rows),
            "feedback_event_count": 0,
            "updated_in_place": True,
        }

    for segment_summary_id, event_type, before_payload, after_payload in operation_rows:
        repository.insert_feedback_event(
            meeting_id=meeting_id,
            summary_id=source_summary_id,
            segment_summary_id=segment_summary_id,
            event_type=event_type,
            before_payload=before_payload,
            after_payload=after_payload,
            user_id=user_id,
        )

    repository.enqueue_user_summary_materialize_task(
        meeting_id,
        edit_session_id=edit_session_id,
    )

    return {
        "edit_session_id": edit_session_id,
        "source_summary_id": source_summary_id,
        "source_summary_type": source_summary_type,
        "operation_count": len(operation_rows),
        "feedback_event_count": len(operation_rows),
        "updated_in_place": False,
    }
