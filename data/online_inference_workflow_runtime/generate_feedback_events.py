from __future__ import annotations

import argparse
import random

from psycopg.types.json import Json

from feedback_common import (
    env,
    get_conn,
    get_local_tmp_root,
    setup_logger,
    upload_file,
    write_jsonl,
)


def tweak_bullets(bullets: list[str]) -> list[str]:
    if not bullets:
        return ["Edited summary bullet."]
    updated = bullets[:]
    updated[0] = updated[0] + " (edited)"
    return updated[:4]


def build_split_event(seg: dict) -> dict | None:
    start_idx = seg["start_idx"]
    end_idx = seg["end_idx"]
    if end_idx - start_idx < 3:
        return None

    split_after = (start_idx + end_idx) // 2
    return {
        "event_type": "split_segment",
        "before_payload": {
            "segment_index": seg["segment_index"],
            "start_utterance_index": start_idx,
            "end_utterance_index": end_idx,
        },
        "after_payload": {
            "left": {
                "segment_index": seg["segment_index"],
                "start_utterance_index": start_idx,
                "end_utterance_index": split_after,
                "topic_label": f"{seg['topic_label']} Part 1",
                "summary_bullets": seg["summary_bullets"][:2] or ["Left split segment."],
            },
            "right": {
                "segment_index": seg["segment_index"] + 1,
                "start_utterance_index": split_after + 1,
                "end_utterance_index": end_idx,
                "topic_label": f"{seg['topic_label']} Part 2",
                "summary_bullets": seg["summary_bullets"][2:] or ["Right split segment."],
            },
        },
    }


def build_merge_event(left: dict, right: dict) -> dict:
    merged_bullets = (left["summary_bullets"] + right["summary_bullets"])[:4]
    return {
        "event_type": "merge_segments",
        "before_payload": {
            "left_segment_index": left["segment_index"],
            "right_segment_index": right["segment_index"],
        },
        "after_payload": {
            "segment_index": left["segment_index"],
            "start_utterance_index": left["start_idx"],
            "end_utterance_index": right["end_idx"],
            "topic_label": left["topic_label"],
            "summary_bullets": merged_bullets or ["Merged segment."],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meeting-id", required=True)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--structural-event", choices=["auto", "merge", "split", "none"], default="auto")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    logger = setup_logger("generate_feedback_events")
    conn = get_conn()

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT summary_id
            FROM summaries
            WHERE meeting_id = %s
              AND summary_type = 'llm_generated'
            ORDER BY version DESC, created_at DESC
            LIMIT 1
            """,
            (args.meeting_id,),
        )
        summary_row = cur.fetchone()
        if not summary_row:
            raise RuntimeError(f"No llm_generated summary found for {args.meeting_id}")
        summary_id = summary_row["summary_id"]

        cur.execute(
            """
            SELECT
                ss.segment_summary_id,
                ss.segment_index,
                ss.topic_label,
                ss.summary_bullets,
                su.utterance_index AS start_idx,
                eu.utterance_index AS end_idx
            FROM segment_summaries ss
            JOIN topic_segments ts ON ts.topic_segment_id = ss.topic_segment_id
            JOIN utterances su ON su.utterance_id = ts.start_utterance_id
            JOIN utterances eu ON eu.utterance_id = ts.end_utterance_id
            WHERE ss.summary_id = %s
            ORDER BY ss.segment_index
            """,
            (summary_id,),
        )
        segments = cur.fetchall()

    events: list[dict] = []
    structural = args.structural_event
    structural_target_ids: set[int] = set()

    if structural == "auto":
        longest = max(segments, key=lambda s: s["end_idx"] - s["start_idx"], default=None)
        if longest and longest["end_idx"] - longest["start_idx"] >= 3:
            structural = "split"
        elif len(segments) >= 2:
            structural = "merge"
        else:
            structural = "none"

    if structural == "split":
        longest = max(segments, key=lambda s: s["end_idx"] - s["start_idx"], default=None)
        if longest:
            structural_target_ids.add(longest["segment_summary_id"])

    if structural == "merge" and len(segments) >= 2:
        structural_target_ids.add(segments[0]["segment_summary_id"])
        structural_target_ids.add(segments[1]["segment_summary_id"])

    for seg in segments:
        if seg["segment_summary_id"] in structural_target_ids:
            continue

        roll = rng.random()
        if roll < 0.60:
            event_type = "accept_summary"
            before = {
                "segment_index": seg["segment_index"],
                "topic_label": seg["topic_label"],
                "summary_bullets": seg["summary_bullets"],
            }
            after = before
        elif roll < 0.80:
            event_type = "edit_topic_label"
            before = {
                "segment_index": seg["segment_index"],
                "topic_label": seg["topic_label"],
            }
            after = {
                "segment_index": seg["segment_index"],
                "topic_label": f"{seg['topic_label']} (edited)",
            }
        else:
            event_type = "edit_summary_bullets"
            before = {
                "segment_index": seg["segment_index"],
                "summary_bullets": seg["summary_bullets"],
            }
            after = {
                "segment_index": seg["segment_index"],
                "summary_bullets": tweak_bullets(seg["summary_bullets"]),
            }

        events.append(
            {
                "meeting_id": args.meeting_id,
                "summary_id": summary_id,
                "segment_summary_id": seg["segment_summary_id"],
                "event_type": event_type,
                "event_source": "emulated",
                "before_payload": before,
                "after_payload": after,
            }
        )

    if structural == "split":
        longest = max(segments, key=lambda s: s["end_idx"] - s["start_idx"], default=None)
        split_ev = build_split_event(longest) if longest else None
        if split_ev:
            events.append(
                {
                    "meeting_id": args.meeting_id,
                    "summary_id": summary_id,
                    "segment_summary_id": longest["segment_summary_id"],
                    "event_type": split_ev["event_type"],
                    "event_source": "emulated",
                    "before_payload": split_ev["before_payload"],
                    "after_payload": split_ev["after_payload"],
                }
            )

    if structural == "merge" and len(segments) >= 2:
        merge_ev = build_merge_event(segments[0], segments[1])
        events.append(
            {
                "meeting_id": args.meeting_id,
                "summary_id": summary_id,
                "segment_summary_id": segments[0]["segment_summary_id"],
                "event_type": merge_ev["event_type"],
                "event_source": "emulated",
                "before_payload": merge_ev["before_payload"],
                "after_payload": merge_ev["after_payload"],
            }
        )

    local_path = (
        get_local_tmp_root()
        / "feedback_events"
        / args.meeting_id
        / f"events_v{args.version}.jsonl"
    )
    write_jsonl(local_path, events)

    object_key = f"{env('FEEDBACK_PREFIX')}/feedback_events/{args.meeting_id}/events_v{args.version}.jsonl"
    upload_file(local_path, object_key, logger)

    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM feedback_events
            WHERE meeting_id = %s
              AND event_source = 'emulated'
            """,
            (args.meeting_id,),
        )
        cur.execute(
            """
            DELETE FROM meeting_artifacts
            WHERE meeting_id = %s
              AND artifact_type = 'feedback_json'
              AND artifact_version = %s
            """,
            (args.meeting_id, args.version),
        )
        cur.execute(
            """
            INSERT INTO meeting_artifacts (
                meeting_id, artifact_type, object_key, content_type, artifact_version
            )
            VALUES (%s, 'feedback_json', %s, 'application/jsonl', %s)
            """,
            (args.meeting_id, object_key, args.version),
        )

        for ev in events:
            cur.execute(
                """
                INSERT INTO feedback_events (
                    meeting_id, summary_id, segment_summary_id,
                    event_type, event_source,
                    before_payload, after_payload, created_by_user_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, NULL)
                """,
                (
                    ev["meeting_id"],
                    ev["summary_id"],
                    ev["segment_summary_id"],
                    ev["event_type"],
                    ev["event_source"],
                    Json(ev["before_payload"]),
                    Json(ev["after_payload"]),
                ),
            )

    conn.commit()
    logger.info("Generated %d feedback events for meeting=%s", len(events), args.meeting_id)


if __name__ == "__main__":
    main()
