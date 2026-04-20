from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone

from psycopg.types.json import Json

from feedback_common import (
    env,
    fetch_meeting_utterance_lookup,
    get_conn,
    get_local_tmp_root,
    normalize_segments_by_next_start,
    setup_logger,
    upload_file,
    write_json,
)


def renumber_segments(segments: list[dict]) -> list[dict]:
    for idx, seg in enumerate(segments, start=1):
        seg["segment_index"] = idx
    return segments


def apply_events(segments: list[dict], events: list[dict]) -> list[dict]:
    current = deepcopy(segments)

    def find_pos(segment_index: int) -> int | None:
        for i, seg in enumerate(current):
            if seg["segment_index"] == segment_index:
                return i
        return None

    for ev in events:
        event_type = ev["event_type"]
        before = ev["before_payload"] or {}
        after = ev["after_payload"] or {}

        if event_type == "accept_summary":
            continue

        if event_type == "edit_topic_label":
            seg_idx = before["segment_index"]
            pos = find_pos(seg_idx)
            if pos is not None:
                current[pos]["topic_label"] = after["topic_label"]

        elif event_type == "edit_summary_bullets":
            seg_idx = before["segment_index"]
            pos = find_pos(seg_idx)
            if pos is not None:
                current[pos]["summary_bullets"] = after["summary_bullets"]

        elif event_type == "merge_segments":
            left_idx = before["left_segment_index"]
            right_idx = before["right_segment_index"]
            left_pos = find_pos(left_idx)
            right_pos = find_pos(right_idx)
            if left_pos is not None and right_pos is not None and left_pos < right_pos:
                merged = {
                    "segment_summary_id": None,
                    "segment_index": left_idx,
                    "start_utterance_index": after["start_utterance_index"],
                    "end_utterance_index": after["end_utterance_index"],
                    "topic_label": after["topic_label"],
                    "summary_bullets": after["summary_bullets"],
                    "status": "complete",
                }
                current = current[:left_pos] + [merged] + current[right_pos + 1 :]

        elif event_type == "split_segment":
            seg_idx = before["segment_index"]
            pos = find_pos(seg_idx)
            if pos is not None:
                left = after["left"]
                right = after["right"]
                current = current[:pos] + [
                    {
                        "segment_summary_id": None,
                        "segment_index": left["segment_index"],
                        "start_utterance_index": left["start_utterance_index"],
                        "end_utterance_index": left["end_utterance_index"],
                        "topic_label": left["topic_label"],
                        "summary_bullets": left["summary_bullets"],
                        "status": "complete",
                    },
                    {
                        "segment_summary_id": None,
                        "segment_index": right["segment_index"],
                        "start_utterance_index": right["start_utterance_index"],
                        "end_utterance_index": right["end_utterance_index"],
                        "topic_label": right["topic_label"],
                        "summary_bullets": right["summary_bullets"],
                        "status": "complete",
                    },
                ] + current[pos + 1 :]

    return renumber_segments(current)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meeting-id", required=True)
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()

    logger = setup_logger("materialize_corrected_recap")
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
        row = cur.fetchone()
        if not row:
            raise RuntimeError(f"No llm_generated summary found for {args.meeting_id}")
        generated_summary_id = row["summary_id"]

        cur.execute(
            """
            SELECT
                ss.segment_summary_id,
                ss.segment_index,
                ss.topic_label,
                ss.summary_bullets,
                su.utterance_index AS start_utterance_index,
                eu.utterance_index AS end_utterance_index
            FROM segment_summaries ss
            JOIN topic_segments ts ON ts.topic_segment_id = ss.topic_segment_id
            JOIN utterances su ON su.utterance_id = ts.start_utterance_id
            JOIN utterances eu ON eu.utterance_id = ts.end_utterance_id
            WHERE ss.summary_id = %s
            ORDER BY ss.segment_index
            """,
            (generated_summary_id,),
        )
        generated_segments = cur.fetchall()

        cur.execute(
            """
            SELECT *
            FROM feedback_events
            WHERE meeting_id = %s
            ORDER BY created_at, feedback_event_id
            """,
            (args.meeting_id,),
        )
        events = cur.fetchall()

    utterance_lookup = fetch_meeting_utterance_lookup(conn, args.meeting_id)
    corrected = apply_events(generated_segments, events)
    normalized_corrected = normalize_segments_by_next_start(corrected, utterance_lookup)

    corrected_payload_segments = [
        {
            "segment_id": seg["segment_index"],
            "t_start": seg["t_start"],
            "t_end": seg["t_end"],
            "topic_label": seg["topic_label"],
            "summary_bullets": seg["summary_bullets"],
            "status": "complete",
        }
        for seg in normalized_corrected
    ]

    payload = {
        "meeting_id": args.meeting_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "feedback-materializer-v1",
        "total_segments": len(corrected_payload_segments),
        "recap": corrected_payload_segments,
    }

    local_path = (
        get_local_tmp_root()
        / "edited_recaps"
        / args.meeting_id
        / f"v{args.version}.json"
    )
    write_json(local_path, payload)

    object_key = f"{env('FEEDBACK_PREFIX')}/recaps/edited/{args.meeting_id}/v{args.version}.json"
    upload_file(local_path, object_key, logger)

    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM summaries
            WHERE meeting_id = %s
              AND summary_type = 'user_edited'
              AND version = %s
            """,
            (args.meeting_id, args.version),
        )
        cur.execute(
            """
            DELETE FROM topic_segments
            WHERE meeting_id = %s
              AND segment_type = 'user_corrected'
            """,
            (args.meeting_id,),
        )

        cur.execute(
            """
            INSERT INTO summaries (
                meeting_id, summary_type, summary_object_key, created_by_user_id, version
            )
            VALUES (%s, 'user_edited', %s, NULL, %s)
            RETURNING summary_id
            """,
            (args.meeting_id, object_key, args.version),
        )
        corrected_summary_id = cur.fetchone()["summary_id"]

        inserted_segment_ids: list[int] = []
        for seg in normalized_corrected:
            start_row = utterance_lookup[seg["start_utterance_index"]]
            end_row = utterance_lookup[seg["end_utterance_index"]]

            cur.execute(
                """
                INSERT INTO topic_segments (
                    meeting_id, segment_type, segment_index,
                    start_utterance_id, end_utterance_id,
                    start_time_sec, end_time_sec, topic_label
                )
                VALUES (%s, 'user_corrected', %s, %s, %s, %s, %s, %s)
                RETURNING topic_segment_id
                """,
                (
                    args.meeting_id,
                    seg["segment_index"],
                    start_row["utterance_id"],
                    end_row["utterance_id"],
                    seg["t_start"],
                    seg["t_end"],
                    seg["topic_label"],
                ),
            )
            inserted_segment_ids.append(cur.fetchone()["topic_segment_id"])

        for seg, topic_segment_id in zip(normalized_corrected, inserted_segment_ids):
            cur.execute(
                """
                INSERT INTO segment_summaries (
                    meeting_id, topic_segment_id, summary_id, segment_index,
                    topic_label, summary_bullets, status,
                    model_name, model_version, prompt_version
                )
                VALUES (%s, %s, %s, %s, %s, %s, 'complete',
                        'feedback-materializer', 'v1', 'feedback-v1')
                """,
                (
                    args.meeting_id,
                    topic_segment_id,
                    corrected_summary_id,
                    seg["segment_index"],
                    seg["topic_label"],
                    Json(seg["summary_bullets"]),
                ),
            )

    conn.commit()
    logger.info("Materialized corrected recap for meeting=%s", args.meeting_id)


if __name__ == "__main__":
    main()
