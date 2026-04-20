from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import re
from pathlib import Path

from feedback_common import (
    env,
    fetch_source_utterances,
    fetch_topic_segments,
    get_conn,
    get_local_tmp_root,
    insert_dataset_version,
    label_counts,
    setup_logger,
    upload_dir,
    write_json,
    write_jsonl,
    build_model_utterances_by_meeting,
    build_stage1_rows,
    pick_stage1_examples,
)

VERSION_DIR_RE = re.compile(r"^v(\d+)$")


def resolve_version(version_arg: str, root: Path) -> int:
    if version_arg != "auto":
        return int(version_arg)

    existing = [
        int(match.group(1))
        for child in root.iterdir()
        if child.is_dir() and (match := VERSION_DIR_RE.match(child.name))
    ] if root.exists() else []
    return max(existing, default=0) + 1


def build_stage1_feedback_pool(
    conn,
    version: int,
    logger,
    candidate_meetings: list[str] | None = None,
) -> None:
    with conn.cursor() as cur:
        if candidate_meetings is None:
            cur.execute(
                """
                SELECT DISTINCT fe.meeting_id
                FROM feedback_events fe
                JOIN meetings m
                  ON m.meeting_id = fe.meeting_id
                WHERE m.is_valid = TRUE
                  AND fe.event_type IN ('merge_segments', 'split_segment', 'boundary_correction')
                ORDER BY meeting_id
                """
            )
        else:
            cur.execute(
                """
                SELECT DISTINCT fe.meeting_id
                FROM feedback_events fe
                JOIN meetings m
                  ON m.meeting_id = fe.meeting_id
                WHERE m.is_valid = TRUE
                  AND fe.meeting_id = ANY(%s)
                  AND fe.event_type IN ('merge_segments', 'split_segment', 'boundary_correction')
                ORDER BY meeting_id
                """,
                (candidate_meetings,),
            )
        candidate_meetings = [row["meeting_id"] for row in cur.fetchall()]

    if not candidate_meetings:
        logger.info("No structural feedback events found; skipping Stage 1 feedback-pool build")
        return

    source_rows = fetch_source_utterances(conn, candidate_meetings)
    corrected_segments = fetch_topic_segments(conn, candidate_meetings, "user_corrected")

    topic_segments_by_meeting: dict[str, list[dict]] = defaultdict(list)
    for row in corrected_segments:
        topic_segments_by_meeting[row["meeting_id"]].append(row)

    model_utterances_by_meeting = build_model_utterances_by_meeting(
        source_rows=source_rows,
        max_words=int(env("MAX_WORDS_PER_UTTERANCE")),
        min_chars=int(env("MIN_UTTERANCE_CHARS")),
    )

    dataset_rows = build_stage1_rows(
        model_utterances_by_meeting=model_utterances_by_meeting,
        topic_segments_by_meeting=topic_segments_by_meeting,
        window_size=int(env("WINDOW_SIZE")),
        transition_index=int(env("TRANSITION_INDEX")),
    )

    eligible_meetings = sorted({row["input"]["meeting_id"] for row in dataset_rows})
    out_root = get_local_tmp_root() / "datasets" / "roberta_stage1_feedback_pool" / f"v{version}"
    out_root.mkdir(parents=True, exist_ok=True)

    write_jsonl(out_root / "feedback_examples.jsonl", dataset_rows)
    write_json(
        out_root / "meeting_ids.json",
        {
            "candidate_meeting_ids": candidate_meetings,
            "eligible_meeting_ids": eligible_meetings,
        },
    )
    write_json(
        out_root / "examples.json",
        pick_stage1_examples(dataset_rows),
    )

    manifest = {
        "dataset_name": "roberta_stage1_feedback_pool",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_type": "production_feedback",
        "ongoing_version": {
            "feedback_pool_version": version,
        },
        "packaging": {
            "type": "retraining_pool",
            "reason": "Initial implementation stores feedback-derived Stage 1 examples as a candidate retraining pool instead of forcing train/val/test splits before temporal retraining policy is applied.",
        },
        "params": {
            "window_size": int(env("WINDOW_SIZE")),
            "transition_index": int(env("TRANSITION_INDEX")),
            "min_utterance_chars": int(env("MIN_UTTERANCE_CHARS")),
            "max_words_per_utterance": int(env("MAX_WORDS_PER_UTTERANCE")),
            "segment_type": "user_corrected",
        },
        "meetings": {
            "candidate": len(candidate_meetings),
            "eligible": len(eligible_meetings),
            "candidate_meeting_ids": candidate_meetings,
            "eligible_meeting_ids": eligible_meetings,
        },
        "rows": label_counts(dataset_rows),
    }
    write_json(out_root / "manifest.json", manifest)

    object_prefix = f"{env('STAGE1_FEEDBACK_POOL_PREFIX', 'datasets/roberta_stage1_feedback_pool')}/v{version}"
    upload_dir(out_root, object_prefix, logger)

    insert_dataset_version(
        conn=conn,
        dataset_name="roberta_stage1_feedback_pool",
        stage="stage1",
        source_type="production_feedback",
        object_key=f"{object_prefix}/manifest.json",
        manifest_json=manifest,
    )

    logger.info("Built Stage 1 feedback retraining pool v%s", version)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="auto")
    parser.add_argument("--meeting-id", action="append", default=[])
    args = parser.parse_args()

    logger = setup_logger("build_feedback_pool")
    conn = get_conn()

    version_root = get_local_tmp_root() / "datasets" / "roberta_stage1_feedback_pool"
    version = resolve_version(args.version, version_root)

    build_stage1_feedback_pool(
        conn,
        version=version,
        logger=logger,
        candidate_meetings=sorted(set(args.meeting_id)) or None,
    )

    logger.info("Finished building Stage 1 feedback retraining pool only")


if __name__ == "__main__":
    main()
