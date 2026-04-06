#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from feedback_common import (  # noqa: E402
    build_model_utterances_by_meeting,
    ensure_dir,
    fetch_source_utterances,
    get_conn,
    upsert_meeting_artifact,
    upload_file,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meeting-id", required=True)
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--transition-index", type=int, default=3)
    parser.add_argument(
        "--min-utterance-chars",
        type=int,
        default=1,
        help="Minimum characters to keep in online inference view. Use 20 to mirror Stage 1 training exactly.",
    )
    parser.add_argument(
        "--max-words-per-utterance",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--stage1-responses-jsonl",
        type=Path,
        default=None,
        help="Optional Stage 1 response file. If supplied, Stage 2 segment inputs will also be built.",
    )
    parser.add_argument(
        "--boundary-threshold",
        type=float,
        default=0.5,
        help="Fallback threshold if Stage 1 responses provide boundary_probability but not is_boundary.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/block/user-behaviour/online_inference/stage1"),
    )
    parser.add_argument(
        "--upload-artifacts",
        action="store_true",
        help="Upload built artifacts to object storage using rclone.",
    )
    parser.add_argument(
        "--stage1-object-prefix",
        default="production/inference_requests/stage1",
    )
    parser.add_argument(
        "--stage2-object-prefix",
        default="production/inference_requests/stage2",
    )
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional local log file path.",
    )
    return parser.parse_args()


def setup_logger(log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("build_online_inference_payloads")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def make_padding(position: int) -> dict:
    return {
        "position": position,
        "speaker": None,
        "t_start": None,
        "t_end": None,
        "text": "",
    }


def make_window_entry(position: int, utterance: dict) -> dict:
    return {
        "position": position,
        "speaker": utterance["speaker_label"],
        "t_start": utterance["start_time_sec"],
        "t_end": utterance["end_time_sec"],
        "text": utterance["text"],
    }


def build_stage1_request_rows(
    meeting_id: str,
    utterances: list[dict],
    window_size: int,
    transition_index: int,
) -> list[dict]:
    rows: list[dict] = []

    for left_idx in range(len(utterances) - 1):
        right_idx = left_idx + 1
        left = utterances[left_idx]
        right = utterances[right_idx]
        start_index = left_idx - transition_index

        window: list[dict] = []
        for pos in range(window_size):
            idx = start_index + pos
            if 0 <= idx < len(utterances):
                window.append(make_window_entry(pos, utterances[idx]))
            else:
                window.append(make_padding(pos))

        first_real_start = next(
            (item["t_start"] for item in window if item["t_start"] is not None),
            0.0,
        )

        rows.append(
            {
                "meeting_id": meeting_id,
                "request_id": f"{meeting_id}_t{left_idx}",
                "window": window,
                "transition_index": transition_index,
                "meeting_offset_seconds": first_real_start,
                "metadata": {
                    "left_model_index": left["model_index"],
                    "right_model_index": right["model_index"],
                    "left_model_utterance_id": left["model_utterance_id"],
                    "right_model_utterance_id": right["model_utterance_id"],
                    "left_source_utterance_id": left["source_utterance_id"],
                    "right_source_utterance_id": right["source_utterance_id"],
                },
            }
        )

    return rows


def load_stage1_responses(path: Path, meeting_id: str) -> dict[int, dict]:
    responses: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("meeting_id") != meeting_id:
            continue
        left_idx = row["left_model_index"]
        responses[left_idx] = row
    return responses


def artifact_uri(local_path: Path, object_key: str, uploaded: bool) -> str:
    if uploaded:
        return object_key
    return f"local://{local_path.resolve()}"


def build_stage2_inputs(
    meeting_id: str,
    utterances: list[dict],
    stage1_responses: dict[int, dict],
    boundary_threshold: float,
) -> list[dict]:
    if not utterances:
        return []

    boundaries: list[int] = []
    for left_idx in range(len(utterances) - 1):
        response = stage1_responses.get(left_idx)
        if not response:
            continue

        if "is_boundary" in response:
            is_boundary = bool(response["is_boundary"])
        else:
            boundary_prob = float(response.get("boundary_probability", 0.0))
            is_boundary = boundary_prob >= boundary_threshold

        if is_boundary:
            boundaries.append(left_idx)

    segments: list[tuple[int, int]] = []
    start_idx = 0
    for boundary_idx in boundaries:
        segments.append((start_idx, boundary_idx))
        start_idx = boundary_idx + 1
    segments.append((start_idx, len(utterances) - 1))

    rows: list[dict] = []
    total_segments = len(segments)
    for segment_number, (start_idx, end_idx) in enumerate(segments, start=1):
        segment_utts = utterances[start_idx:end_idx + 1]
        rows.append(
            {
                "meeting_id": meeting_id,
                "segment_id": segment_number,
                "t_start": segment_utts[0]["start_time_sec"],
                "t_end": segment_utts[-1]["end_time_sec"],
                "utterances": [
                    {
                        "position": pos,
                        "speaker": utt["speaker_label"],
                        "t_start": utt["start_time_sec"],
                        "t_end": utt["end_time_sec"],
                        "text": utt["text"],
                    }
                    for pos, utt in enumerate(segment_utts)
                ],
                "total_utterances": len(segment_utts),
                "meeting_context": {
                    "total_segments": total_segments,
                    "segment_index_in_meeting": segment_number,
                },
                "metadata": {
                    "start_model_index": segment_utts[0]["model_index"],
                    "end_model_index": segment_utts[-1]["model_index"],
                },
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    if not (0 <= args.transition_index < args.window_size):
        raise ValueError("--transition-index must be between 0 and window_size - 1")

    logger = setup_logger(args.log_file)
    conn = get_conn()

    source_rows = fetch_source_utterances(conn, [args.meeting_id])
    if not source_rows:
        raise RuntimeError(f"No utterances found for meeting {args.meeting_id}")

    model_utterances_by_meeting = build_model_utterances_by_meeting(
        source_rows=source_rows,
        max_words=args.max_words_per_utterance,
        min_chars=args.min_utterance_chars,
    )
    utterances = model_utterances_by_meeting.get(args.meeting_id, [])
    if len(utterances) < 2:
        raise RuntimeError(
            f"Meeting {args.meeting_id} does not have enough derived utterances after filtering to build transitions."
        )

    stage1_requests = build_stage1_request_rows(
        meeting_id=args.meeting_id,
        utterances=utterances,
        window_size=args.window_size,
        transition_index=args.transition_index,
    )

    out_root = args.output_root / args.meeting_id / f"v{args.version}"
    ensure_dir(out_root)

    model_utterances_path = out_root / "model_utterances.json"
    stage1_path = out_root / "stage1_requests.jsonl"
    stage1_json_path = out_root / "stage1_requests.json"
    write_json(model_utterances_path, {"meeting_id": args.meeting_id, "utterances": utterances})
    write_jsonl(stage1_path, stage1_requests)
    write_json(
        stage1_json_path,
        {
            "meeting_id": args.meeting_id,
            "request_count": len(stage1_requests),
            "requests": stage1_requests,
        },
    )

    manifest = {
        "meeting_id": args.meeting_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "window_size": args.window_size,
        "transition_index": args.transition_index,
        "min_utterance_chars": args.min_utterance_chars,
        "max_words_per_utterance": args.max_words_per_utterance,
        "derived_utterances": len(utterances),
        "stage1_request_count": len(stage1_requests),
        "stage2_input_count": 0,
        "stage1_responses_used": args.stage1_responses_jsonl is not None,
    }

    stage2_path: Path | None = None
    if args.stage1_responses_jsonl is not None:
        stage1_responses = load_stage1_responses(args.stage1_responses_jsonl, args.meeting_id)
        stage2_inputs = build_stage2_inputs(
            args.meeting_id,
            utterances,
            stage1_responses,
            boundary_threshold=args.boundary_threshold,
        )
        stage2_path = out_root / "stage2_inputs.jsonl"
        write_jsonl(stage2_path, stage2_inputs)
        write_json(
            out_root / "stage2_inputs.json",
            {
                "meeting_id": args.meeting_id,
                "input_count": len(stage2_inputs),
                "segments": stage2_inputs,
            },
        )
        manifest["stage2_input_count"] = len(stage2_inputs)

    manifest_path = out_root / "manifest.json"
    write_json(manifest_path, manifest)

    stage1_prefix = f"{args.stage1_object_prefix.strip('/')}/{args.meeting_id}/v{args.version}"
    stage1_requests_jsonl_key = f"{stage1_prefix}/requests.jsonl"
    stage1_requests_json_key = f"{stage1_prefix}/requests.json"
    stage1_model_utterances_key = f"{stage1_prefix}/model_utterances.json"
    stage1_manifest_key = f"{stage1_prefix}/manifest.json"

    stage2_inputs_jsonl_key: str | None = None
    stage2_inputs_json_key: str | None = None
    stage2_json_path: Path | None = None
    if stage2_path is not None:
        stage2_prefix = f"{args.stage2_object_prefix.strip('/')}/{args.meeting_id}/v{args.version}"
        stage2_inputs_jsonl_key = f"{stage2_prefix}/inputs.jsonl"
        stage2_inputs_json_key = f"{stage2_prefix}/inputs.json"
        stage2_json_path = out_root / "stage2_inputs.json"

    if args.upload_artifacts:
        upload_file(stage1_path, stage1_requests_jsonl_key, logger)
        upload_file(stage1_json_path, stage1_requests_json_key, logger)
        upload_file(model_utterances_path, stage1_model_utterances_key, logger)
        upload_file(manifest_path, stage1_manifest_key, logger)

        if stage2_path is not None and stage2_inputs_jsonl_key and stage2_inputs_json_key and stage2_json_path is not None:
            upload_file(stage2_path, stage2_inputs_jsonl_key, logger)
            upload_file(stage2_json_path, stage2_inputs_json_key, logger)

    upsert_meeting_artifact(
        conn,
        args.meeting_id,
        "stage1_requests_jsonl",
        artifact_uri(stage1_path, stage1_requests_jsonl_key, args.upload_artifacts),
        "application/x-ndjson",
        args.version,
    )
    upsert_meeting_artifact(
        conn,
        args.meeting_id,
        "stage1_requests_json",
        artifact_uri(stage1_json_path, stage1_requests_json_key, args.upload_artifacts),
        "application/json",
        args.version,
    )
    upsert_meeting_artifact(
        conn,
        args.meeting_id,
        "stage1_model_utterances_json",
        artifact_uri(model_utterances_path, stage1_model_utterances_key, args.upload_artifacts),
        "application/json",
        args.version,
    )
    upsert_meeting_artifact(
        conn,
        args.meeting_id,
        "stage1_manifest_json",
        artifact_uri(manifest_path, stage1_manifest_key, args.upload_artifacts),
        "application/json",
        args.version,
    )

    if stage2_path is not None and stage2_inputs_jsonl_key and stage2_inputs_json_key and stage2_json_path is not None:
        upsert_meeting_artifact(
            conn,
            args.meeting_id,
            "stage2_inputs_jsonl",
            artifact_uri(stage2_path, stage2_inputs_jsonl_key, args.upload_artifacts),
            "application/x-ndjson",
            args.version,
        )
        upsert_meeting_artifact(
            conn,
            args.meeting_id,
            "stage2_inputs_json",
            artifact_uri(stage2_json_path, stage2_inputs_json_key, args.upload_artifacts),
            "application/json",
            args.version,
        )

    conn.commit()

    logger.info(
        "Built online inference payloads for meeting=%s stage1_requests=%d stage2_inputs=%d",
        args.meeting_id,
        manifest["stage1_request_count"],
        manifest["stage2_input_count"],
    )


if __name__ == "__main__":
    main()
