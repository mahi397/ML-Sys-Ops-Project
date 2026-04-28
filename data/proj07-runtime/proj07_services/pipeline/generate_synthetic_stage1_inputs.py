#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


TOPIC_BLOCKS = [
    {
        "topic": "boundary model calibration",
        "speakers": ["Ava", "Noah", "Lena"],
        "utterances": [
            "We should verify whether the boundary threshold is too conservative on shorter meetings.",
            "The earlier run only produced one segment, so the probability cutoff may need calibration.",
            "I want synthetic meetings to include clearer pauses and topic shifts for the replay demo.",
            "Let us log the request and response payloads so the serving contract is easy to inspect.",
            "We can compare speaker changes and silence gaps to see which factor drives the mock boundary score.",
        ],
    },
    {
        "topic": "stage two summarization contract",
        "speakers": ["Mia", "Owen", "Zara"],
        "utterances": [
            "The stage two endpoint should receive reconstructed segments instead of raw transcript windows.",
            "We also need the output to contain a topic label and a small list of summary bullets.",
            "That way the recap artifact can be stored directly without extra transformation logic.",
            "I would like the payload to include segment timing so it lines up with the corrected recap later.",
            "For the demo we can keep the prompt simple as long as the JSON contract stays stable.",
        ],
    },
    {
        "topic": "data leakage prevention",
        "speakers": ["Iris", "Ben", "Kai"],
        "utterances": [
            "The retraining snapshot should keep newer meetings out of the training split.",
            "If we reuse the same meeting in both train and validation, the results will be misleading.",
            "We should stamp meetings with dataset version once they are consumed into a snapshot.",
            "That makes the next batch run easy because only unseen meetings remain eligible.",
            "I want the manifest to record exactly which meetings were assigned to validation and test.",
        ],
    },
    {
        "topic": "service replay observability",
        "speakers": ["Emma", "Luca", "Sara"],
        "utterances": [
            "The runtime should be independently runnable from a fresh virtual machine.",
            "We can generate synthetic stage one requests locally and still upload them to object storage.",
            "That would let the TA see request artifacts even without running the online feature path first.",
            "Please include multiple synthetic meetings so the runtime shows a few minutes of realistic traffic.",
            "The logs should clearly state which meetings were generated and which were replayed.",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/block/user-behaviour/inference_requests/stage1"),
    )
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--meeting-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--transition-index", type=int, default=3)
    parser.add_argument("--upload-artifacts", action="store_true")
    parser.add_argument("--rclone-remote", default="rclone_s3")
    parser.add_argument("--bucket", default="objstore-proj07")
    parser.add_argument("--stage1-object-prefix", default="production/inference_requests/stage1")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--meeting-id", action="append", default=[])
    return parser.parse_args()


def setup_logger(log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("generate_synthetic_stage1_inputs")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def ensure_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            "Cannot create synthetic output directory "
            f"{path}. Synthetic meeting folders such as "
            "'synthetic_endpoint_01' are created automatically; "
            f"the parent output root is not writable: {path.parent}"
        ) from exc


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_command(cmd: list[str], logger: logging.Logger, label: str) -> None:
    logger.info("START | %s", label)
    logger.info("CMD   | %s", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        if result.stdout:
            logger.error("STDOUT:\n%s", result.stdout)
        if result.stderr:
            logger.error("STDERR:\n%s", result.stderr)
        raise RuntimeError(f"Command failed: {label}")
    logger.info("DONE  | %s", label)


def upload_file(local_path: Path, object_key: str, remote: str, bucket: str, logger: logging.Logger) -> None:
    destination = f"{remote}:{bucket}/{object_key}"
    run_command(["rclone", "copyto", str(local_path), destination, "-P"], logger, f"upload {local_path.name}")


def make_padding(position: int) -> dict:
    return {"position": position, "speaker": None, "t_start": None, "t_end": None, "text": ""}


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

        first_real_start = next((item["t_start"] for item in window if item["t_start"] is not None), 0.0)
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


def synthetic_meeting_ids(args: argparse.Namespace) -> list[str]:
    if args.meeting_id:
        return args.meeting_id
    return [f"synthetic_endpoint_{idx:02d}" for idx in range(1, args.meeting_count + 1)]


def build_meeting_utterances(meeting_id: str, rng: random.Random) -> list[dict]:
    utterances: list[dict] = []
    current_time = 0.0
    source_idx = 0

    block_count = rng.randint(3, 4)
    chosen_blocks = rng.sample(TOPIC_BLOCKS, k=block_count)

    for block_number, block in enumerate(chosen_blocks, start=1):
        turns = block["utterances"][:]
        rng.shuffle(turns)
        for turn_number, text in enumerate(turns, start=1):
            speaker = block["speakers"][(turn_number - 1) % len(block["speakers"])]
            word_count = max(1, len(text.split()))
            duration = round(1.8 + word_count * 0.32, 3)
            start_time = round(current_time, 3)
            end_time = round(start_time + duration, 3)

            utterances.append(
                {
                    "meeting_id": meeting_id,
                    "model_utterance_id": f"{meeting_id}_utt_{source_idx + 1}",
                    "source_utterance_id": source_idx + 1,
                    "source_utterance_index": source_idx,
                    "model_index": source_idx,
                    "speaker_label": speaker,
                    "start_time_sec": start_time,
                    "end_time_sec": end_time,
                    "text": text,
                }
            )

            source_idx += 1
            current_time = end_time + rng.uniform(1.2, 3.8)

        if block_number < len(chosen_blocks):
            current_time += rng.uniform(9.0, 18.0)

    return utterances


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_file)
    rng = random.Random(args.seed)

    if not (0 <= args.transition_index < args.window_size):
        raise ValueError("--transition-index must be between 0 and window_size - 1")

    meeting_ids = synthetic_meeting_ids(args)
    generated_manifest_rows: list[dict] = []

    for meeting_id in meeting_ids:
        utterances = build_meeting_utterances(meeting_id, rng)
        requests = build_stage1_request_rows(meeting_id, utterances, args.window_size, args.transition_index)

        out_root = args.output_root / meeting_id / f"v{args.version}"
        ensure_dir(out_root)

        model_utterances_path = out_root / "model_utterances.json"
        stage1_requests_jsonl = out_root / "stage1_requests.jsonl"
        stage1_requests_json = out_root / "stage1_requests.json"
        manifest_path = out_root / "manifest.json"

        write_json(model_utterances_path, {"meeting_id": meeting_id, "utterances": utterances})
        write_jsonl(stage1_requests_jsonl, requests)
        write_json(stage1_requests_json, {"meeting_id": meeting_id, "request_count": len(requests), "requests": requests})
        write_json(
            manifest_path,
            {
                "meeting_id": meeting_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_type": "synthetic",
                "source_subtype": "synthetic_stage1_bootstrap",
                "window_size": args.window_size,
                "transition_index": args.transition_index,
                "derived_utterances": len(utterances),
                "stage1_request_count": len(requests),
            },
        )

        if args.upload_artifacts:
            object_prefix = f"{args.stage1_object_prefix.strip('/')}/{meeting_id}/v{args.version}"
            upload_file(stage1_requests_jsonl, f"{object_prefix}/requests.jsonl", args.rclone_remote, args.bucket, logger)
            upload_file(stage1_requests_json, f"{object_prefix}/requests.json", args.rclone_remote, args.bucket, logger)
            upload_file(model_utterances_path, f"{object_prefix}/model_utterances.json", args.rclone_remote, args.bucket, logger)
            upload_file(manifest_path, f"{object_prefix}/manifest.json", args.rclone_remote, args.bucket, logger)

        generated_manifest_rows.append(
            {
                "meeting_id": meeting_id,
                "utterances": len(utterances),
                "requests": len(requests),
                "output_root": str(out_root),
            }
        )
        logger.info("Generated synthetic Stage 1 inputs for meeting=%s requests=%d", meeting_id, len(requests))
        print(meeting_id)

    manifest_root = args.output_root / "_synthetic_generation"
    ensure_dir(manifest_root)
    write_json(
        manifest_root / f"v{args.version}.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "meeting_ids": meeting_ids,
            "rows": generated_manifest_rows,
            "generator": "proj07_services.pipeline.generate_synthetic_stage1_inputs",
        },
    )


if __name__ == "__main__":
    main()
