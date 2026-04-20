#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from psycopg.types.json import Json

from feedback_common import fetch_meeting_utterance_lookup, get_conn, upsert_meeting_artifact


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_TEMPLATE = SCRIPT_DIR / "flowise_stage2_prompt.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meeting-id", required=True)
    parser.add_argument("--version", type=int, default=1)

    parser.add_argument(
        "--stage1-requests-jsonl",
        type=Path,
        default=None,
        help="Defaults to /mnt/block/user-behaviour/online_inference/stage1/<meeting_id>/v<version>/stage1_requests.jsonl",
    )
    parser.add_argument(
        "--model-utterances-json",
        type=Path,
        default=None,
        help="Defaults to /mnt/block/user-behaviour/online_inference/stage1/<meeting_id>/v<version>/model_utterances.json",
    )

    parser.add_argument(
        "--stage1-mode",
        choices=["mock", "http"],
        default="mock",
        help="Use a mock Stage 1 response generator or call a hypothetical HTTP endpoint.",
    )
    parser.add_argument("--stage1-url", default=None)
    parser.add_argument(
        "--stage1-threshold",
        type=float,
        default=0.5,
        help="Threshold used when deriving boundaries from Stage 1 probabilities.",
    )

    parser.add_argument(
        "--stage2-mode",
        choices=["mock", "flowise", "skip"],
        default="flowise",
        help="Use Flowise for Stage 2, a local mock response generator, or skip Stage 2.",
    )
    parser.add_argument("--stage2-url", default=None, help="Full Flowise prediction URL, if preferred.")
    parser.add_argument("--flowise-base-url", default=os.getenv("FLOWISE_BASE_URL"))
    parser.add_argument("--flowise-flow-id", default=os.getenv("FLOWISE_FLOW_ID"))
    parser.add_argument("--flowise-api-key", default=os.getenv("FLOWISE_API_KEY"))
    parser.add_argument("--prompt-template", type=Path, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument(
        "--model-version",
        default=os.getenv("MODEL_VERSION", "flowise-stage2-v1"),
    )

    parser.add_argument(
        "--stage1-response-root",
        type=Path,
        default=Path("/mnt/block/user-behaviour/inference_responses/stage1"),
    )
    parser.add_argument(
        "--stage2-input-root",
        type=Path,
        default=Path("/mnt/block/user-behaviour/online_inference/stage2"),
    )
    parser.add_argument(
        "--stage2-response-root",
        type=Path,
        default=Path("/mnt/block/user-behaviour/inference_responses/stage2"),
    )
    parser.add_argument(
        "--segments-root",
        type=Path,
        default=Path("/mnt/block/user-behaviour/reconstructed_segments"),
    )
    parser.add_argument(
        "--recap-root",
        type=Path,
        default=Path("/mnt/block/user-behaviour/recaps/generated"),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path.",
    )

    parser.add_argument("--upload-artifacts", action="store_true")
    parser.add_argument("--rclone-remote", default=os.getenv("RCLONE_REMOTE", "rclone_s3"))
    parser.add_argument("--bucket", default=os.getenv("BUCKET", "objstore-proj07"))
    parser.add_argument("--stage1-response-prefix", default="production/inference_responses/stage1")
    parser.add_argument("--stage2-input-prefix", default="production/inference_requests/stage2")
    parser.add_argument("--stage2-response-prefix", default="production/inference_responses/stage2")
    parser.add_argument("--segments-prefix", default="production/reconstructed_segments")
    parser.add_argument("--recap-prefix", default="production/recaps/generated")

    return parser.parse_args()


def setup_logger(log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("replay_to_hypothetical_endpoints")
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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
    cmd = ["rclone", "copyto", str(local_path), destination, "-P"]
    run_command(cmd, logger, f"upload {local_path.name}")


def artifact_uri(local_path: Path, object_key: str, uploaded: bool) -> str:
    if uploaded:
        return object_key
    return f"local://{local_path.resolve()}"


def resolve_stage1_request_path(meeting_id: str, version: int, path: Path | None) -> Path:
    if path is not None:
        return path
    return Path("/mnt/block/user-behaviour/online_inference/stage1") / meeting_id / f"v{version}" / "stage1_requests.jsonl"


def resolve_model_utterance_path(meeting_id: str, version: int, path: Path | None) -> Path:
    if path is not None:
        return path
    return Path("/mnt/block/user-behaviour/online_inference/stage1") / meeting_id / f"v{version}" / "model_utterances.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def stage1_response_from_mock(request_row: dict[str, Any]) -> dict[str, Any]:
    left = request_row["window"][request_row["transition_index"]]
    right = request_row["window"][request_row["transition_index"] + 1]

    gap = 0.0
    if left["t_end"] is not None and right["t_start"] is not None:
        gap = max(0.0, float(right["t_start"]) - float(left["t_end"]))

    speaker_change = left["speaker"] != right["speaker"]
    left_words = len((left["text"] or "").split())
    right_words = len((right["text"] or "").split())

    probability = 0.10
    if gap >= 5:
        probability += 0.55
    if gap >= 20:
        probability += 0.20
    if speaker_change:
        probability += 0.10
    if abs(left_words - right_words) >= 10:
        probability += 0.05

    probability = round(min(probability, 0.99), 3)
    return {
        "meeting_id": request_row["meeting_id"],
        "request_id": request_row["request_id"],
        "left_model_index": request_row["metadata"]["left_model_index"],
        "right_model_index": request_row["metadata"]["right_model_index"],
        "boundary_probability": probability,
        "is_boundary": probability >= 0.5,
    }


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
    data = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)

    req = request.Request(url, data=data, headers=request_headers, method="POST")
    try:
        with request.urlopen(req) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} calling {url}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc}") from exc


def normalize_stage1_response(raw: Any, request_row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RuntimeError("Stage 1 endpoint must return a JSON object")

    probability = raw.get("boundary_probability", raw.get("pred_boundary_prob", raw.get("score", 0.0)))
    probability = float(probability)
    is_boundary = raw.get("is_boundary")
    if is_boundary is None:
        is_boundary = raw.get("pred_boundary_label")
    if is_boundary is None:
        is_boundary = probability >= 0.5

    return {
        "meeting_id": request_row["meeting_id"],
        "request_id": request_row["request_id"],
        "left_model_index": request_row["metadata"]["left_model_index"],
        "right_model_index": request_row["metadata"]["right_model_index"],
        "boundary_probability": round(probability, 3),
        "is_boundary": bool(is_boundary),
        "raw_response": raw,
    }


def build_stage2_inputs(
    meeting_id: str,
    utterances: list[dict[str, Any]],
    stage1_responses: dict[int, dict[str, Any]],
    boundary_threshold: float,
) -> list[dict[str, Any]]:
    if not utterances:
        return []

    boundaries: list[int] = []
    for left_idx in range(len(utterances) - 1):
        response = stage1_responses.get(left_idx)
        if not response:
            continue

        is_boundary = response.get("is_boundary")
        if is_boundary is None:
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

    rows: list[dict[str, Any]] = []
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
                    "start_source_utterance_index": segment_utts[0]["source_utterance_index"],
                    "end_source_utterance_index": segment_utts[-1]["source_utterance_index"],
                },
            }
        )

    return rows


def build_reconstructed_segments(stage2_inputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "meeting_id": stage2_inputs[0]["meeting_id"] if stage2_inputs else None,
        "segment_count": len(stage2_inputs),
        "segments": [
            {
                "segment_id": segment["segment_id"],
                "t_start": segment["t_start"],
                "t_end": segment["t_end"],
                "total_utterances": segment["total_utterances"],
                "start_model_index": segment["metadata"]["start_model_index"],
                "end_model_index": segment["metadata"]["end_model_index"],
                "start_source_utterance_index": segment["metadata"]["start_source_utterance_index"],
                "end_source_utterance_index": segment["metadata"]["end_source_utterance_index"],
            }
            for segment in stage2_inputs
        ],
    }


STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "but", "by", "for", "from", "has", "have", "in",
    "is", "it", "of", "on", "or", "that", "the", "their", "this", "to", "was", "with",
}


def topic_label_from_segment(segment: dict[str, Any]) -> str:
    words: list[str] = []
    seen: set[str] = set()
    for utterance in segment["utterances"]:
        for word in re.findall(r"[A-Za-z']+", utterance["text"].lower()):
            if word in STOPWORDS or len(word) < 4 or word in seen:
                continue
            seen.add(word)
            words.append(word)
            if len(words) == 4:
                break
        if len(words) == 4:
            break

    if not words:
        return f"Segment {segment['segment_id']}"
    return " ".join(word.capitalize() for word in words)


def stage2_response_from_mock(segment: dict[str, Any], model_version: str) -> dict[str, Any]:
    topic_label = topic_label_from_segment(segment)
    utterance_count = segment["total_utterances"]
    speakers = []
    for utterance in segment["utterances"]:
        speaker = utterance["speaker"]
        if speaker not in speakers:
            speakers.append(speaker)

    bullets = [f"The discussion focused on {topic_label.lower()}."]
    if speakers:
        bullets.append(f"Speaker participation in this segment included {', '.join(speakers[:3])}.")
    if utterance_count >= 6:
        bullets.append("This segment covered the topic through a longer multi-turn exchange.")
    elif utterance_count >= 3:
        bullets.append("This segment captured a short focused exchange.")

    return {
        "meeting_id": segment["meeting_id"],
        "segment_id": segment["segment_id"],
        "topic_label": topic_label,
        "summary_bullets": bullets[:3],
        "status": "complete",
        "model_version": model_version,
    }


def build_flowise_question(prompt_template: str, segment: dict[str, Any]) -> str:
    return prompt_template.replace(
        "{{SEGMENT_JSON}}",
        json.dumps(segment, ensure_ascii=False, indent=2),
    )


def extract_text_from_flowise_response(response_json: Any) -> str:
    if isinstance(response_json, str):
        return response_json

    if isinstance(response_json, list):
        parts = [extract_text_from_flowise_response(item) for item in response_json]
        return "\n".join(part for part in parts if part)

    if isinstance(response_json, dict):
        for key in ("text", "message", "output", "result", "response"):
            value = response_json.get(key)
            if isinstance(value, str):
                return value
        if "json" in response_json and isinstance(response_json["json"], (dict, list)):
            return json.dumps(response_json["json"], ensure_ascii=False)
    return json.dumps(response_json, ensure_ascii=False)


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


def parse_stage2_json(text: str) -> dict[str, Any]:
    candidate = strip_code_fences(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(candidate[start:end + 1])


def resolve_flowise_prediction_url(args: argparse.Namespace) -> str:
    if args.stage2_url:
        return args.stage2_url
    if not args.flowise_base_url or not args.flowise_flow_id:
        raise RuntimeError("Flowise mode requires --stage2-url or both --flowise-base-url and --flowise-flow-id")
    return args.flowise_base_url.rstrip("/") + f"/api/v1/prediction/{args.flowise_flow_id}"


def call_flowise_for_segment(
    args: argparse.Namespace,
    segment: dict[str, Any],
    prompt_template: str,
) -> dict[str, Any]:
    url = resolve_flowise_prediction_url(args)
    payload = {
        "question": build_flowise_question(prompt_template, segment),
        "streaming": False,
        "overrideConfig": {
            "sessionId": f"{segment['meeting_id']}_segment_{segment['segment_id']}",
        },
    }

    headers: dict[str, str] = {}
    if args.flowise_api_key:
        headers["Authorization"] = f"Bearer {args.flowise_api_key}"

    raw_response = post_json(url, payload, headers=headers)
    response_text = extract_text_from_flowise_response(raw_response)
    parsed = parse_stage2_json(response_text)

    if "topic_label" not in parsed or "summary_bullets" not in parsed:
        raise RuntimeError(f"Flowise response did not contain required keys: {parsed}")

    bullets = parsed["summary_bullets"]
    if not isinstance(bullets, list) or not bullets:
        raise RuntimeError("Flowise summary_bullets must be a non-empty list")

    return {
        "meeting_id": segment["meeting_id"],
        "segment_id": segment["segment_id"],
        "topic_label": str(parsed["topic_label"]).strip(),
        "summary_bullets": [str(item).strip() for item in bullets if str(item).strip()],
        "status": str(parsed.get("status", "complete")).strip() or "complete",
        "model_version": args.model_version,
        "raw_response": raw_response,
    }


def assemble_recap(
    meeting_id: str,
    stage2_inputs: list[dict[str, Any]],
    stage2_outputs: list[dict[str, Any]],
    model_version: str,
) -> dict[str, Any]:
    by_segment = {row["segment_id"]: row for row in stage2_outputs}
    recap_rows: list[dict[str, Any]] = []

    for segment in stage2_inputs:
        output = by_segment[segment["segment_id"]]
        recap_rows.append(
            {
                "segment_id": segment["segment_id"],
                "t_start": segment["t_start"],
                "t_end": segment["t_end"],
                "topic_label": output["topic_label"],
                "summary_bullets": output["summary_bullets"],
                "status": output["status"],
            }
        )

    return {
        "meeting_id": meeting_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "total_segments": len(recap_rows),
        "recap": recap_rows,
    }


def register_stage1_predictions(
    conn,
    meeting_id: str,
    stage1_requests: list[dict[str, Any]],
    stage1_responses: list[dict[str, Any]],
) -> None:
    request_by_id = {row["request_id"]: row for row in stage1_requests}

    with conn.cursor() as cur:
        for response in stage1_responses:
            request_row = request_by_id.get(response["request_id"])
            if not request_row:
                continue

            left_source_utterance_id = request_row["metadata"].get("left_source_utterance_id")
            right_source_utterance_id = request_row["metadata"].get("right_source_utterance_id")
            if not left_source_utterance_id or not right_source_utterance_id:
                continue
            if left_source_utterance_id == right_source_utterance_id:
                continue

            cur.execute(
                """
                UPDATE utterance_transitions
                SET pred_boundary_prob = %s,
                    pred_boundary_label = %s
                WHERE meeting_id = %s
                  AND left_utterance_id = %s
                  AND right_utterance_id = %s
                """,
                (
                    float(response["boundary_probability"]),
                    bool(response["is_boundary"]),
                    meeting_id,
                    left_source_utterance_id,
                    right_source_utterance_id,
                ),
            )


def register_recap_outputs(
    conn,
    meeting_id: str,
    version: int,
    recap_uri: str,
    stage2_inputs: list[dict[str, Any]],
    stage2_outputs: list[dict[str, Any]],
    model_version: str,
    prompt_version: str,
    stage2_mode: str,
) -> None:
    if not stage2_inputs or not stage2_outputs:
        return

    utterance_lookup = fetch_meeting_utterance_lookup(conn, meeting_id)
    outputs_by_segment_id = {row["segment_id"]: row for row in stage2_outputs}
    model_name = "mock-stage2" if stage2_mode == "mock" else "flowise-stage2"

    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM summaries
            WHERE meeting_id = %s
              AND summary_type = 'llm_generated'
              AND version = %s
            """,
            (meeting_id, version),
        )
        cur.execute(
            """
            DELETE FROM topic_segments
            WHERE meeting_id = %s
              AND segment_type = 'predicted'
            """,
            (meeting_id,),
        )

        inserted_segment_ids: list[tuple[int, int]] = []
        for segment in stage2_inputs:
            start_idx = segment["metadata"]["start_source_utterance_index"]
            end_idx = segment["metadata"]["end_source_utterance_index"]
            start_row = utterance_lookup[start_idx]
            end_row = utterance_lookup[end_idx]
            output = outputs_by_segment_id.get(segment["segment_id"], {})

            cur.execute(
                """
                INSERT INTO topic_segments (
                    meeting_id, segment_type, segment_index,
                    start_utterance_id, end_utterance_id,
                    start_time_sec, end_time_sec, topic_label
                )
                VALUES (%s, 'predicted', %s, %s, %s, %s, %s, %s)
                RETURNING topic_segment_id
                """,
                (
                    meeting_id,
                    segment["segment_id"],
                    start_row["utterance_id"],
                    end_row["utterance_id"],
                    segment["t_start"],
                    segment["t_end"],
                    output.get("topic_label"),
                ),
            )
            inserted_segment_ids.append((segment["segment_id"], cur.fetchone()["topic_segment_id"]))

        cur.execute(
            """
            INSERT INTO summaries (
                meeting_id, summary_type, summary_object_key, created_by_user_id, version
            )
            VALUES (%s, 'llm_generated', %s, NULL, %s)
            RETURNING summary_id
            """,
            (meeting_id, recap_uri, version),
        )
        summary_id = cur.fetchone()["summary_id"]

        for segment_id, topic_segment_id in inserted_segment_ids:
            output = outputs_by_segment_id[segment_id]
            cur.execute(
                """
                INSERT INTO segment_summaries (
                    meeting_id, topic_segment_id, summary_id, segment_index,
                    topic_label, summary_bullets, status,
                    model_name, model_version, prompt_version
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    meeting_id,
                    topic_segment_id,
                    summary_id,
                    segment_id,
                    output["topic_label"],
                    Json(output["summary_bullets"]),
                    output["status"],
                    model_name,
                    model_version,
                    prompt_version,
                ),
            )


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_file)

    stage1_request_path = resolve_stage1_request_path(args.meeting_id, args.version, args.stage1_requests_jsonl)
    model_utterance_path = resolve_model_utterance_path(args.meeting_id, args.version, args.model_utterances_json)

    if not stage1_request_path.exists():
        raise FileNotFoundError(f"Missing Stage 1 request file: {stage1_request_path}")
    if not model_utterance_path.exists():
        raise FileNotFoundError(f"Missing model_utterances file: {model_utterance_path}")

    stage1_requests = load_jsonl(stage1_request_path)
    model_utterances_payload = load_json(model_utterance_path)
    model_utterances = model_utterances_payload["utterances"]

    stage1_responses: list[dict[str, Any]] = []
    for row in stage1_requests:
        if args.stage1_mode == "mock":
            response_row = stage1_response_from_mock(row)
        else:
            if not args.stage1_url:
                raise RuntimeError("HTTP Stage 1 mode requires --stage1-url")
            raw = post_json(args.stage1_url, row)
            response_row = normalize_stage1_response(raw, row)
        stage1_responses.append(response_row)

    stage1_response_dir = args.stage1_response_root / args.meeting_id / f"v{args.version}"
    ensure_dir(stage1_response_dir)
    stage1_responses_jsonl = stage1_response_dir / "responses.jsonl"
    stage1_responses_json = stage1_response_dir / "responses.json"
    write_jsonl(stage1_responses_jsonl, stage1_responses)
    write_json(
        stage1_responses_json,
        {"meeting_id": args.meeting_id, "response_count": len(stage1_responses), "responses": stage1_responses},
    )

    stage1_response_map = {row["left_model_index"]: row for row in stage1_responses}
    stage2_inputs = build_stage2_inputs(
        args.meeting_id,
        model_utterances,
        stage1_response_map,
        boundary_threshold=args.stage1_threshold,
    )

    stage2_input_dir = args.stage2_input_root / args.meeting_id / f"v{args.version}"
    ensure_dir(stage2_input_dir)
    stage2_inputs_jsonl = stage2_input_dir / "stage2_inputs.jsonl"
    stage2_inputs_json = stage2_input_dir / "stage2_inputs.json"
    write_jsonl(stage2_inputs_jsonl, stage2_inputs)
    write_json(
        stage2_inputs_json,
        {"meeting_id": args.meeting_id, "input_count": len(stage2_inputs), "segments": stage2_inputs},
    )

    reconstructed_segments = build_reconstructed_segments(stage2_inputs)
    reconstructed_segments_path = args.segments_root / args.meeting_id / f"v{args.version}.json"
    write_json(reconstructed_segments_path, reconstructed_segments)

    stage2_outputs: list[dict[str, Any]] = []
    if args.stage2_mode != "skip":
        prompt_template = args.prompt_template.read_text(encoding="utf-8")
        for segment in stage2_inputs:
            if args.stage2_mode == "mock":
                output_row = stage2_response_from_mock(segment, args.model_version)
            else:
                output_row = call_flowise_for_segment(args, segment, prompt_template)
            stage2_outputs.append(output_row)

    stage2_response_dir = args.stage2_response_root / args.meeting_id / f"v{args.version}"
    ensure_dir(stage2_response_dir)
    stage2_responses_jsonl = stage2_response_dir / "responses.jsonl"
    stage2_responses_json = stage2_response_dir / "responses.json"
    if stage2_outputs:
        write_jsonl(stage2_responses_jsonl, stage2_outputs)
        write_json(
            stage2_responses_json,
            {"meeting_id": args.meeting_id, "response_count": len(stage2_outputs), "responses": stage2_outputs},
        )

    recap_path = args.recap_root / args.meeting_id / f"v{args.version}.json"
    if stage2_outputs:
        recap_payload = assemble_recap(args.meeting_id, stage2_inputs, stage2_outputs, args.model_version)
        write_json(recap_path, recap_payload)

    stage1_response_jsonl_key = f"{args.stage1_response_prefix.strip('/')}/{args.meeting_id}/v{args.version}/responses.jsonl"
    stage1_response_json_key = f"{args.stage1_response_prefix.strip('/')}/{args.meeting_id}/v{args.version}/responses.json"
    stage2_inputs_jsonl_key = f"{args.stage2_input_prefix.strip('/')}/{args.meeting_id}/v{args.version}/inputs.jsonl"
    stage2_inputs_json_key = f"{args.stage2_input_prefix.strip('/')}/{args.meeting_id}/v{args.version}/inputs.json"
    reconstructed_segments_key = f"{args.segments_prefix.strip('/')}/{args.meeting_id}/v{args.version}.json"
    stage2_response_jsonl_key = f"{args.stage2_response_prefix.strip('/')}/{args.meeting_id}/v{args.version}/responses.jsonl"
    stage2_response_json_key = f"{args.stage2_response_prefix.strip('/')}/{args.meeting_id}/v{args.version}/responses.json"
    recap_key = f"{args.recap_prefix.strip('/')}/{args.meeting_id}/v{args.version}.json"

    if args.upload_artifacts:
        upload_file(
            stage1_responses_jsonl,
            stage1_response_jsonl_key,
            args.rclone_remote,
            args.bucket,
            logger,
        )
        upload_file(
            stage1_responses_json,
            stage1_response_json_key,
            args.rclone_remote,
            args.bucket,
            logger,
        )
        upload_file(
            stage2_inputs_jsonl,
            stage2_inputs_jsonl_key,
            args.rclone_remote,
            args.bucket,
            logger,
        )
        upload_file(
            stage2_inputs_json,
            stage2_inputs_json_key,
            args.rclone_remote,
            args.bucket,
            logger,
        )
        upload_file(
            reconstructed_segments_path,
            reconstructed_segments_key,
            args.rclone_remote,
            args.bucket,
            logger,
        )
        if stage2_outputs:
            upload_file(
                stage2_responses_jsonl,
                stage2_response_jsonl_key,
                args.rclone_remote,
                args.bucket,
                logger,
            )
            upload_file(
                stage2_responses_json,
                stage2_response_json_key,
                args.rclone_remote,
                args.bucket,
                logger,
            )
            upload_file(
                recap_path,
                recap_key,
                args.rclone_remote,
                args.bucket,
                logger,
            )

    conn = get_conn()
    upsert_meeting_artifact(
        conn,
        args.meeting_id,
        "stage1_responses_jsonl",
        artifact_uri(stage1_responses_jsonl, stage1_response_jsonl_key, args.upload_artifacts),
        "application/x-ndjson",
        args.version,
    )
    upsert_meeting_artifact(
        conn,
        args.meeting_id,
        "stage1_responses_json",
        artifact_uri(stage1_responses_json, stage1_response_json_key, args.upload_artifacts),
        "application/json",
        args.version,
    )
    upsert_meeting_artifact(
        conn,
        args.meeting_id,
        "stage2_inputs_jsonl",
        artifact_uri(stage2_inputs_jsonl, stage2_inputs_jsonl_key, args.upload_artifacts),
        "application/x-ndjson",
        args.version,
    )
    upsert_meeting_artifact(
        conn,
        args.meeting_id,
        "stage2_inputs_json",
        artifact_uri(stage2_inputs_json, stage2_inputs_json_key, args.upload_artifacts),
        "application/json",
        args.version,
    )
    upsert_meeting_artifact(
        conn,
        args.meeting_id,
        "reconstructed_segments_json",
        artifact_uri(reconstructed_segments_path, reconstructed_segments_key, args.upload_artifacts),
        "application/json",
        args.version,
    )

    register_stage1_predictions(conn, args.meeting_id, stage1_requests, stage1_responses)

    if stage2_outputs:
        upsert_meeting_artifact(
            conn,
            args.meeting_id,
            "stage2_responses_jsonl",
            artifact_uri(stage2_responses_jsonl, stage2_response_jsonl_key, args.upload_artifacts),
            "application/x-ndjson",
            args.version,
        )
        upsert_meeting_artifact(
            conn,
            args.meeting_id,
            "stage2_responses_json",
            artifact_uri(stage2_responses_json, stage2_response_json_key, args.upload_artifacts),
            "application/json",
            args.version,
        )
        upsert_meeting_artifact(
            conn,
            args.meeting_id,
            "summary_json",
            artifact_uri(recap_path, recap_key, args.upload_artifacts),
            "application/json",
            args.version,
        )
        register_recap_outputs(
            conn,
            args.meeting_id,
            args.version,
            artifact_uri(recap_path, recap_key, args.upload_artifacts),
            stage2_inputs,
            stage2_outputs,
            args.model_version,
            args.prompt_template.name,
            args.stage2_mode,
        )

    conn.commit()

    logger.info(
        "Completed replay for meeting=%s stage1_responses=%d stage2_inputs=%d stage2_outputs=%d",
        args.meeting_id,
        len(stage1_responses),
        len(stage2_inputs),
        len(stage2_outputs),
    )


if __name__ == "__main__":
    main()
