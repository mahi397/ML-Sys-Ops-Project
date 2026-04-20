#!/usr/bin/env python3
"""Create a lightly augmented Stage 1 training split.

This script is intentionally conservative:
- it reads only the existing Stage 1 train split
- it leaves val/test untouched
- it creates synthetic rows from real train rows only
- it writes a new dataset version layout, typically v2
- it uploads the completed v2 dataset to object storage

The augmentation is text-only and keeps the boundary label unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import random
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


FILLERS = ("um", "uh", "er", "erm")
LIGHT_TOKENS = {"a", "an", "the", "to", "of", "and"}
PUNCT_RE = re.compile(r"[.,!?;:\"()]")
MULTISPACE_RE = re.compile(r"\s+")

PHRASE_SWAPS = (
    ("okay", "ok"),
    ("kind of", "kinda"),
    ("sort of", "sorta"),
    ("going to", "gonna"),
    ("want to", "wanna"),
    ("got to", "gotta"),
    ("i am", "i'm"),
    ("do not", "don't"),
    ("cannot", "can't"),
)


def setup_logger(log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("augment_stage1_train")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/mnt/block/roberta_stage1/cache/v1"),
        help="Local cache directory for downloading the real-only Stage 1 v1 dataset from object storage.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/block/roberta_stage1/v2"),
        help="Output directory for the augmented dataset version.",
    )
    parser.add_argument(
        "--augment-ratio",
        type=float,
        default=0.3,
        help="Synthetic rows to add as a fraction of original train rows.",
    )
    parser.add_argument(
        "--synthetic-positive-fraction",
        type=float,
        default=0.6,
        help="Desired fraction of synthetic rows drawn from positive seeds.",
    )
    parser.add_argument(
        "--max-edits-per-row",
        type=int,
        default=2,
        help="Maximum non-padding utterances to perturb in one synthetic row.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("/mnt/block/roberta_stage1/logs/augment_stage1_train.log"),
        help="Optional log file path.",
    )
    parser.add_argument(
        "--skip-download-input",
        action="store_true",
        help="Use --input-root directly instead of downloading v1 from object storage first.",
    )
    parser.add_argument("--rclone-remote", default="rclone_s3")
    parser.add_argument("--object-bucket", default="objstore-proj07")
    parser.add_argument("--input-object-prefix", default="datasets/roberta_stage1/v1")
    parser.add_argument("--object-prefix", default="datasets/roberta_stage1/v2")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_counts(rows: list[dict]) -> dict[str, int]:
    positives = sum(row["output"]["label"] for row in rows)
    return {
        "rows": len(rows),
        "positive": positives,
        "negative": len(rows) - positives,
    }


def pick_examples(rows: list[dict]) -> dict[str, dict | None]:
    positive = next((row for row in rows if row["output"]["label"] == 1), None)
    negative = next((row for row in rows if row["output"]["label"] == 0), None)
    return {
        "positive": positive,
        "negative": negative,
    }


def normalize_whitespace(text: str) -> str:
    return MULTISPACE_RE.sub(" ", text).strip()


def strip_punctuation(text: str) -> str:
    updated = PUNCT_RE.sub(" ", text)
    return normalize_whitespace(updated)


def remove_fillers(text: str) -> str:
    tokens = text.split()
    kept = [token for token in tokens if token.lower().strip(",.!?;:") not in FILLERS]
    updated = " ".join(kept)
    return normalize_whitespace(updated)


def insert_filler(text: str, rng: random.Random) -> str:
    tokens = text.split()
    if len(tokens) < 4:
        return text
    insert_at = rng.randint(1, min(len(tokens) - 1, 6))
    filler = rng.choice(FILLERS)
    updated = tokens[:insert_at] + [filler] + tokens[insert_at:]
    return normalize_whitespace(" ".join(updated))


def drop_light_token(text: str, rng: random.Random) -> str:
    tokens = text.split()
    eligible = [
        idx for idx, token in enumerate(tokens)
        if token.lower().strip(",.!?;:") in LIGHT_TOKENS
    ]
    if not eligible:
        return text
    drop_idx = rng.choice(eligible)
    updated = tokens[:drop_idx] + tokens[drop_idx + 1:]
    return normalize_whitespace(" ".join(updated))


def swap_short_forms(text: str, rng: random.Random) -> str:
    candidates = [(src, dst) for src, dst in PHRASE_SWAPS if src in text]
    if not candidates:
        return text
    src, dst = rng.choice(candidates)
    return normalize_whitespace(text.replace(src, dst, 1))


def collapse_repeated_words(text: str) -> str:
    tokens = text.split()
    if not tokens:
        return text

    collapsed = [tokens[0]]
    for token in tokens[1:]:
        prev = collapsed[-1].lower().strip(",.!?;:")
        cur = token.lower().strip(",.!?;:")
        if prev == cur:
            continue
        collapsed.append(token)
    return normalize_whitespace(" ".join(collapsed))


TEXT_OPERATIONS = (
    ("strip_punctuation", lambda text, rng: strip_punctuation(text)),
    ("remove_fillers", lambda text, rng: remove_fillers(text)),
    ("insert_filler", insert_filler),
    ("drop_light_token", drop_light_token),
    ("swap_short_forms", swap_short_forms),
    ("collapse_repeated_words", lambda text, rng: collapse_repeated_words(text)),
)


def seed_row_id(row: dict) -> str:
    metadata = row.get("metadata", {})
    return "|".join(
        [
            str(row["input"]["meeting_id"]),
            str(metadata.get("left_model_utterance_id")),
            str(metadata.get("right_model_utterance_id")),
            str(row["output"]["label"]),
        ]
    )


def row_signature(row: dict) -> str:
    texts = [
        entry.get("text", "")
        for entry in row["input"]["window"]
    ]
    payload = {
        "seed_row_id": seed_row_id(row),
        "texts": texts,
        "label": row["output"]["label"],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def choose_positions(window: list[dict], transition_index: int, max_positions: int, rng: random.Random) -> list[int]:
    candidates: list[tuple[int, float]] = []
    for entry in window:
        if entry.get("is_padding"):
            continue
        if not entry.get("text", "").strip():
            continue
        position = entry["position"]
        distance = abs(position - transition_index)
        weight = 3.0 if distance <= 1 else 2.0 if distance == 2 else 1.0
        candidates.append((position, weight))

    if not candidates:
        return []

    target = rng.randint(1, min(max_positions, len(candidates)))
    selected: list[int] = []
    pool = candidates[:]

    while pool and len(selected) < target:
        positions = [pos for pos, _ in pool]
        weights = [weight for _, weight in pool]
        chosen = rng.choices(positions, weights=weights, k=1)[0]
        selected.append(chosen)
        pool = [(pos, weight) for pos, weight in pool if pos != chosen]

    return sorted(selected)


def augment_text(text: str, rng: random.Random) -> tuple[str, list[str]]:
    current = normalize_whitespace(text)
    if not current:
        return current, []

    op_count = rng.randint(1, 3)
    op_pool = list(TEXT_OPERATIONS)
    rng.shuffle(op_pool)

    applied: list[str] = []
    for name, fn in op_pool:
        updated = normalize_whitespace(fn(current, rng))
        if updated and updated != current:
            current = updated
            applied.append(name)
        if len(applied) >= op_count:
            break

    return current, applied


def augment_row(row: dict, rng: random.Random, synthetic_index: int, max_edits_per_row: int) -> tuple[dict | None, list[dict]]:
    synthetic = copy.deepcopy(row)
    transition_index = synthetic["input"]["transition_index"]
    positions = choose_positions(synthetic["input"]["window"], transition_index, max_edits_per_row, rng)
    if not positions:
        return None, []

    edits: list[dict] = []
    changed = False

    for position in positions:
        entry = next(item for item in synthetic["input"]["window"] if item["position"] == position)
        original_text = entry.get("text", "")
        updated_text, ops = augment_text(original_text, rng)
        if not ops or updated_text == original_text:
            continue

        entry["text"] = updated_text
        changed = True
        edits.append(
            {
                "position": position,
                "source_utterance_id": entry.get("source_utterance_id"),
                "model_utterance_id": entry.get("model_utterance_id"),
                "operations": ops,
                "original_text": original_text,
                "augmented_text": updated_text,
            }
        )

    if not changed:
        return None, []

    metadata = synthetic.setdefault("metadata", {})
    metadata["is_synthetic"] = True
    metadata["synthetic_id"] = f"synthetic_{synthetic_index:07d}"
    metadata["synthetic_parent_row_id"] = seed_row_id(row)
    metadata["augmentation"] = {
        "type": "text_only_seeded_window_augmentation",
        "edited_positions": edits,
    }

    return synthetic, edits


def copy_if_exists(src: Path, dst: Path, logger: logging.Logger) -> None:
    if not src.exists():
        logger.info("Skipping missing file: %s", src)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    logger.info("Copied %s -> %s", src, dst)


def build_rclone_target(remote: str, bucket: str, prefix: str) -> str:
    cleaned = prefix.strip("/")
    if cleaned:
        return f"{remote}:{bucket}/{cleaned}/"
    return f"{remote}:{bucket}/"


def ensure_rclone_available() -> None:
    if shutil.which("rclone") is None:
        raise RuntimeError("rclone is not installed or not in PATH")


def download_input_dataset(cache_root: Path, remote: str, bucket: str, prefix: str, logger: logging.Logger) -> str:
    ensure_rclone_available()

    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    source = build_rclone_target(remote, bucket, prefix)
    cmd = [
        "rclone",
        "copy",
        source,
        str(cache_root),
        "-P",
        "--stats",
        "10s",
        "--include",
        "train.jsonl",
        "--include",
        "val.jsonl",
        "--include",
        "test.jsonl",
        "--include",
        "split_info.json",
        "--include",
        "manifest.json",
    ]
    logger.info("Downloading input dataset: %s", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        logger.error("rclone stdout:\n%s", result.stdout)
        logger.error("rclone stderr:\n%s", result.stderr)
        raise RuntimeError("rclone input download failed")

    logger.info("Downloaded v1 dataset from %s into %s", source, cache_root)
    return source


def upload_dataset(output_root: Path, remote: str, bucket: str, prefix: str, logger: logging.Logger) -> str:
    ensure_rclone_available()

    target = build_rclone_target(remote, bucket, prefix)

    cmd = [
        "rclone",
        "copy",
        str(output_root),
        target,
        "-P",
        "--stats",
        "10s",
    ]
    logger.info("Uploading dataset: %s", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        logger.error("rclone stdout:\n%s", result.stdout)
        logger.error("rclone stderr:\n%s", result.stderr)
        raise RuntimeError("rclone upload failed")
    logger.info("Upload complete: %s", target)
    return target


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_file)
    rng = random.Random(args.seed)

    logger.info("============================================================")
    logger.info("Stage 1 train augmentation started")
    logger.info("input_root                  = %s", args.input_root)
    logger.info("output_root                 = %s", args.output_root)
    logger.info("input_object_target         = %s", build_rclone_target(args.rclone_remote, args.object_bucket, args.input_object_prefix))
    logger.info("augment_ratio               = %.3f", args.augment_ratio)
    logger.info("synthetic_positive_fraction = %.3f", args.synthetic_positive_fraction)
    logger.info("max_edits_per_row           = %d", args.max_edits_per_row)
    logger.info("seed                        = %d", args.seed)
    logger.info("============================================================")

    input_object_target = None
    if args.skip_download_input:
        logger.info("Using local input root directly without object-store download")
    else:
        input_object_target = download_input_dataset(
            cache_root=args.input_root,
            remote=args.rclone_remote,
            bucket=args.object_bucket,
            prefix=args.input_object_prefix,
            logger=logger,
        )

    train_path = args.input_root / "train.jsonl"
    val_path = args.input_root / "val.jsonl"
    test_path = args.input_root / "test.jsonl"
    split_info_path = args.input_root / "split_info.json"
    input_manifest_path = args.input_root / "manifest.json"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train split: {train_path}")

    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path) if val_path.exists() else []
    test_rows = read_jsonl(test_path) if test_path.exists() else []
    input_manifest = read_json(input_manifest_path) if input_manifest_path.exists() else {}

    logger.info("Loaded rows | train=%d val=%d test=%d", len(train_rows), len(val_rows), len(test_rows))

    target_synthetic_rows = round(len(train_rows) * args.augment_ratio)
    positive_rows = [row for row in train_rows if row["output"]["label"] == 1]
    negative_rows = [row for row in train_rows if row["output"]["label"] == 0]

    if not train_rows:
        raise RuntimeError("Input train split is empty")
    if not positive_rows or not negative_rows:
        raise RuntimeError("Need both positive and negative train rows for controlled augmentation")

    logger.info(
        "Seed pools | positive=%d negative=%d target_synthetic=%d",
        len(positive_rows),
        len(negative_rows),
        target_synthetic_rows,
    )

    seen_signatures = {row_signature(row) for row in train_rows}
    synthetic_rows: list[dict] = []
    op_counter: Counter[str] = Counter()
    rejected_no_change = 0
    rejected_duplicate = 0
    synthetic_positive_target = round(target_synthetic_rows * args.synthetic_positive_fraction)
    max_attempts = max(100, target_synthetic_rows * 25)

    attempts = 0
    while len(synthetic_rows) < target_synthetic_rows and attempts < max_attempts:
        attempts += 1
        want_positive = len([row for row in synthetic_rows if row["output"]["label"] == 1]) < synthetic_positive_target
        seed_pool = positive_rows if want_positive else negative_rows
        seed_row = rng.choice(seed_pool)

        synthetic_row, edits = augment_row(
            row=seed_row,
            rng=rng,
            synthetic_index=len(synthetic_rows) + 1,
            max_edits_per_row=args.max_edits_per_row,
        )
        if synthetic_row is None:
            rejected_no_change += 1
            continue

        signature = row_signature(synthetic_row)
        if signature in seen_signatures:
            rejected_duplicate += 1
            continue

        seen_signatures.add(signature)
        synthetic_rows.append(synthetic_row)
        for edit in edits:
            op_counter.update(edit["operations"])

    if len(synthetic_rows) < target_synthetic_rows:
        logger.warning(
            "Generated fewer synthetic rows than requested | requested=%d generated=%d attempts=%d",
            target_synthetic_rows,
            len(synthetic_rows),
            attempts,
        )

    final_train_rows = train_rows + synthetic_rows

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_root / "train_synthetic.jsonl", synthetic_rows)
    write_jsonl(args.output_root / "train_augmented.jsonl", final_train_rows)
    write_jsonl(args.output_root / "train.jsonl", final_train_rows)

    copy_if_exists(val_path, args.output_root / "val.jsonl", logger)
    copy_if_exists(test_path, args.output_root / "test.jsonl", logger)
    copy_if_exists(split_info_path, args.output_root / "split_info.json", logger)

    examples = {
        "train": pick_examples(final_train_rows),
        "val": pick_examples(val_rows),
        "test": pick_examples(test_rows),
        "train_synthetic": pick_examples(synthetic_rows),
    }
    write_json(args.output_root / "examples.json", examples)

    manifest = {
        "dataset_name": "roberta_stage1_boundary_augmented",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "base_dataset_root": str(args.input_root),
            "base_object_storage_target": input_object_target or build_rclone_target(args.rclone_remote, args.object_bucket, args.input_object_prefix),
            "base_manifest_created_at": input_manifest.get("created_at"),
            "base_source": input_manifest.get("source", {}),
        },
        "params": {
            "seed": args.seed,
            "augment_ratio": args.augment_ratio,
            "synthetic_positive_fraction": args.synthetic_positive_fraction,
            "max_edits_per_row": args.max_edits_per_row,
            "operations": [name for name, _ in TEXT_OPERATIONS],
            "train_only": True,
            "val_test_unchanged": True,
        },
        "base_rows": {
            "train": label_counts(train_rows),
            "val": label_counts(val_rows),
            "test": label_counts(test_rows),
        },
        "augmented_rows": {
            "train_synthetic": label_counts(synthetic_rows),
            "train_final": label_counts(final_train_rows),
            "val": label_counts(val_rows),
            "test": label_counts(test_rows),
        },
        "augmentation": {
            "target_synthetic_rows": target_synthetic_rows,
            "generated_synthetic_rows": len(synthetic_rows),
            "rejected_no_change": rejected_no_change,
            "rejected_duplicate": rejected_duplicate,
            "attempts": attempts,
            "operation_counts": dict(op_counter),
        },
        "artifacts": {
            "output_root": str(args.output_root),
            "train_synthetic": str(args.output_root / "train_synthetic.jsonl"),
            "train_augmented": str(args.output_root / "train_augmented.jsonl"),
            "train_final_alias": str(args.output_root / "train.jsonl"),
            "val": str(args.output_root / "val.jsonl"),
            "test": str(args.output_root / "test.jsonl"),
            "log_file": str(args.log_file),
        },
    }
    write_json(args.output_root / "manifest.json", manifest)

    uploaded_target = upload_dataset(
        output_root=args.output_root,
        remote=args.rclone_remote,
        bucket=args.object_bucket,
        prefix=args.object_prefix,
        logger=logger,
    )
    manifest["artifacts"]["object_storage_target"] = uploaded_target
    write_json(args.output_root / "manifest.json", manifest)

    logger.info("Synthetic rows generated: %d", len(synthetic_rows))
    logger.info("Final train rows: %d", len(final_train_rows))
    logger.info("Wrote augmented dataset version to %s", args.output_root)
    logger.info("Uploaded augmented dataset to %s", uploaded_target)


if __name__ == "__main__":
    main()
