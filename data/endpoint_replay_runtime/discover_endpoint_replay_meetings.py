#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage1-root",
        type=Path,
        default=Path("/mnt/block/user-behaviour/online_inference/stage1"),
    )
    parser.add_argument("--version", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.stage1_root.exists():
        return

    for meeting_dir in sorted(args.stage1_root.iterdir()):
        if not meeting_dir.is_dir():
            continue
        request_path = meeting_dir / f"v{args.version}" / "stage1_requests.jsonl"
        model_path = meeting_dir / f"v{args.version}" / "model_utterances.json"
        if request_path.exists() and model_path.exists():
            print(meeting_dir.name)


if __name__ == "__main__":
    main()
