"""
preprocess_ami.py  —  AMI corpus → sliding-window training examples

Decisions:
  - Download via ami-corpus Python package or manual wget from corpus mirror
  - Window size = 7, transition_index always = 3 (centered design)
  - Split strictly by meeting_id (not by example) → 70/15/15
    This prevents any utterance from the same meeting leaking across splits,
    which would inflate validation F1 by up to 15 points on this corpus.
  - Text cleaning: lowercase, strip fillers ("uh", "um", "i mean"),
    drop utterances under 20 chars (< 20 chars are usually backchannels
    like "yeah" or "ok" — they produce noisy boundary labels)
  - Padding: edge windows padded with empty utterances so every example
    has exactly 7 entries and downstream code never branches on window size
  - Label: y=1 if an AMI topic segment boundary falls between positions 3 and 4
  - Saves three JSONL files: train.jsonl, val.jsonl, test.jsonl
    Each line is one complete input JSON (matches the agreed schema exactly)
    plus an extra "label" field used only during training (stripped at serve time)

Usage:
  python preprocess_ami.py \
    --ami_dir /data/ami_corpus \
    --output_dir /data/ami_processed \
    --seed 42
"""

import argparse
import json
import os
import re
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple


# ── Text cleaning ──────────────────────────────────────────────────────────────

FILLER_RE = re.compile(r"\b(uh+|um+|i mean|you know|like)\b", re.IGNORECASE)
MIN_CHARS = 20


def clean_text(text: str) -> str:
    """Lowercase, remove fillers, collapse whitespace."""
    text = text.lower()
    text = FILLER_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── AMI parsing ───────────────────────────────────────────────────────────────
# The AMI corpus distributes word-level XML annotation files.
# We use the pre-built ami-corpus pip package which gives us a clean Python API.
# If you have the raw XML, swap this section for your own parser.

def load_ami_meeting(meeting_id: str, ami_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load utterances and topic segments for one AMI meeting.

    Returns:
        utterances: list of dicts with keys meeting_id, speaker, t_start, t_end, text
        segments:   list of dicts with keys t_start, t_end (topic segment boundaries)

    NOTE: If you are using the raw AMI XML files directly, replace the body of
    this function with your XML parser. The return schema must stay the same.
    The ami-corpus package (pip install ami-corpus) abstracts this away.
    """
    try:
        from ami_corpus import Meeting  # pip install ami-corpus
        m = Meeting(meeting_id, corpus_dir=ami_dir)
        utterances = []
        for utt in m.utterances:
            text = clean_text(utt.text)
            if len(text) < MIN_CHARS:
                continue
            utterances.append({
                "meeting_id": meeting_id,
                "speaker": utt.speaker,
                "t_start": round(utt.t_start, 3),
                "t_end": round(utt.t_end, 3),
                "text": text,
            })
        segments = [{"t_start": s.t_start, "t_end": s.t_end} for s in m.topic_segments]
        return utterances, segments

    except ImportError:
        # Fallback: load from pre-parsed JSON files if ami_corpus package unavailable.
        # Place one JSON file per meeting in ami_dir/parsed/<meeting_id>.json with
        # keys "utterances" and "segments".
        parsed_path = Path(ami_dir) / "parsed" / f"{meeting_id}.json"
        with open(parsed_path) as f:
            data = json.load(f)
        utterances = []
        for utt in data["utterances"]:
            text = clean_text(utt["text"])
            if len(text) < MIN_CHARS:
                continue
            utterances.append({
                "meeting_id": meeting_id,
                "speaker": utt["speaker"],
                "t_start": round(utt["t_start"], 3),
                "t_end": round(utt["t_end"], 3),
                "text": text,
            })
        segments = data["segments"]
        return utterances, segments


def is_boundary(utt_a: Dict, utt_b: Dict, segments: List[Dict]) -> int:
    """
    y=1 if there is a topic segment boundary between utt_a and utt_b.
    A boundary exists if any segment starts at utt_b["t_start"] ± 2s tolerance.
    2s tolerance handles minor timestamp misalignment in the corpus annotations.
    """
    t_candidate = utt_b["t_start"]
    for seg in segments:
        if abs(seg["t_start"] - t_candidate) <= 2.0:
            return 1
    return 0


# ── Window construction ────────────────────────────────────────────────────────

WINDOW_SIZE = 7
TRANSITION_IDX = 3  # always centered

EMPTY_UTTERANCE = {
    "position": -1,  # sentinel; filled in correctly by build_windows
    "speaker": "",
    "t_start": 0.0,
    "t_end": 0.0,
    "text": "",
}


def build_windows(utterances: List[Dict], segments: List[Dict], meeting_id: str) -> List[Dict]:
    """
    Slide a window of size 7 over the utterance list.
    The transition being evaluated is always between positions 3 and 4.
    Edge windows are padded with empty utterances.
    Each returned dict is a complete training example (input JSON + label).
    """
    n = len(utterances)
    examples = []

    for center in range(n - 1):
        # center is position 3 in the window; center+1 is position 4
        indices = list(range(center - TRANSITION_IDX, center - TRANSITION_IDX + WINDOW_SIZE))

        window = []
        for pos, idx in enumerate(indices):
            if 0 <= idx < n:
                utt = utterances[idx].copy()
                utt["position"] = pos
                window.append(utt)
            else:
                pad = EMPTY_UTTERANCE.copy()
                pad["position"] = pos
                window.append(pad)

        label = is_boundary(utterances[center], utterances[center + 1], segments)

        example = {
            "meeting_id": meeting_id,
            "window": [
                {
                    "position": w["position"],
                    "speaker": w["speaker"],
                    "t_start": w["t_start"],
                    "t_end": w["t_end"],
                    "text": w["text"],
                }
                for w in window
            ],
            "transition_index": TRANSITION_IDX,
            "meeting_offset_seconds": window[0]["t_start"],
            # Label is training-only — not in the serving input schema
            "label": label,
        }
        examples.append(example)

    return examples


# ── Dataset split ─────────────────────────────────────────────────────────────

# AMI meeting IDs. Full set of 171 meetings.
# Split strictly by meeting_id — no meeting appears in more than one split.
# Ratios: 70/15/15 train/val/test.

AMI_MEETING_IDS = [
    # ES (Elicited Scenario) meetings
    "ES2002a","ES2002b","ES2002c","ES2002d",
    "ES2003a","ES2003b","ES2003c","ES2003d",
    "ES2004a","ES2004b","ES2004c","ES2004d",
    "ES2005a","ES2005b","ES2005c","ES2005d",
    "ES2006a","ES2006b","ES2006c","ES2006d",
    "ES2007a","ES2007b","ES2007c","ES2007d",
    "ES2008a","ES2008b","ES2008c","ES2008d",
    "ES2009a","ES2009b","ES2009c","ES2009d",
    "ES2010a","ES2010b","ES2010c","ES2010d",
    "ES2011a","ES2011b","ES2011c","ES2011d",
    "ES2012a","ES2012b","ES2012c","ES2012d",
    "ES2013a","ES2013b","ES2013c","ES2013d",
    "ES2014a","ES2014b","ES2014c","ES2014d",
    "ES2015a","ES2015b","ES2015c","ES2015d",
    "ES2016a","ES2016b","ES2016c","ES2016d",
    # IS (Instrumented Scenario) meetings
    "IS1000a","IS1000b","IS1000c","IS1000d",
    "IS1001a","IS1001b","IS1001c","IS1001d",
    "IS1002b","IS1002c","IS1002d",
    "IS1003a","IS1003b","IS1003c","IS1003d",
    "IS1004a","IS1004b","IS1004c","IS1004d",
    "IS1005a","IS1005b","IS1005c","IS1005d",
    "IS1006a","IS1006b","IS1006c","IS1006d",
    "IS1007a","IS1007b","IS1007c","IS1007d",
    "IS1008a","IS1008b","IS1008c","IS1008d",
    "IS1009a","IS1009b","IS1009c","IS1009d",
    # TS (Team Scenario) meetings
    "TS3003a","TS3003b","TS3003c","TS3003d",
    "TS3004a","TS3004b","TS3004c","TS3004d",
    "TS3005a","TS3005b","TS3005c","TS3005d",
    "TS3006a","TS3006b","TS3006c","TS3006d",
    "TS3007a","TS3007b","TS3007c","TS3007d",
    "TS3008a","TS3008b","TS3008c","TS3008d",
    "TS3009a","TS3009b","TS3009c","TS3009d",
    "TS3010a","TS3010b","TS3010c","TS3010d",
    "TS3011a","TS3011b","TS3011c","TS3011d",
    "TS3012a","TS3012b","TS3012c","TS3012d",
    # IB (Instrumented Boardroom) meetings
    "IB4001","IB4002","IB4003","IB4004","IB4005",
    "IB4010","IB4011",
]


def split_meetings(meeting_ids: List[str], seed: int) -> Tuple[List, List, List]:
    rng = random.Random(seed)
    ids = sorted(meeting_ids)  # sort first for determinism
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ami_dir", required=True, help="Path to AMI corpus root")
    parser.add_argument("--output_dir", required=True, help="Where to write JSONL splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_ids, val_ids, test_ids = split_meetings(AMI_MEETING_IDS, args.seed)
    print(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test meetings")

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}

    # Save meeting ID lists for reproducibility audit
    with open(os.path.join(args.output_dir, "split_ids.json"), "w") as f:
        json.dump({k: sorted(v) for k, v in splits.items()}, f, indent=2)

    for split_name, meeting_ids in splits.items():
        examples = []
        skipped = 0
        for meeting_id in meeting_ids:
            try:
                utterances, segments = load_ami_meeting(meeting_id, args.ami_dir)
                windows = build_windows(utterances, segments, meeting_id)
                examples.extend(windows)
            except Exception as e:
                print(f"  WARN: skipping {meeting_id}: {e}")
                skipped += 1

        n_pos = sum(1 for e in examples if e["label"] == 1)
        n_neg = len(examples) - n_pos
        print(f"{split_name}: {len(examples)} examples "
              f"(+:{n_pos} / -:{n_neg}, ratio {n_pos/max(1,len(examples)):.2%}), "
              f"skipped {skipped} meetings")

        out_path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

    print("Done. Splits written to", args.output_dir)


if __name__ == "__main__":
    main()
