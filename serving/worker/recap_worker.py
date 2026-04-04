"""
Async recap worker.
Chains Stage A (segmentation) → Stage B (summarization) → writes recap JSON.
"""
import json
import time
import requests
import os

SEGMENT_URL   = os.getenv("SEGMENT_URL",   "http://api_pytorch:8000/segment")
SUMMARIZE_URL = os.getenv("SUMMARIZE_URL", "http://api_pytorch:8000/summarize")
OUTPUT_DIR    = os.getenv("OUTPUT_DIR",    "/outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_windows(utterances: list, window_size: int = 7) -> list:
    """Slide a window of size 7 over utterances, centered on each transition."""
    windows = []
    half = window_size // 2
    for i in range(len(utterances) - 1):
        start = max(0, i - half)
        end   = min(len(utterances), i + half + 1)
        window = utterances[start:end]

        # pad to window_size if near edges
        while len(window) < window_size:
            window.append({
                "position": len(window),
                "speaker": "",
                "t_start": 0.0,
                "t_end":   0.0,
                "text":    ""
            })

        # reindex positions 0..6
        window = [
            {**u, "position": j}
            for j, u in enumerate(window)
        ]

        windows.append({
            "transition_index":        half,
            "meeting_offset_seconds":  window[0]["t_start"],
            "window":                  window
        })
    return windows


def assemble_segments(utterances: list, boundary_outputs: list) -> list:
    """Group utterances into confirmed segments from boundary decisions."""
    segments   = []
    current    = []
    seg_id     = 1
    seg_start  = utterances[0]["t_start"] if utterances else 0.0

    for i, decision in enumerate(boundary_outputs):
        current.append(utterances[i])
        if decision["is_boundary"] or i == len(boundary_outputs) - 1:
            segments.append({
                "segment_id":  seg_id,
                "t_start":     seg_start,
                "t_end":       utterances[i]["t_end"],
                "utterances":  current,
                "total_utterances": len(current)
            })
            seg_id   += 1
            seg_start = utterances[i]["t_end"]
            current   = []

    return segments


def process_meeting(meeting_id: str, utterances: list):
    print(f"[worker] Processing {meeting_id} — {len(utterances)} utterances")
    t0 = time.perf_counter()

    # Stage A — segmentation
    windows  = build_windows(utterances)
    decisions = []
    for w in windows:
        payload = {"meeting_id": meeting_id, **w}
        r = requests.post(SEGMENT_URL, json=payload)
        r.raise_for_status()
        decisions.append(r.json())

    # assemble segments
    segments = assemble_segments(utterances, decisions)
    print(f"[worker] Found {len(segments)} segments")

    # Stage B — summarization
    summaries = []
    for seg in segments:
        payload = {
            "meeting_id":      meeting_id,
            "segment_id":      seg["segment_id"],
            "t_start":         seg["t_start"],
            "t_end":           seg["t_end"],
            "utterances":      seg["utterances"],
            "total_utterances": seg["total_utterances"],
            "meeting_context": {
                "total_segments":          len(segments),
                "segment_index_in_meeting": seg["segment_id"]
            }
        }
        try:
            r = requests.post(SUMMARIZE_URL, json=payload, timeout=120)
            r.raise_for_status()
            summaries.append(r.json())
        except Exception as e:
            print(f"[worker] Summarizer failed for segment {seg['segment_id']}: {e}")
            summaries.append({
                "meeting_id":      meeting_id,
                "segment_id":      seg["segment_id"],
                "t_start":         seg["t_start"],
                "t_end":           seg["t_end"],
                "topic_label":     "",
                "summary_bullets": [],
                "status":          "draft"
            })

    # assemble final recap JSON
    recap = {
        "meeting_id":     meeting_id,
        "generated_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_segments": len(summaries),
        "recap":          summaries
    }

    output_path = os.path.join(OUTPUT_DIR, f"{meeting_id}_recap.json")
    with open(output_path, "w") as f:
        json.dump(recap, f, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"[worker] Done. Recap saved to {output_path} in {elapsed:.1f}s")
    return recap


if __name__ == "__main__":
    # test with a small synthetic meeting
    test_utterances = [
        {"position": i, "speaker": ["A","B","C"][i%3],
         "t_start": float(i*10), "t_end": float(i*10+9),
         "text": f"utterance number {i} about topic {"alpha" if i < 5 else "beta"}"}
        for i in range(10)
    ]
    process_meeting("TEST001", test_utterances)