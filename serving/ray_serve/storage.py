"""
Local JSON file storage for recaps and feedback corrections
Acts as a drop-in replacement until Postgres is available
Swapping: replace RecapStore methods with Postgres queries, API stays the same

Files written to /data/ inside the container (mounted as a volume).
"""

import json
import os
import time
import threading
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
RECAPS_FILE      = DATA_DIR / "recaps.jsonl"
UTTERANCES_FILE  = DATA_DIR / "meeting_utterances.jsonl"
FEEDBACK_FILE    = DATA_DIR / "feedback_corrections.jsonl"


class RecapStore:
    """
    Thread-safe local JSON store.
    Each file is newline-delimited JSON (one record per line).
    """
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ── Recaps ──────────────────────────────────────────────────────

    def save_recap(self, meeting_id: str, model_version: str, segments_json: list):
        """Write recap row — mirrors Postgres recaps table."""
        record = {
            "recap_id": f"{meeting_id}_{int(time.time())}",
            "meeting_id": meeting_id,
            "model_version": model_version,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "segments_json": segments_json
        }
        with self._lock:
            with open(RECAPS_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
        return record["recap_id"]

    def get_recap(self, meeting_id: str) -> dict | None:
        """Return most recent recap for a meeting_id."""
        if not RECAPS_FILE.exists():
            return None
        result = None
        with self._lock:
            with open(RECAPS_FILE) as f:
                for line in f:
                    rec = json.loads(line.strip())
                    if rec["meeting_id"] == meeting_id:
                        result = rec  # keep overwriting — last one wins
        return result

    # ── Utterances ──────────────────────────────────────────────────

    def save_utterances(self, meeting_id: str, utterances: list, decisions: list):
        """
        Write per-utterance rows — mirrors Postgres meeting_utterances table.
        decisions: list of segmenter outputs (one per transition window).
        """
        rows = []
        for i, u in enumerate(utterances):
            # decisions has len = len(utterances) - 1
            # last utterance has no decision, mark as continuation
            if i < len(decisions):
                predicted_label = 1 if decisions[i].get("is_boundary") else 0
                boundary_confidence = decisions[i].get("boundary_probability", 0.0)
            else:
                predicted_label = 0
                boundary_confidence = 0.0

            rows.append({
                "meeting_id": meeting_id,
                "utterance_idx": i,
                "speaker": u.get("speaker", ""),
                "text": u.get("text", ""),
                "t_start": u.get("t_start", 0),
                "t_end": u.get("t_end", 0),
                "predicted_label": predicted_label,
                "boundary_confidence": boundary_confidence
            })

        with self._lock:
            with open(UTTERANCES_FILE, "a") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

    def get_utterances(self, meeting_id: str) -> list:
        """Return all utterances for a meeting, sorted by utterance_idx."""
        if not UTTERANCES_FILE.exists():
            return []
        rows = []
        with self._lock:
            with open(UTTERANCES_FILE) as f:
                for line in f:
                    rec = json.loads(line.strip())
                    if rec["meeting_id"] == meeting_id:
                        rows.append(rec)
        return sorted(rows, key=lambda r: r["utterance_idx"])

    # ── Feedback corrections ─────────────────────────────────────────

    def save_feedback(self, meeting_id: str, utterance_idx: int,
                      action: str, original_label: int) -> dict:
        """
        Write correction row — mirrors Postgres feedback_corrections table.
        action: 'remove_boundary' (y=1→0) or 'add_boundary' (y=0→1)
        """
        corrected_label = 0 if action == "remove_boundary" else 1
        record = {
            "correction_id": f"{meeting_id}_{utterance_idx}_{int(time.time())}",
            "meeting_id": meeting_id,
            "utterance_idx": utterance_idx,
            "original_label": original_label,
            "corrected_label": corrected_label,
            "action": action,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "used_in_retrain_version": None  # Aneesh's pipeline marks this
        }
        with self._lock:
            with open(FEEDBACK_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
        return record

    def count_pending_corrections(self) -> int:
        """How many corrections haven't been used in retraining yet."""
        if not FEEDBACK_FILE.exists():
            return 0
        count = 0
        with self._lock:
            with open(FEEDBACK_FILE) as f:
                for line in f:
                    rec = json.loads(line.strip())
                    if rec.get("used_in_retrain_version") is None:
                        count += 1
        return count

    def get_all_feedback(self) -> list:
        """Return all feedback corrections (for Aneesh's retraining pipeline)."""
        if not FEEDBACK_FILE.exists():
            return []
        rows = []
        with self._lock:
            with open(FEEDBACK_FILE) as f:
                for line in f:
                    rows.append(json.loads(line.strip()))
        return rows