#!/usr/bin/env python3
"""
emulate_production.py — Production data emulator for NeuralOps demo

Sends synthetic Jitsi meeting transcripts to the /recap endpoint, then injects
realistic user feedback corrections (merge/split boundary events) into Postgres.
This drives the full production loop:

  /recap → DB saves recap → feedback_events written → retrain_watcher fires

Usage (standalone):
  python scripts/emulate_production.py

Usage (docker compose):
  docker compose --profile emulated-traffic up -d traffic-generator

Environment variables:
  RECAP_URL          Serving API base URL  (default: http://serving-api:8000)
  DATABASE_URL       Postgres DSN          (default: postgresql://...)
  MEETING_COUNT      Meetings per batch    (default: 5, 0 = run forever)
  DELAY_SECONDS      Pause between meetings(default: 10)
  FEEDBACK_RATE      Fraction of segments corrected by "users" (default: 0.25)
  INGEST_TOKEN       Bearer token for /recap if auth is enabled (default: none)
  SEED               Random seed           (default: 42)
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("emulate_production")

# ─── config ───────────────────────────────────────────────────────────────────

RECAP_URL = os.environ.get("RECAP_URL", "http://serving-api:8000")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://proj07_user:proj07@postgres:5432/proj07_sql_db",
)
MEETING_COUNT = int(os.environ.get("MEETING_COUNT", "5"))
DELAY_SECONDS = float(os.environ.get("DELAY_SECONDS", "10"))
FEEDBACK_RATE = float(os.environ.get("FEEDBACK_RATE", "0.25"))
INGEST_TOKEN = os.environ.get("INGEST_TOKEN", "")
SEED = int(os.environ.get("SEED", "42"))

# ─── realistic meeting content ─────────────────────────────────────────────────

TOPIC_BLOCKS = [
    {
        "topic": "project status review",
        "utterances": [
            ("Alice", "Let's kick off with a quick status check on the ML pipeline."),
            ("Bob",   "The serving container has been stable since last Tuesday — no restarts."),
            ("Alice", "Good. What about the retraining job? Did the last run pass quality gates?"),
            ("Bob",   "Yes, Pk came in at 0.21 and F1 at 0.23, both within threshold."),
            ("Carol", "I verified the candidate alias is set in MLflow. Ready for promotion review."),
            ("Alice", "Let's schedule that for tomorrow once the team has had a chance to review the model card."),
        ],
    },
    {
        "topic": "feedback data quality",
        "utterances": [
            ("Bob",   "We have 82 new boundary corrections from last week's meetings."),
            ("Carol", "That's enough to rebuild the feedback pool but still below the production threshold."),
            ("Alice", "Are any of the corrections concentrated in single-speaker meetings?"),
            ("Bob",   "About a third of them, yes. That slice has historically lower Pk."),
            ("Carol", "We should check the slice metrics in MLflow after the next retrain."),
            ("Alice", "Agreed. I'll add a note to the model card review checklist."),
        ],
    },
    {
        "topic": "infrastructure costs",
        "utterances": [
            ("Dave",  "The GPU node has been running continuously since April. Block storage is at 40 percent."),
            ("Alice", "Are the MinIO ray-checkpoints growing unbounded?"),
            ("Dave",  "Ray keeps the last two checkpoints per run, so it stays bounded."),
            ("Carol", "Good. What about MLflow artifact storage on chi.tacc?"),
            ("Dave",  "Each model is around 400 MB. We have five versions so far — about 2 GB total."),
            ("Alice", "That's fine for now. Let's revisit when we approach the bucket limit."),
        ],
    },
    {
        "topic": "serving latency discussion",
        "utterances": [
            ("Bob",   "Segmentation p95 is 173 ms at concurrency 5 — well inside the 2 second SLA."),
            ("Carol", "Summarization is slower, but the async pipeline means users don't feel it."),
            ("Dave",  "I noticed the GPU memory metric in Grafana shows zero. Is that a bug?"),
            ("Bob",   "Yes, it's a known issue — MetricsDeployment runs in a separate Ray worker process."),
            ("Carol", "The actual usage visible in nvidia-smi is around 6 GB, which is expected."),
            ("Alice", "We should switch to pynvml for system-wide GPU queries before the final demo."),
        ],
    },
    {
        "topic": "data pipeline health",
        "utterances": [
            ("Carol", "Stage 1 forward service processed 200 requests overnight without errors."),
            ("Dave",  "The drift monitor ran at 3 AM and found no significant feature drift."),
            ("Alice", "Good. How many meetings were marked valid in the last 24 hours?"),
            ("Carol", "Fourteen Jitsi meetings — all passed the utterance and stage artifact checks."),
            ("Bob",   "That brings the unconsumed valid meeting count to 32, below the retraining threshold."),
            ("Dave",  "We'll hit the threshold after the next busy meeting day."),
        ],
    },
    {
        "topic": "model promotion process",
        "utterances": [
            ("Alice", "The promotion workflow is: retrain passes gates, candidate alias is set automatically."),
            ("Bob",   "Then a team member reviews the model card in MLflow and manually sets the production alias."),
            ("Carol", "The serving layer polls the registry every 5 minutes and hot-reloads without a restart."),
            ("Dave",  "What triggers rollback if the promoted model degrades in production?"),
            ("Alice", "Online correction rate. If it exceeds 15 percent, we roll back to the fallback alias."),
            ("Bob",   "The retrain log and audit log both capture the watermark and gate results for each run."),
        ],
    },
    {
        "topic": "fairness evaluation review",
        "utterances": [
            ("Carol", "The fairness gate checks six slices: short, medium, and long meetings plus speaker count."),
            ("Dave",  "Single-speaker meetings have the worst Pk because there's no speaker-change signal."),
            ("Alice", "That's expected. The gate threshold for slices is 0.40, higher than the aggregate 0.25."),
            ("Bob",   "Speaker relabeling invariance also passed — Pk didn't degrade when we renamed speakers."),
            ("Carol", "That confirms the model is reading content, not just latching onto speaker identity tokens."),
            ("Alice", "Good. All of this is documented in the model card logged to MLflow."),
        ],
    },
    {
        "topic": "next sprint planning",
        "utterances": [
            ("Dave",  "For the next sprint I want to add pynvml-based GPU metrics to the Grafana dashboard."),
            ("Alice", "I'd like to increase the retrain threshold back to 500 after the demo."),
            ("Bob",   "We should also look at adding a second feedback type — thumbs up on segments."),
            ("Carol", "That could lower the correction rate metric and help distinguish good boundaries."),
            ("Dave",  "I can wire it up on the serving side once we agree on the event schema."),
            ("Alice", "Let's draft the schema in the contracts folder and review it async."),
        ],
    },
]


def make_meeting_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"jitsi_{ts}_{uuid.uuid4().hex[:8]}"


def build_utterances(rng: random.Random) -> list[dict]:
    """Pick 2–3 topic blocks and expand into a flat utterance list with timestamps."""
    blocks = rng.sample(TOPIC_BLOCKS, k=rng.randint(2, 3))
    utterances: list[dict] = []
    t = 0.0
    for block in blocks:
        for speaker, text in block["utterances"]:
            duration = rng.uniform(3.0, 8.0)
            utterances.append({
                "speaker": speaker,
                "text": text,
                "t_start": round(t, 1),
                "t_end": round(t + duration, 1),
            })
            t += duration + rng.uniform(0.5, 2.0)
    return utterances


def post_recap(meeting_id: str, utterances: list[dict]) -> dict[str, Any] | None:
    payload = {"meeting_id": meeting_id, "utterances": utterances}
    headers = {"Content-Type": "application/json"}
    if INGEST_TOKEN:
        headers["Authorization"] = f"Bearer {INGEST_TOKEN}"
    try:
        resp = requests.post(
            f"{RECAP_URL}/recap",
            json=payload,
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        log.error("POST /recap failed for %s: %s", meeting_id, exc)
        return None


def inject_feedback(meeting_id: str, recap: dict, rng: random.Random) -> int:
    """
    Simulate users correcting boundary decisions.
    Randomly flips some segment boundaries and writes feedback_events to Postgres.
    Returns number of corrections inserted.
    """
    try:
        import psycopg2
    except ImportError:
        log.warning("psycopg2 not available — skipping feedback injection")
        return 0

    segments: list[dict] = recap.get("recap", [])
    if len(segments) < 2:
        return 0

    corrections = 0
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        for i, seg in enumerate(segments[:-1]):
            if rng.random() > FEEDBACK_RATE:
                continue
            # Randomly merge (remove boundary) or split (add boundary)
            event_type = rng.choice(["merge_segments", "split_segment"])
            before = {"segment_id": seg.get("segment_id", i), "t_end": seg.get("t_end", 0)}
            after  = {"segment_id": seg.get("segment_id", i), "corrected": True}
            try:
                cur.execute(
                    """
                    INSERT INTO feedback_events
                        (meeting_id, event_type, before_payload, after_payload, created_at)
                    VALUES (%s, %s, %s, %s, NOW())
                    """,
                    (
                        meeting_id,
                        event_type,
                        json.dumps(before),
                        json.dumps(after),
                    ),
                )
                corrections += 1
            except Exception as e:
                log.warning("feedback_events insert failed: %s", e)
                conn.rollback()
                break
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        log.warning("Could not connect to Postgres for feedback injection: %s", e)

    return corrections


def wait_for_serving(max_wait: int = 120) -> bool:
    log.info("Waiting for serving API at %s ...", RECAP_URL)
    for _ in range(max_wait // 5):
        try:
            r = requests.get(f"{RECAP_URL}/health", timeout=5)
            if r.status_code == 200:
                log.info("Serving API is up: %s", r.json())
                return True
        except Exception:
            pass
        time.sleep(5)
    log.error("Serving API did not become ready within %ds", max_wait)
    return False


def run_batch(rng: random.Random, batch_size: int) -> None:
    total_corrections = 0
    for i in range(batch_size):
        meeting_id = make_meeting_id()
        utterances = build_utterances(rng)
        log.info(
            "[%d/%d] Sending meeting %s (%d utterances)",
            i + 1, batch_size, meeting_id, len(utterances),
        )

        recap = post_recap(meeting_id, utterances)
        if recap is None:
            log.warning("  Skipping feedback injection — recap failed")
        else:
            segments = len(recap.get("recap", []))
            duration = recap.get("processing_time_seconds", "?")
            warnings = recap.get("warnings", [])
            log.info(
                "  Recap OK: %d segments, %.1fs processing%s",
                segments,
                duration if isinstance(duration, float) else 0,
                f", warnings={warnings}" if warnings else "",
            )

            corrections = inject_feedback(meeting_id, recap, rng)
            total_corrections += corrections
            if corrections:
                log.info("  Injected %d feedback correction(s)", corrections)

        if i < batch_size - 1:
            time.sleep(DELAY_SECONDS)

    log.info(
        "Batch complete: %d meetings sent, %d total feedback corrections injected",
        batch_size, total_corrections,
    )


def main() -> None:
    log.info("Production data emulator starting")
    log.info("  RECAP_URL:      %s", RECAP_URL)
    log.info("  MEETING_COUNT:  %s", MEETING_COUNT if MEETING_COUNT > 0 else "infinite")
    log.info("  DELAY_SECONDS:  %s", DELAY_SECONDS)
    log.info("  FEEDBACK_RATE:  %s", FEEDBACK_RATE)

    if not wait_for_serving():
        sys.exit(1)

    rng = random.Random(SEED)

    if MEETING_COUNT > 0:
        run_batch(rng, MEETING_COUNT)
    else:
        # Continuous mode: run forever in batches of 5
        batch = 0
        while True:
            batch += 1
            log.info("--- Continuous batch %d ---", batch)
            run_batch(rng, 5)
            log.info("Sleeping %ds before next batch...", int(DELAY_SECONDS * 3))
            time.sleep(DELAY_SECONDS * 3)


if __name__ == "__main__":
    main()
