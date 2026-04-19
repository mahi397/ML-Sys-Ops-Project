"""
retrain_watcher.py — Polls feedback store and triggers retraining

Adapted to Aneesh's data design:
  - Feedback is in `feedback_events` table (not feedback_corrections)
    with event_type in ('merge_segments', 'split_segment')
    and payloads in before_payload / after_payload JSONB columns
  - Watermark tracking via `retrain_log` table: we track the highest
    feedback_event_id consumed by each retrain run, and only count
    events above that watermark as "new"
  - Training datasets live in objstore-proj07 at
    datasets/roberta_stage1/vN/ and are registered in dataset_versions

Fixes applied vs previous version:
  - [FIX 3] DATABASE_URL default now uses proj07_sql_db (not recap_system)
  - [FIX 5] retrain_log table writes are soft-fail guarded
  - [FIX 5] get_high_watermark() falls back gracefully if retrain_log doesn't exist yet
  - General: clearer logging of which DB it's connecting to on startup

Runs as a long-lived container. Periodically checks:
  1. How many unconsumed boundary feedback events exist
  2. How long since the last retrain

If either threshold is met, triggers:
  Aneesh's retraining_dataset_runtime (build_feedback_pool.py + build_retraining_snapshot.py)
  → Mahima's retrain.py (Ray Train fault-tolerant training)
"""

import json
import os
import time
import logging
import subprocess
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config from environment ──
# [FIX 3] Default DB name corrected to proj07_sql_db (Aneesh's actual DB)
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://recap:changeme@postgres:5432/proj07_sql_db"
)
RETRAIN_THRESHOLD = int(os.environ.get("RETRAIN_THRESHOLD", "5"))     # 5 for demo, 500 for prod
CHECK_INTERVAL = int(os.environ.get("RETRAIN_CHECK_INTERVAL_SECONDS", "300"))
MAX_DAYS_BETWEEN_RETRAINS = int(os.environ.get("MAX_DAYS_BETWEEN_RETRAINS", "30"))

# Feedback event types that represent boundary corrections.
# These are the event_type values Shruti's API writes to feedback_events
# when users click "Remove boundary" or "Add boundary" in the recap UI.
BOUNDARY_FEEDBACK_TYPES = ("merge_segments", "split_segment")


def _get_conn():
    import psycopg2
    return psycopg2.connect(DATABASE_URL)


def get_high_watermark():
    """
    Get the highest feedback_event_id consumed by the last successful retrain.
    Uses the retrain_log table. Returns 0 if no retrain has ever run.

    [FIX 5] Soft-fail: if retrain_log doesn't exist yet (add_mlops_tables.sql
    not yet run), returns 0 and logs a warning instead of crashing.
    """
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT COALESCE(MAX(high_watermark_event_id), 0)
            FROM retrain_log
            WHERE passed_gates = TRUE
        """)
        wm = cur.fetchone()[0]
        cur.close()
        conn.close()
        return wm
    except Exception as e:
        # [FIX 5] Table may not exist on first run — treat as watermark=0
        log.warning(f"Could not read retrain_log (table may not exist yet): {e}. "
                    f"Treating watermark as 0.")
        return 0


def get_unconsumed_feedback_count():
    """
    Count boundary-correction feedback events above the last retrain watermark.

    Aneesh's feedback_events schema:
      feedback_event_id BIGSERIAL PK
      meeting_id TEXT
      event_type TEXT          -- 'merge_segments', 'split_segment', 'thumbs_up', etc.
      before_payload JSONB     -- e.g. {"transition_index": 3, "label": 1}
      after_payload JSONB      -- e.g. {"transition_index": 3, "label": 0}
      created_at TIMESTAMPTZ

    We only count events whose event_type indicates a boundary correction
    (merge or split), not thumbs-up/down or other interaction events.
    """
    watermark = get_high_watermark()
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*)
            FROM feedback_events
            WHERE event_type IN %s
              AND feedback_event_id > %s
        """, (BOUNDARY_FEEDBACK_TYPES, watermark))
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return count, watermark
    except Exception as e:
        log.error(f"Failed to query feedback count: {e}")
        return 0, watermark


def get_last_retrain_time():
    """
    Get the timestamp of the most recent retrain run.

    [FIX 5] Soft-fail if retrain_log doesn't exist yet.
    """
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("SELECT MAX(finished_at) FROM retrain_log")
        result = cur.fetchone()[0]
        cur.close()
        conn.close()
        return result
    except Exception as e:
        log.warning(f"Could not query retrain_log (table may not exist yet): {e}")
        return None


def get_current_max_feedback_id():
    """Get the current highest feedback_event_id for boundary corrections."""
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT COALESCE(MAX(feedback_event_id), 0)
            FROM feedback_events
            WHERE event_type IN %s
        """, (BOUNDARY_FEEDBACK_TYPES,))
        max_id = cur.fetchone()[0]
        cur.close()
        conn.close()
        return max_id
    except Exception as e:
        log.warning(f"Failed to get max feedback id: {e}")
        return 0


def get_latest_dataset_version():
    """
    Query Aneesh's dataset_versions table for the latest roberta_stage1 dataset.
    Returns the object_key path (e.g. 'datasets/roberta_stage1/v3/') or None.
    """
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT object_key, dataset_version_id
            FROM dataset_versions
            WHERE dataset_name = 'roberta_stage1'
            ORDER BY dataset_version_id DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return row[0], row[1]
        return None, None
    except Exception as e:
        log.warning(f"Failed to query dataset_versions: {e}")
        return None, None


def trigger_retrain(correction_count, watermark):
    """
    Trigger the retrain pipeline:
    1. Run Aneesh's retraining dataset build scripts
    2. Run retrain.py with Ray Train for fault-tolerant training
    """
    log.info("=" * 60)
    log.info("TRIGGERING RETRAIN PIPELINE")
    log.info("=" * 60)

    # Snapshot the current max feedback_event_id before we start.
    # This becomes the watermark written to retrain_log after the run.
    new_watermark = get_current_max_feedback_id()

    _log_audit("retrain_triggered", {
        "corrections_count": correction_count,
        "old_watermark": watermark,
        "new_watermark": new_watermark,
        "trigger_time": datetime.utcnow().isoformat(),
    })

    # ── Step 1: Build retraining dataset ──
    # Aneesh's retraining_dataset_runtime produces:
    #   - datasets/roberta_stage1_feedback_pool/vN/ (feedback examples)
    #   - datasets/roberta_stage1/vN/ (merged AMI + feedback snapshot)
    #   - Registers new version in dataset_versions table
    log.info("Step 1: Running retraining dataset build...")
    batch_scripts = [
        ("build_feedback_pool.py", "Build feedback pool from corrected meetings"),
        ("build_retraining_snapshot.py", "Build merged training snapshot"),
    ]
    for script_name, description in batch_scripts:
        try:
            log.info(f"  Running {description} ({script_name})...")
            result = subprocess.run(
                [sys.executable, script_name],
                capture_output=True, text=True, timeout=600,
                cwd="/app",
            )
            if result.returncode != 0:
                log.error(f"  {script_name} failed: {result.stderr[:500]}")
                _log_audit("retrain_batch_failed", {
                    "script": script_name, "stderr": result.stderr[:500]
                })
                # Don't abort — the retrain can still run on existing dataset
            else:
                log.info(f"  {script_name} completed successfully")
        except FileNotFoundError:
            log.warning(f"  {script_name} not found — skipping (will use existing dataset)")
        except subprocess.TimeoutExpired:
            log.error(f"  {script_name} timed out after 600s")

    # ── Resolve training data paths ──
    dataset_obj_key, dataset_version_id = get_latest_dataset_version()
    feedback_pool_dir = None
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT object_key FROM dataset_versions
            WHERE dataset_name = 'roberta_stage1_feedback_pool'
            ORDER BY dataset_version_id DESC LIMIT 1
        """)
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            feedback_pool_dir = row[0]
    except Exception:
        pass

    # ── Step 2: Retrain with Ray Train ──
    log.info("Step 2: Launching retrain with Ray Train...")
    retrain_env = os.environ.copy()
    if dataset_obj_key:
        retrain_env["DATASET_VERSION"] = str(dataset_version_id)
        log.info(f"  Using dataset: {dataset_obj_key} (version {dataset_version_id})")
    if feedback_pool_dir:
        retrain_env["FEEDBACK_DATA_DIR"] = feedback_pool_dir
        log.info(f"  Using feedback pool: {feedback_pool_dir}")
    # Pass new_watermark so retrain.py can record it in audit_log
    retrain_env["MAX_FEEDBACK_EVENT_ID"] = str(new_watermark)

    try:
        result = subprocess.run(
            [
                "docker", "compose",
                "--profile", "retrain",
                "run", "--rm",
                "-e", f"DATASET_VERSION={dataset_version_id or 'unknown'}",
                "-e", f"FEEDBACK_DATA_DIR={feedback_pool_dir or ''}",
                "-e", f"MAX_FEEDBACK_EVENT_ID={new_watermark}",
                "retrain-job",
                "python", "retrain.py",
            ],
            capture_output=False,
            timeout=7200,
            cwd="/project",
            env=retrain_env,
        )
        success = result.returncode == 0
        log.info(f"Retrain {'SUCCEEDED' if success else 'FAILED'} "
                 f"(exit code {result.returncode})")

        _log_retrain_completion(
            success=success,
            new_watermark=new_watermark,
            corrections_used=correction_count,
            dataset_version=str(dataset_version_id) if dataset_version_id else "ami-only",
        )

        return success
    except subprocess.TimeoutExpired:
        log.error("Retrain timed out after 2 hours")
        _log_audit("retrain_timeout", {})
        return False
    except Exception as e:
        log.error(f"Retrain error: {e}")
        _log_audit("retrain_error", {"error": str(e)})
        return False


def _log_retrain_completion(success, new_watermark, corrections_used, dataset_version):
    """
    Log retrain completion to retrain_log table.
    The high_watermark_event_id column tracks which feedback events
    were consumed — the watcher only counts events above this mark.

    [FIX 5] Soft-fail: if retrain_log doesn't exist yet, logs a warning
    instead of crashing. Run add_mlops_tables.sql to create it.
    """
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO retrain_log
                (dataset_version, corrections_used, passed_gates,
                 high_watermark_event_id, finished_at)
            VALUES (%s, %s, %s, %s, NOW())
        """, (dataset_version, corrections_used, success, new_watermark))
        conn.commit()
        cur.close()
        conn.close()
        log.info(f"Retrain logged: watermark={new_watermark}, "
                 f"corrections={corrections_used}, passed={success}")
    except Exception as e:
        # [FIX 5] Table may not exist — warn but don't crash
        log.warning(f"Failed to log retrain completion (run add_mlops_tables.sql if missing): {e}")


def _log_audit(event_type: str, details: dict):
    """
    [FIX 5] Soft-fail: audit_log may not exist on first run.
    """
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO audit_log (event_type, details) VALUES (%s, %s)",
            (event_type, json.dumps(details))
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        log.warning(f"Audit log write failed (table may not exist yet): {e}")


def main():
    log.info("Retrain watcher started")
    log.info(f"  Database:          {DATABASE_URL.split('@')[-1]}")   # log host/db, not password
    log.info(f"  Correction threshold: {RETRAIN_THRESHOLD}")
    log.info(f"  Check interval:    {CHECK_INTERVAL}s")
    log.info(f"  Max days between:  {MAX_DAYS_BETWEEN_RETRAINS}")
    log.info(f"  Feedback types:    {BOUNDARY_FEEDBACK_TYPES}")

    while True:
        try:
            count, watermark = get_unconsumed_feedback_count()
            last_retrain = get_last_retrain_time()
            days_since = None

            if last_retrain:
                days_since = (datetime.utcnow() - last_retrain).days

            trigger_reason = None

            if count >= RETRAIN_THRESHOLD:
                trigger_reason = (f"correction count ({count}) >= "
                                  f"threshold ({RETRAIN_THRESHOLD})")
            elif days_since is not None and days_since >= MAX_DAYS_BETWEEN_RETRAINS:
                trigger_reason = (f"days since last retrain ({days_since}) >= "
                                  f"max ({MAX_DAYS_BETWEEN_RETRAINS})")
            elif last_retrain is None and count > 0:
                trigger_reason = (f"first retrain with {count} correction(s) available")

            if trigger_reason:
                log.info(f"Retrain trigger: {trigger_reason}")
                trigger_retrain(count, watermark)
            else:
                log.info(f"No retrain needed: {count}/{RETRAIN_THRESHOLD} corrections "
                         f"above watermark {watermark}"
                         f"{f', {days_since}d since last retrain' if days_since else ''}")

        except Exception as e:
            log.error(f"Watcher loop error: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
