"""
retrain_watcher.py — Polls feedback store and triggers retraining

Runs as a long-lived container. Periodically checks:
  1. How many unused feedback corrections have accumulated
  2. How long since the last retrain

If either threshold is met, triggers the retrain pipeline:
  batch_compile.py (Aneesh's) → retrain.py (Mahima's)

This is the automation that makes the system run "with minimal human
intervention" as required by the project guidelines.
"""

import json
import os
import time
import logging
import subprocess
import sys
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config from environment ──
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://recap:changeme@postgres:5432/recap_system")
RETRAIN_THRESHOLD = int(os.environ.get("RETRAIN_THRESHOLD", "5"))  # 5 for demo, 500 for prod
CHECK_INTERVAL = int(os.environ.get("RETRAIN_CHECK_INTERVAL_SECONDS", "300"))  # 5 minutes
MAX_DAYS_BETWEEN_RETRAINS = int(os.environ.get("MAX_DAYS_BETWEEN_RETRAINS", "30"))


def get_unused_correction_count():
    """Count feedback corrections not yet consumed by a retrain run."""
    import psycopg2
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM feedback_corrections WHERE used_in_retrain_version IS NULL"
        )
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return count
    except Exception as e:
        log.error(f"Failed to query feedback count: {e}")
        return 0


def get_last_retrain_time():
    """Get the timestamp of the most recent retrain run."""
    import psycopg2
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT MAX(finished_at) FROM retrain_log")
        result = cur.fetchone()[0]
        cur.close()
        conn.close()
        return result
    except Exception as e:
        log.warning(f"Failed to query retrain log: {e}")
        return None


def trigger_retrain():
    """
    Trigger the retrain pipeline:
    1. Run batch_compile.py to build a versioned training dataset
    2. Run retrain.py with Ray Train for fault-tolerant training
    """
    log.info("=" * 60)
    log.info("TRIGGERING RETRAIN PIPELINE")
    log.info("=" * 60)

    # Log the trigger to audit table
    _log_audit("retrain_triggered", {
        "corrections_count": get_unused_correction_count(),
        "trigger_time": datetime.utcnow().isoformat(),
    })

    # Step 1: Batch compile (builds versioned training dataset from AMI + feedback)
    # In production this would be: docker compose --profile retrain run batch-pipeline
    # Here we call it as a subprocess since we're in the same container network
    log.info("Step 1: Running batch dataset compilation...")
    try:
        result = subprocess.run(
            [sys.executable, "batch_compile.py"],
            capture_output=True, text=True, timeout=600,
            cwd="/app",  # or wherever batch_compile.py lives
        )
        if result.returncode != 0:
            log.error(f"Batch compile failed: {result.stderr}")
            _log_audit("retrain_batch_failed", {"stderr": result.stderr[:500]})
            return False
        log.info("Batch compilation complete")
    except FileNotFoundError:
        log.warning("batch_compile.py not found — skipping (Aneesh may not have deployed yet)")
    except subprocess.TimeoutExpired:
        log.error("Batch compile timed out after 600s")
        return False

    # Step 2: Retrain with Ray Train
    log.info("Step 2: Launching retrain with Ray Train...")
    try:
        result = subprocess.run(
            [sys.executable, "retrain.py"],
            capture_output=False,  # let retrain output stream to our logs
            timeout=7200,  # 2 hour max for training
            cwd="/app",
        )
        success = result.returncode == 0
        log.info(f"Retrain {'SUCCEEDED' if success else 'FAILED'} (exit code {result.returncode})")

        # Log completion
        _log_retrain_completion(success)

        return success
    except subprocess.TimeoutExpired:
        log.error("Retrain timed out after 2 hours")
        _log_audit("retrain_timeout", {})
        return False
    except Exception as e:
        log.error(f"Retrain error: {e}")
        _log_audit("retrain_error", {"error": str(e)})
        return False


def _log_retrain_completion(success: bool):
    """Log retrain completion to retrain_log table."""
    import psycopg2
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO retrain_log (passed_gates, finished_at)
               VALUES (%s, NOW())""",
            (success,)
        )
        # Mark consumed corrections
        if success:
            version_tag = f"retrain-{int(time.time())}"
            cur.execute(
                """UPDATE feedback_corrections
                   SET used_in_retrain_version = %s
                   WHERE used_in_retrain_version IS NULL""",
                (version_tag,)
            )
            log.info(f"Marked corrections as consumed with version: {version_tag}")
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        log.warning(f"Failed to log retrain completion: {e}")


def _log_audit(event_type: str, details: dict):
    import psycopg2
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO audit_log (event_type, details) VALUES (%s, %s)",
            (event_type, json.dumps(details))
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        log.warning(f"Audit log write failed: {e}")


def main():
    log.info(f"Retrain watcher started")
    log.info(f"  Correction threshold: {RETRAIN_THRESHOLD}")
    log.info(f"  Check interval: {CHECK_INTERVAL}s")
    log.info(f"  Max days between retrains: {MAX_DAYS_BETWEEN_RETRAINS}")

    while True:
        try:
            count = get_unused_correction_count()
            last_retrain = get_last_retrain_time()
            days_since = None

            if last_retrain:
                days_since = (datetime.utcnow() - last_retrain).days

            trigger_reason = None

            if count >= RETRAIN_THRESHOLD:
                trigger_reason = f"correction count ({count}) >= threshold ({RETRAIN_THRESHOLD})"
            elif days_since is not None and days_since >= MAX_DAYS_BETWEEN_RETRAINS:
                trigger_reason = f"days since last retrain ({days_since}) >= max ({MAX_DAYS_BETWEEN_RETRAINS})"
            elif last_retrain is None and count > 0:
                # First retrain ever — trigger if we have any corrections at all
                trigger_reason = f"first retrain with {count} correction(s) available"

            if trigger_reason:
                log.info(f"Retrain trigger: {trigger_reason}")
                trigger_retrain()
            else:
                log.info(f"No retrain needed: {count}/{RETRAIN_THRESHOLD} corrections"
                         f"{f', {days_since} days since last retrain' if days_since else ''}")

        except Exception as e:
            log.error(f"Watcher loop error: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
