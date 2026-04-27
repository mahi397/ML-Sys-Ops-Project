"""
online_eval.py — Online evaluation for the jitsi-topic-segmenter

Queries production data from PostgreSQL to compute online quality metrics
based on user feedback corrections. Logs results to MLflow under the
'online-evaluation' experiment.

Metrics computed:
  - correction_rate: corrections per meeting (proxy for model error rate)
  - online_fpr: false positive rate (merge_segments events / total predictions)
  - online_fnr: false negative rate (split_segment events / total predictions)
  - corrections_per_model_version: tracks quality over model lifecycle

Run:
  - Manually: python online_eval.py
  - On a schedule: add to cron or docker compose as a periodic job
  - After promotion: python online_eval.py --days 7  (last 7 days)

Usage:
  python online_eval.py                          # evaluate last 30 days
  python online_eval.py --days 7                 # evaluate last 7 days
  python online_eval.py --model_version v3       # evaluate specific version
  python online_eval.py --since 2026-04-01       # evaluate since date
"""

import argparse
import json
import logging
import os
import time
import tempfile
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "experiment_name": "online-evaluation",
    "model_registry_name": "jitsi-topic-segmenter",
    "lookback_days": 30,
    # Alert thresholds — if exceeded, logged as warnings
    "alert_correction_rate": 0.15,   # > 15% of transitions corrected = degradation signal
    "alert_fpr": 0.10,               # > 10% false positive rate
    "alert_fnr": 0.10,               # > 10% false negative rate
}

def _default_database_url() -> str:
    user = os.environ.get("POSTGRES_USER", "proj07_user")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    host = os.environ.get("POSTGRES_HOST", "postgres")
    port = os.environ.get("POSTGRES_PORT", "5432")
    database = os.environ.get("POSTGRES_DB", "proj07_sql_db")
    auth = f"{user}:{password}@" if password else f"{user}@"
    return f"postgresql://{auth}{host}:{port}/{database}"


DATABASE_URL = os.environ.get("DATABASE_URL") or _default_database_url()


def _get_conn():
    import psycopg2
    return psycopg2.connect(DATABASE_URL)


# ── Queries ───────────────────────────────────────────────────────────────────

def get_overall_correction_rate(since: datetime, until: datetime) -> Dict:
    """
    Compute the overall correction rate across all meetings in the window.
    correction_rate = corrections / total_predictions
    """
    conn = _get_conn()
    cur = conn.cursor()

    # Total user boundary corrections in window
    cur.execute("""
        SELECT COUNT(*) 
        FROM feedback_events
        WHERE event_type IN ('merge_segments', 'split_segment')
          AND event_source = 'user'
          AND created_at BETWEEN %s AND %s
    """, (since, until))
    total_corrections = cur.fetchone()[0]

    # Total predictions made (utterance transitions with a prediction)
    cur.execute("""
        SELECT COUNT(*)
        FROM utterance_transitions ut
        JOIN meetings m USING (meeting_id)
        WHERE m.ended_at BETWEEN %s AND %s
          AND ut.pred_boundary_label IS NOT NULL
    """, (since, until))
    total_predictions = cur.fetchone()[0]

    # Meetings with at least one correction
    cur.execute("""
        SELECT COUNT(DISTINCT meeting_id)
        FROM feedback_events
        WHERE event_type IN ('merge_segments', 'split_segment')
          AND event_source = 'user'
          AND created_at BETWEEN %s AND %s
    """, (since, until))
    meetings_with_corrections = cur.fetchone()[0]

    # Total meetings in window
    cur.execute("""
        SELECT COUNT(DISTINCT meeting_id)
        FROM meetings
        WHERE ended_at BETWEEN %s AND %s
    """, (since, until))
    total_meetings = cur.fetchone()[0]

    cur.close()
    conn.close()

    correction_rate = total_corrections / max(total_predictions, 1)
    meeting_correction_rate = meetings_with_corrections / max(total_meetings, 1)

    return {
        "total_corrections": total_corrections,
        "total_predictions": total_predictions,
        "total_meetings": total_meetings,
        "meetings_with_corrections": meetings_with_corrections,
        "correction_rate": round(correction_rate, 4),
        "meeting_correction_rate": round(meeting_correction_rate, 4),
    }


def get_fpr_fnr(since: datetime, until: datetime) -> Dict:
    """
    Compute approximate online false positive and false negative rates.

    merge_segments events = user removed a boundary the model predicted
                          = model false positive (pred=1, gold=0)
    split_segment events  = user added a boundary the model missed
                          = model false negative (pred=0, gold=1)
    """
    conn = _get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT event_type, COUNT(*) as cnt
        FROM feedback_events
        WHERE event_type IN ('merge_segments', 'split_segment')
          AND event_source = 'user'
          AND created_at BETWEEN %s AND %s
        GROUP BY event_type
    """, (since, until))
    rows = {row[0]: row[1] for row in cur.fetchall()}

    # Total predicted positive boundaries in window
    cur.execute("""
        SELECT COUNT(*)
        FROM utterance_transitions ut
        JOIN meetings m USING (meeting_id)
        WHERE m.ended_at BETWEEN %s AND %s
          AND ut.pred_boundary_label = TRUE
    """, (since, until))
    predicted_positives = cur.fetchone()[0]

    # Total predicted negative (non-boundary)
    cur.execute("""
        SELECT COUNT(*)
        FROM utterance_transitions ut
        JOIN meetings m USING (meeting_id)
        WHERE m.ended_at BETWEEN %s AND %s
          AND ut.pred_boundary_label = FALSE
    """, (since, until))
    predicted_negatives = cur.fetchone()[0]

    cur.close()
    conn.close()

    false_positives = rows.get("merge_segments", 0)
    false_negatives = rows.get("split_segment", 0)

    # FPR = false positives / predicted positives (of things we said were boundaries, how many were wrong)
    # FNR = false negatives / predicted negatives (of things we said weren't boundaries, how many were wrong)
    fpr = false_positives / max(predicted_positives, 1)
    fnr = false_negatives / max(predicted_negatives, 1)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "predicted_positives": predicted_positives,
        "predicted_negatives": predicted_negatives,
        "online_fpr": round(fpr, 4),
        "online_fnr": round(fnr, 4),
    }


def get_corrections_by_model_version(since: datetime, until: datetime) -> List[Dict]:
    """
    Break down correction rates by model version.
    Requires utterance_transitions to have a model_version column or
    that we can join through meetings.dataset_version.
    Falls back to grouping by week if model version not available.
    """
    conn = _get_conn()
    cur = conn.cursor()

    # Try to get per-model-version breakdown via meetings.dataset_version
    try:
        cur.execute("""
            SELECT 
                COALESCE(m.dataset_version::text, 'unknown') as model_version,
                COUNT(fe.feedback_event_id) as corrections,
                COUNT(DISTINCT fe.meeting_id) as meetings,
                MIN(fe.created_at) as first_correction,
                MAX(fe.created_at) as last_correction
            FROM feedback_events fe
            JOIN meetings m USING (meeting_id)
            WHERE fe.event_type IN ('merge_segments', 'split_segment')
              AND fe.event_source = 'user'
              AND fe.created_at BETWEEN %s AND %s
            GROUP BY m.dataset_version
            ORDER BY m.dataset_version
        """, (since, until))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        return [
            {
                "model_version": row[0],
                "corrections": row[1],
                "meetings": row[2],
                "first_correction": row[3].isoformat() if row[3] else None,
                "last_correction": row[4].isoformat() if row[4] else None,
                "corrections_per_meeting": round(row[1] / max(row[2], 1), 4),
            }
            for row in rows
        ]
    except Exception as e:
        log.warning(f"Could not get per-version breakdown: {e}")
        cur.close()
        conn.close()
        return []


def get_correction_trend(since: datetime, until: datetime, interval_days: int = 7) -> List[Dict]:
    """
    Week-over-week correction rate to detect degradation trends.
    """
    conn = _get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT 
            DATE_TRUNC('week', created_at) as week,
            COUNT(*) as corrections,
            COUNT(DISTINCT meeting_id) as meetings
        FROM feedback_events
        WHERE event_type IN ('merge_segments', 'split_segment')
          AND event_source = 'user'
          AND created_at BETWEEN %s AND %s
        GROUP BY DATE_TRUNC('week', created_at)
        ORDER BY week
    """, (since, until))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {
            "week": row[0].isoformat() if row[0] else None,
            "corrections": row[1],
            "meetings": row[2],
            "corrections_per_meeting": round(row[1] / max(row[2], 1), 4),
        }
        for row in rows
    ]


def get_retrain_log_summary() -> List[Dict]:
    """Get recent retrain history for context."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT retrain_id, dataset_version, corrections_used, passed_gates,
               high_watermark_event_id, finished_at
        FROM retrain_log
        ORDER BY retrain_id DESC
        LIMIT 10
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "retrain_id": row[0],
            "dataset_version": row[1],
            "corrections_used": row[2],
            "passed_gates": row[3],
            "watermark": row[4],
            "finished_at": row[5].isoformat() if row[5] else None,
        }
        for row in rows
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def run_online_eval(cfg: Dict, since: datetime, until: datetime) -> Dict:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(cfg["experiment_name"])

    log.info(f"Online evaluation window: {since.date()} → {until.date()}")

    # Gather metrics
    log.info("Computing correction rates...")
    correction_stats = get_overall_correction_rate(since, until)

    log.info("Computing FPR/FNR...")
    error_stats = get_fpr_fnr(since, until)

    log.info("Getting per-version breakdown...")
    version_breakdown = get_corrections_by_model_version(since, until)

    log.info("Getting weekly trend...")
    trend = get_correction_trend(since, until)

    log.info("Getting retrain history...")
    retrain_history = get_retrain_log_summary()

    # Check alerts
    alerts = []
    if correction_stats["correction_rate"] > cfg["alert_correction_rate"]:
        alerts.append(
            f"correction_rate={correction_stats['correction_rate']:.4f} "
            f"> threshold={cfg['alert_correction_rate']} — possible model degradation"
        )
    if error_stats["online_fpr"] > cfg["alert_fpr"]:
        alerts.append(
            f"online_fpr={error_stats['online_fpr']:.4f} "
            f"> threshold={cfg['alert_fpr']} — high false positive rate"
        )
    if error_stats["online_fnr"] > cfg["alert_fnr"]:
        alerts.append(
            f"online_fnr={error_stats['online_fnr']:.4f} "
            f"> threshold={cfg['alert_fnr']} — high false negative rate"
        )

    for alert in alerts:
        log.warning(f"ALERT: {alert}")

    # Log to MLflow
    run_name = f"online-eval-{since.date()}-to-{until.date()}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "eval_since": since.isoformat(),
            "eval_until": until.isoformat(),
            "lookback_days": cfg["lookback_days"],
            "total_meetings": correction_stats["total_meetings"],
            "total_predictions": correction_stats["total_predictions"],
        })
        mlflow.log_metrics({
            "online_correction_rate": correction_stats["correction_rate"],
            "online_meeting_correction_rate": correction_stats["meeting_correction_rate"],
            "online_total_corrections": correction_stats["total_corrections"],
            "online_fpr": error_stats["online_fpr"],
            "online_fnr": error_stats["online_fnr"],
            "online_false_positives": error_stats["false_positives"],
            "online_false_negatives": error_stats["false_negatives"],
            "n_alerts": len(alerts),
        })

        # Log per-version metrics
        for vd in version_breakdown:
            v = vd["model_version"].replace("/", "_")
            mlflow.log_metric(f"corrections_per_meeting_v{v}", vd["corrections_per_meeting"])

        # Log full summary as artifact
        summary = {
            "window": {"since": since.isoformat(), "until": until.isoformat()},
            "correction_stats": correction_stats,
            "error_stats": error_stats,
            "version_breakdown": version_breakdown,
            "weekly_trend": trend,
            "recent_retrains": retrain_history,
            "alerts": alerts,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = os.path.join(tmpdir, "online_eval_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            mlflow.log_artifact(summary_path)

        log.info(f"MLflow run: {run.info.run_id}")

    # Print summary
    log.info("=" * 60)
    log.info("ONLINE EVAL SUMMARY")
    log.info("=" * 60)
    log.info(f"  Window:              {since.date()} → {until.date()}")
    log.info(f"  Total meetings:      {correction_stats['total_meetings']}")
    log.info(f"  Total predictions:   {correction_stats['total_predictions']}")
    log.info(f"  Total corrections:   {correction_stats['total_corrections']}")
    log.info(f"  Correction rate:     {correction_stats['correction_rate']:.4f} "
             f"({'ALERT' if correction_stats['correction_rate'] > cfg['alert_correction_rate'] else 'OK'})")
    log.info(f"  Online FPR:          {error_stats['online_fpr']:.4f} "
             f"({'ALERT' if error_stats['online_fpr'] > cfg['alert_fpr'] else 'OK'})")
    log.info(f"  Online FNR:          {error_stats['online_fnr']:.4f} "
             f"({'ALERT' if error_stats['online_fnr'] > cfg['alert_fnr'] else 'OK'})")
    if alerts:
        log.warning(f"  ⚠ {len(alerts)} ALERT(s) — consider triggering retrain")
    else:
        log.info("  No alerts — model performing within expected bounds")
    log.info("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Online evaluation for jitsi-topic-segmenter")
    parser.add_argument("--days", type=int, default=30,
                        help="Lookback window in days (default: 30)")
    parser.add_argument("--since", default=None,
                        help="Start date (YYYY-MM-DD). Overrides --days.")
    parser.add_argument("--until", default=None,
                        help="End date (YYYY-MM-DD). Defaults to now.")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg["lookback_days"] = args.days

    until = datetime.now(timezone.utc)
    if args.until:
        until = datetime.fromisoformat(args.until).replace(tzinfo=timezone.utc)

    if args.since:
        since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
    else:
        since = until - timedelta(days=args.days)

    result = run_online_eval(cfg, since, until)
    return 1 if result.get("alerts") else 0


if __name__ == "__main__":
    exit(main())
