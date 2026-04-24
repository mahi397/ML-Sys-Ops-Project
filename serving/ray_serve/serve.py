"""
Ray Serve deployment for Jitsi Meeting Recap pipeline.
Prometheus metrics for Grafana monitoring.

Endpoints:
  GET  /health      → system health
  POST /segment     → Stage A: RoBERTa boundary detection
  POST /summarize   → Stage B: Mistral-7B summarization
  POST /recap       → Full pipeline
  GET  /metrics     → Prometheus scrape endpoint
"""

import ray
from ray import serve
import torch
import numpy as np
import json
import psutil
import threading
from pathlib import Path
import os
import time
import psutil
import psycopg2.extras
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware
from transformers import RobertaTokenizer, RobertaForSequenceClassification


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL STORAGE — JSONL file store (drop-in until Postgress ready)
# Swap: replace RecapStore methods with Postgres queries, API stays the same
# ═══════════════════════════════════════════════════════════════════════════════

_DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
_RECAPS_FILE     = _DATA_DIR / "recaps.jsonl"
_UTTERANCES_FILE = _DATA_DIR / "meeting_utterances.jsonl"
_FEEDBACK_FILE   = _DATA_DIR / "feedback_corrections.jsonl"


class RecapStore:
    """Thread-safe local JSONL store. One record per line."""
    def __init__(self):
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ── Recaps ──────────────────────────────────────────────────────

    def save_recap(self, meeting_id: str, model_version: str, segments_json: list):
        record = {
            "recap_id": f"{meeting_id}_{int(time.time())}",
            "meeting_id": meeting_id,
            "model_version": model_version,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "segments_json": segments_json
        }
        with self._lock:
            with open(_RECAPS_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
        return record["recap_id"]

    def get_recap(self, meeting_id: str):
        if not _RECAPS_FILE.exists():
            return None
        result = None
        with self._lock:
            with open(_RECAPS_FILE) as f:
                for line in f:
                    rec = json.loads(line.strip())
                    if rec["meeting_id"] == meeting_id:
                        result = rec
        return result
    
    def list_meetings(self) -> list:
        if not _RECAPS_FILE.exists():
            return []
        seen = {}
        with self._lock:
            with open(_RECAPS_FILE) as f:
                for line in f:
                    rec = json.loads(line.strip())
                    seen[rec["meeting_id"]] = rec 
        result = []
        for rec in seen.values():
            segs = rec.get("segments_json") or []
            result.append({
                "meeting_id":    rec["meeting_id"],
                "model_version": rec.get("model_version", ""),
                "created_at":    rec.get("created_at", ""),
                "segment_count": len(segs),
            })
        result.sort(key=lambda r: r["created_at"], reverse=True)
        return result[:50]

    # ── Utterances ──────────────────────────────────────────────────

    def save_utterances(self, meeting_id: str, utterances: list, decisions: list):
        rows = []
        for i, u in enumerate(utterances):
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
            with open(_UTTERANCES_FILE, "a") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

    def get_utterances(self, meeting_id: str) -> list:
        if not _UTTERANCES_FILE.exists():
            return []
        seen = {}  # deduplicate by utterance_idx — last write wins
        with self._lock:
            with open(_UTTERANCES_FILE) as f:
                for line in f:
                    rec = json.loads(line.strip())
                    if rec["meeting_id"] == meeting_id:
                        seen[rec["utterance_idx"]] = rec
        return sorted(seen.values(), key=lambda r: r["utterance_idx"])
        
    
    

    # ── Feedback corrections ─────────────────────────────────────────

    def save_feedback(self, meeting_id: str, utterance_idx: int,
                      action: str, original_label: int) -> dict:
        corrected_label = 0 if action == "remove_boundary" else 1
        record = {
            "correction_id": f"{meeting_id}_{utterance_idx}_{int(time.time())}",
            "meeting_id": meeting_id,
            "utterance_idx": utterance_idx,
            "original_label": original_label,
            "corrected_label": corrected_label,
            "action": action,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "used_in_retrain_version": None
        }
        with self._lock:
            with open(_FEEDBACK_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
        return record

    def count_pending_corrections(self) -> int:
        if not _FEEDBACK_FILE.exists():
            return 0
        count = 0
        with self._lock:
            with open(_FEEDBACK_FILE) as f:
                for line in f:
                    rec = json.loads(line.strip())
                    if rec.get("used_in_retrain_version") is None:
                        count += 1
        return count

    def get_all_feedback(self) -> list:
        if not _FEEDBACK_FILE.exists():
            return []
        rows = []
        with self._lock:
            with open(_FEEDBACK_FILE) as f:
                for line in f:
                    rows.append(json.loads(line.strip()))
        return rows

class PostgresRecapStore:
    """Reads meetings/utterances/segments from Postgres. Writes feedback to feedback_events"""

    def __init__(self):
        import psycopg2
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL env var not set")
        self._url = DATABASE_URL
        conn = psycopg2.connect(DATABASE_URL)
        conn.close()
        print("[postgres] Connection OK")

    def _conn(self):
        import psycopg2
        import psycopg2.extras
        return psycopg2.connect(self._url)

    def list_meetings(self) -> list:
        import psycopg2.extras
        sql = """
            SELECT m.meeting_id, m.source_name, m.started_at, m.ended_at,
                   COUNT(DISTINCT ss.segment_summary_id) AS segment_count
            FROM meetings m
            LEFT JOIN segment_summaries ss ON ss.meeting_id = m.meeting_id
            GROUP BY m.meeting_id, m.source_name, m.started_at, m.ended_at
            ORDER BY m.started_at DESC NULLS LAST LIMIT 50
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql)
                rows = cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            for k in ('started_at', 'ended_at'):
                if d.get(k) is not None:
                    d[k] = d[k].isoformat()
            result.append(d)
        return result

    def get_recap(self, meeting_id: str):
        import psycopg2.extras
        sql = """
    SELECT ss.segment_summary_id, ss.segment_index, ss.topic_label,
           ss.summary_bullets, ss.status, ss.model_version,
           ts.start_time_sec        AS t_start,
           ts.end_time_sec          AS t_end,
           u_start.utterance_index  AS start_utterance_index,
           u_end.utterance_index    AS end_utterance_index
    FROM segment_summaries ss
    JOIN topic_segments ts ON ss.topic_segment_id = ts.topic_segment_id
    LEFT JOIN utterances u_start ON u_start.utterance_id = ts.start_utterance_id
    LEFT JOIN utterances u_end   ON u_end.utterance_id   = ts.end_utterance_id
    WHERE ss.meeting_id = %s
    ORDER BY ss.segment_index
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (meeting_id,))
                rows = cur.fetchall()
        return [dict(r) for r in rows] if rows else None

    def get_utterances(self, meeting_id: str) -> list:
        import psycopg2.extras
        sql = """
            SELECT u.utterance_id, u.utterance_index, ms.speaker_label,
                   COALESCE(u.clean_text, u.raw_text) AS text,
                   u.start_time_sec, u.end_time_sec,
                   ut.pred_boundary_prob  AS boundary_confidence,
                   ut.pred_boundary_label AS is_boundary
            FROM utterances u
            JOIN  meeting_speakers ms ON ms.meeting_speaker_id = u.meeting_speaker_id
            LEFT JOIN utterance_transitions ut ON ut.left_utterance_id = u.utterance_id
            WHERE u.meeting_id = %s
            ORDER BY u.utterance_index
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (meeting_id,))
                return [dict(r) for r in cur.fetchall()]
            
            
    
    def get_segment_summary(self, segment_summary_id) -> dict:
        import psycopg2.extras
        sql = """
            SELECT ss.segment_summary_id, ss.topic_label, ss.summary_bullets,
                ss.segment_index, ts.start_utterance_id, ts.end_utterance_id
            FROM segment_summaries ss
            JOIN topic_segments ts ON ss.topic_segment_id = ts.topic_segment_id
            WHERE ss.segment_summary_id = %s
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (segment_summary_id,))
                row = cur.fetchone()
        return dict(row) if row else {}

    _ACTION_MAP = {
    "remove_boundary":  "merge_segments",
    "add_boundary":     "split_segment",
    "overall_good":     "accept_summary",
    "overall_positive": "accept_summary",
    "overall_bad":      "boundary_correction",
    "overall_negative": "boundary_correction",
    "overall_needs_work": "boundary_correction",
}

    def save_feedback(self, meeting_id: str, segment_summary_id,
                      event_type: str, before_payload: dict, after_payload: dict) -> int:
        if event_type.startswith("overall_"):
            positive_words = ("good", "positive", "great", "ok")
            event_type = "accept_summary" if any(w in event_type for w in positive_words) \
                        else "boundary_correction"
        else:
            event_type = {"remove_boundary": "merge_segments",
                        "add_boundary":    "split_segment"}.get(event_type, "boundary_correction")
        sql = """
            INSERT INTO feedback_events
                (meeting_id, segment_summary_id, event_type, event_source,
                 before_payload, after_payload)
            VALUES (%s, %s, %s, 'user', %s::jsonb, %s::jsonb)
            RETURNING feedback_event_id
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    meeting_id,
                    segment_summary_id,
                    event_type,
                    json.dumps(before_payload),
                    json.dumps(after_payload)
                ))
                fid = cur.fetchone()[0]
            conn.commit()
        return fid
# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

BOUNDARY_THRESHOLD    = float(os.getenv("BOUNDARY_THRESHOLD", "0.5"))
MODEL_PATH            = os.getenv("MODEL_PATH", "roberta-base")
LLM_MODEL_PATH        = os.getenv("LLM_MODEL_PATH", "")
MAX_SEGMENT_UTTERANCES = int(os.getenv("MAX_SEGMENT_UTTERANCES", "200"))
DEVICE             = "cuda"  # overridden per-actor in __init__

# MLflow model registry 
MLFLOW_TRACKING_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_ALIAS           = os.getenv("MODEL_ALIAS", "production")   # switch to "fallback" to rollback
MLFLOW_MODEL_NAME     = "jitsi-topic-segmenter"
DATABASE_URL = os.getenv("DATABASE_URL", "")



def _normalize_utterance(u: dict) -> dict:
    """Normalize null/None values frompipeline"""
    return {
        "position": u.get("position", 0),
        "speaker":  u.get("speaker") or "",
        "t_start":  u.get("t_start") or 0.0,
        "t_end":    u.get("t_end")   or 0.0,
        "text":     u.get("text")    or "",
    }

def format_window_for_roberta(window: list) -> str:
    """skip empty text utterances"""
    parts = []
    for u in sorted(window, key=lambda u: u["position"]):
        text = (u.get("text") or "").strip()
        speaker = u.get("speaker") or ""
        if text:  # skip padding/empty utterances — matches training code
            parts.append(f"[SPEAKER_{speaker}]: {text}")
    return " ".join(parts)
    #sorted_window = sorted(window, key=lambda u: u["position"])
    #return " ".join(f"[SPEAKER_{u['speaker']}]: {u['text']}" for u in sorted_window)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS ENDPOINT — owns the Prometheus registry
# Must be defined FIRST so other deployments don't touch prometheus objects
# ═══════════════════════════════════════════════════════════════════════════════

@serve.deployment(name="metrics", num_replicas=1)
class MetricsDeployment:
    def __init__(self):
        from prometheus_client import (
            Counter, Histogram, Gauge, Info,
            CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
        )
        self.registry = CollectorRegistry()
        self.generate_latest = generate_latest
        self.CONTENT_TYPE_LATEST = CONTENT_TYPE_LATEST

        # All metrics live here — other deployments push data via /metrics/record
        self.request_count = Counter(
            'jitsi_requests_total', 'Total requests',
            ['endpoint', 'status'], registry=self.registry
        )
        self.request_latency = Histogram(
            'jitsi_request_latency_seconds', 'Latency',
            ['endpoint'],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
            registry=self.registry
        )
        self.batch_size = Histogram(
            'jitsi_batch_size', 'Batch sizes',
            ['model'], buckets=[1, 2, 4, 8, 16], registry=self.registry
        )
        self.confidence = Histogram(
            'jitsi_boundary_confidence', 'Confidence scores',
            [], buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        self.segments_detected = Counter(
            'jitsi_segments_detected_total', 'Boundaries detected',
            registry=self.registry
        )
        self.summary_length = Histogram(
            'jitsi_summary_length_chars', 'Summary length',
            [], buckets=[50, 100, 200, 500, 1000, 2000, 5000],
            registry=self.registry
        )
        self.gpu_mem_used = Gauge(
            'jitsi_gpu_memory_used_mb', 'GPU mem used', registry=self.registry
        )
        self.gpu_mem_total = Gauge(
            'jitsi_gpu_memory_total_mb', 'GPU mem total', registry=self.registry
        )
        self.cpu_util = Gauge(
            'jitsi_cpu_utilization_percent', 'CPU %', registry=self.registry
        )
        self.ram_used = Gauge(
            'jitsi_ram_used_mb', 'RAM used', registry=self.registry
        )
        self.model_loaded = Gauge(
            'jitsi_model_loaded', 'Model loaded',
            ['model_name', 'model_version'], registry=self.registry
        )
        self.sla_violations = Counter(
            'jitsi_sla_violations_total', 'SLA violations',
            ['endpoint', 'sla_type'], registry=self.registry
        )
        self.recap_duration = Histogram(
            'jitsi_recap_duration_seconds', 'Recap duration',
            [], buckets=[5, 10, 30, 60, 120, 180, 300, 600],
            registry=self.registry
        )
        self.recap_segments = Histogram(
            'jitsi_recap_segments_per_meeting', 'Segments per meeting',
            [], buckets=[1, 2, 3, 5, 8, 10, 15, 20],
            registry=self.registry
        )
        self.active_requests = Gauge(
            'jitsi_active_requests', 'Active requests',
            ['endpoint'], registry=self.registry
        )
        self.feedback_corrections = Counter(
            'jitsi_feedback_corrections_total', 'User boundary corrections',
            ['action'], registry=self.registry
        )
        # Initialize all label combinations so Prometheus exposes them from startup
        # (without this, panels show "No data" until the first feedback is submitted)
        for _action in ("remove_boundary", "add_boundary", "overall_good", "overall_bad"):
            self.feedback_corrections.labels(action=_action)

    def _update_system(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.gpu_mem_used.set(info.used / 1024 / 1024)
            self.gpu_mem_total.set(info.total / 1024 / 1024)
        except Exception:
            try:
                if torch.cuda.is_available():
                    self.gpu_mem_used.set(torch.cuda.memory_allocated(0) / 1024 / 1024)
                    self.gpu_mem_total.set(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
            except Exception:
                pass
        self.cpu_util.set(psutil.cpu_percent())
        self.ram_used.set(psutil.virtual_memory().used / 1024 / 1024)

    def record(self, data: dict):
        """Record metrics from other deployments."""
        endpoint = data.get("endpoint", "")
        status = data.get("status", "success")
        latency = data.get("latency", 0)

        self.request_count.labels(endpoint=endpoint, status=status).inc()
        if latency > 0:
            self.request_latency.labels(endpoint=endpoint).observe(latency)

        if "batch_size" in data:
            self.batch_size.labels(model="segmenter").observe(data["batch_size"])    
        if "confidence" in data:
            self.confidence.observe(data["confidence"])
        if data.get("is_boundary"):
            self.segments_detected.inc()
        if "summary_length" in data:
            self.summary_length.observe(data["summary_length"])
        if "sla_violation" in data:
            self.sla_violations.labels(
                endpoint=endpoint, sla_type=data["sla_violation"]
            ).inc()
        if "model_loaded" in data:
            self.model_loaded.labels(
                model_name=data["model_name"],
                model_version=data.get("model_version", "unknown")
            ).set(1 if data["model_loaded"] else 0)

        if "recap_duration" in data:
            self.recap_duration.observe(data["recap_duration"])
        if "recap_segments" in data:
            self.recap_segments.observe(data["recap_segments"])
        if "feedback_action" in data:
            self.feedback_corrections.labels(action=data["feedback_action"]).inc()
        if data.get("active_inc"):
            self.active_requests.labels(endpoint=endpoint).inc()
        if data.get("active_dec"):
            self.active_requests.labels(endpoint=endpoint).dec()    

    async def __call__(self, request: Request) -> Response:
        self._update_system()
        return Response(
            content=self.generate_latest(self.registry),
            media_type=self.CONTENT_TYPE_LATEST
        )


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE A: RoBERTa SEGMENTER
# ═══════════════════════════════════════════════════════════════════════════════

@serve.deployment(
    name="segmenter",
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.3},
    #max_ongoing_requests=10, 
)
class SegmenterDeployment:
    def __init__(self):
        self.metrics = serve.get_deployment_handle("metrics", app_name="metrics")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = BOUNDARY_THRESHOLD
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

                # ── Load model: MLflow registry → local path → base weights ──
        self.model = None
        self.model_version = "base"
        self.current_mlflow_version = "unknown"
        self.threshold = BOUNDARY_THRESHOLD  # default fallback 

        # 1. Try MLflow registry
        if MLFLOW_TRACKING_URI:
            try:
                import mlflow.pytorch
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                # Instantiate client once here so it's available in the fallback block
                _client = mlflow.tracking.MlflowClient()
                mlflow_uri = f"models:/{MLFLOW_MODEL_NAME}@{MODEL_ALIAS}"
                self.model = mlflow.pytorch.load_model(mlflow_uri)
                self.model_version = f"mlflow@{MODEL_ALIAS}"
                _alias_mv = _client.get_model_version_by_alias(MLFLOW_MODEL_NAME, MODEL_ALIAS)
                self.current_mlflow_version = str(_alias_mv.version)
                # ★ READ best_threshold FROM MODEL VERSION TAGS ★
                tags = _alias_mv.tags or {}
                if "best_threshold" in tags:
                    self.threshold = float(tags["best_threshold"])
                    print(f"[segmenter] Using best_threshold={self.threshold} from MLflow v{_alias_mv.version}")
                else:
                    print(f"[segmenter] No best_threshold tag on v{_alias_mv.version}, using default {self.threshold}")
                print(f"[segmenter] Loaded model from MLflow: {mlflow_uri}")
            except Exception as e:
                print(f"[segmenter] MLflow @{MODEL_ALIAS} failed ({e}), trying @fallback")
                try:
                    import mlflow.pytorch
                    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                    _client = mlflow.tracking.MlflowClient()
                    fallback_uri = f"models:/{MLFLOW_MODEL_NAME}@fallback"
                    self.model = mlflow.pytorch.load_model(fallback_uri)
                    self.model_version = "mlflow@fallback"
                    # Also read threshold from fallback version
                    _fb_mv = _client.get_model_version_by_alias(MLFLOW_MODEL_NAME, "fallback")
                    fb_tags = _fb_mv.tags or {}
                    if "best_threshold" in fb_tags:
                        self.threshold = float(fb_tags["best_threshold"])
                        print(f"[segmenter] Using fallback best_threshold={self.threshold}")
                    print(f"[segmenter] Loaded fallback model from MLflow")
                except Exception as e2:
                    print(f"[segmenter] MLflow fallback also failed ({e2}), using local path")
        # 2. Fall back to local fine-tuned weights
        if self.model is None and os.path.exists(MODEL_PATH):
            self.model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
            self.model_version = "local-fine-tuned"
            print(f"[segmenter] Loaded local fine-tuned model from {MODEL_PATH}")

        # 3. Last resort — base weights (no fine-tuning)
        if self.model is None:
            self.model = RobertaForSequenceClassification.from_pretrained(
                "roberta-base", num_labels=2
            )
            self.model_version = "base"
            print(f"[segmenter] WARNING: Using base roberta weights (no fine-tuning)")

        self.model.to(self.device)
        self.model.eval()
        self._model_lock = threading.Lock()
        threading.Thread(target=self._reload_loop, daemon=True).start()

        # Report model loaded
        self.metrics.record.remote({
            "endpoint": "segment", "model_loaded": True,
             "model_name": "roberta_segmenter",
            "model_version": self.model_version
        })
        print(f"[segmenter] Ready on {self.device}, version={self.model_version}, threshold={self.threshold}")

    def _reload_loop(self):
        import mlflow.pytorch
        check_interval = int(os.getenv("MODEL_RELOAD_INTERVAL_SECONDS", "300"))
        while True:
            time.sleep(check_interval)
            try:
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                client = mlflow.tracking.MlflowClient()
                alias_mv = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, MODEL_ALIAS)
                new_version = alias_mv.version

                if str(new_version) != str(self.current_mlflow_version):
                    print(f"[segmenter] New model version detected: {new_version}, reloading...")
                    new_model = mlflow.pytorch.load_model(
                        f"models:/{MLFLOW_MODEL_NAME}@{MODEL_ALIAS}"
                    )
                    new_model.to(self.device)
                    new_model.eval()

                     #  READ NEW THRESHOLD 
                    tags = alias_mv.tags or {}
                    new_threshold = float(tags.get("best_threshold", self.threshold))
                    with self._model_lock:
                        self.model = new_model
                        self.model_version = f"mlflow@{MODEL_ALIAS}_v{new_version}"  # ← consistent format
                        self.current_mlflow_version = new_version
                        self.threshold = new_threshold 
                    print(f"[segmenter] Model reloaded: {new_version}")
                    self.metrics.record.remote({
                        "endpoint": "segment", "model_loaded": True,
                        "model_name": "roberta_segmenter",
                        "model_version": new_version
                    })
            except Exception as e:
                print(f"[segmenter] Reload check failed: {e}")

    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.05)
    async def batch_predict(self, requests: list) -> list:
        batch_size = len(requests)

        texts = []
        metadata = []
        for req in requests:
            # Normalize null values from pipeline
            window = [_normalize_utterance(u) for u in req["window"]]
            texts.append(format_window_for_roberta(window))
            #texts.append(format_window_for_roberta(req["window"]))
            ti = req["transition_index"]
            metadata.append({
                "meeting_id": req["meeting_id"],
                "transition_index": ti,
                "meeting_offset_seconds": req["meeting_offset_seconds"],
                "t_boundary": window[ti]["t_end"]
            })

        inputs = self.tokenizer(
            texts, return_tensors="pt", truncation=True,
            max_length=512, padding="max_length"
        ).to(self.device)

        with self._model_lock:
            model = self.model
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            boundary_probs = probs[:, 1].cpu().tolist()

        results = []
        for prob, meta in zip(boundary_probs, metadata):
            is_boundary = prob >= self.threshold
            # Record metrics async (non-blocking)
            self.metrics.record.remote({
                "endpoint": "segment", "batch_size": batch_size,
                "confidence": prob, "is_boundary": is_boundary
            })
            results.append({
                "meeting_id": meta["meeting_id"],
                "transition_after_position": meta["transition_index"],
                "boundary_probability": round(prob, 4),
                "is_boundary": is_boundary,
                "t_boundary": meta["t_boundary"],
                "segment_so_far": {
                    "t_start": meta["meeting_offset_seconds"],
                    "t_end": meta["t_boundary"]
                }
            })
        return results

    async def predict_single(self, body: dict) -> dict:
        result = await self.batch_predict(body)
        return result

    async def predict_batch(self, bodies: list) -> list:
        """Run predict_single concurrently for a list of window bodies"""
        import asyncio
        tasks = [self.batch_predict(b) for b in bodies]
        return await asyncio.gather(*tasks)
    
    async def __call__(self, request: Request) -> JSONResponse:
        start = time.time()
        try:
            body = await request.json()
            # ── Batch format: {"meeting_id":..., "requests": [...]} ──
            if "requests" in body:
                reqs = body["requests"]
                if not reqs:
                    return JSONResponse({"error": "Empty requests list"}, status_code=400)
                results = await self.predict_batch(reqs)
                latency = time.time() - start
                # Attach request_id back to each result
                for i, res in enumerate(results):
                    res["request_id"] = reqs[i].get("request_id", f"t{i}")
                    meta = reqs[i].get("metadata", {})
                    res["left_model_index"]  = meta.get("left_model_index")
                    res["right_model_index"] = meta.get("right_model_index")
                self.metrics.record.remote({
                    "endpoint": "segment", "status": "success",
                    "latency": latency, "batch_size": len(reqs)
                })
                return JSONResponse(content={
                    "meeting_id": body.get("meeting_id"),
                    "request_count": len(results),
                    "results": results
                })

            # ── Single window format: {"window": [...]} ──
            if "window" not in body:
                self.metrics.record.remote({"endpoint": "segment", "status": "error"})
                return JSONResponse({"error": "Missing 'window' or 'requests'"}, status_code=400)
                #return JSONResponse({"error": "Missing 'window'"}, status_code=400)

            result = await self.predict_single(body)
            latency = time.time() - start
            record = {"endpoint": "segment", "status": "success", "latency": latency}
            if latency > 2.0:
                record["sla_violation"] = "latency_2s"
            self.metrics.record.remote(record)
            return JSONResponse(content=result)
        except Exception as e:
            self.metrics.record.remote({"endpoint": "segment", "status": "error"})
            return JSONResponse({"error": str(e)}, status_code=500)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE B: LLM SUMMARIZER
# ═══════════════════════════════════════════════════════════════════════════════

@serve.deployment(
    name="summarizer",
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.7},
    #max_ongoing_requests=3,
)
class SummarizerDeployment:
    def __init__(self):
        self.metrics = serve.get_deployment_handle("metrics", app_name="metrics")
        self.llm = None
        if LLM_MODEL_PATH and os.path.exists(LLM_MODEL_PATH):
            from llama_cpp import Llama
            self.llm = Llama(
                model_path=LLM_MODEL_PATH,
                n_gpu_layers=-1,
                n_ctx=4096,
                verbose=False
            )
            llm_version = os.path.basename(LLM_MODEL_PATH).replace(".gguf", "").split(".")[-1] or "gguf"
            self.metrics.record.remote({
                "endpoint": "summarize", "model_loaded": True,
                "model_name": "mistral_summarizer",
                "model_version": llm_version
            })
            print(f"[summarizer] LLM loaded from {LLM_MODEL_PATH}")
        else:
            self.metrics.record.remote({
                "endpoint": "summarize", "model_loaded": False,
                "model_name": "mistral_summarizer",
                "model_version": "none"
            })
            print(f"[summarizer] No LLM at '{LLM_MODEL_PATH}' — draft mode")

    def _summarize(self, body: dict) -> dict:
        meeting_id = body.get("meeting_id", "unknown")
        segment_id = body.get("segment_id", 0)
        t_start = body.get("t_start", 0)
        t_end = body.get("t_end", 0)

        # ── VALIDATION: require utterances ──
        utterances = body.get("utterances", [])
        if not utterances:
            return {
                "meeting_id": meeting_id, "segment_id": segment_id,
                "t_start": t_start, "t_end": t_end,
                "topic_label": "", "summary_bullets": [],
                "status": "error", "error": "No utterances provided"
            }


        if self.llm is None:
            return {
                "meeting_id": meeting_id, "segment_id": segment_id,
                "t_start": t_start, "t_end": t_end,
                "topic_label": "", "summary_bullets": [], "status": "draft"
            }

        MAX_RETRIES = 10
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                utterances = body.get("utterances", [])[:MAX_SEGMENT_UTTERANCES]
                transcript = "\n".join(
                    f"[SPEAKER_{u['speaker']}]: {u['text']}" for u in utterances
                )
                seg_ctx = body.get("meeting_context", {})
                prompt = f"""<s>[INST] You are a meeting assistant. Summarize this meeting segment.

    Segment {seg_ctx.get('segment_index_in_meeting', 1)} of {seg_ctx.get('total_segments', 1)}.

    Transcript:
    {transcript}

    Respond with ONLY this JSON, no other text:
    {{"topic_label": "2-5 word label", "summary_bullets": ["point 1", "point 2", "point 3"]}} [/INST]"""

                response = self.llm(prompt, max_tokens=300, temperature=0.1, stop=["```"])
                text = response["choices"][0]["text"].strip()
                print(f"[summarizer] attempt {attempt} raw output: {text[:300]}")
                start_idx = text.find("{")
                end_idx = text.rfind("}") + 1
                parsed = json.loads(text[start_idx:end_idx])

                if not parsed.get("topic_label"):
                    raise ValueError(f"Empty topic_label in response: {text}")

                return {
                    "meeting_id": meeting_id, "segment_id": segment_id,
                    "t_start": t_start, "t_end": t_end,
                    "topic_label": parsed["topic_label"],
                    "summary_bullets": parsed.get("summary_bullets", []),
                    "status": "complete"
                }
            except Exception as e:
                last_error = e
                print(f"[summarizer] Attempt {attempt}/{MAX_RETRIES} failed for segment {segment_id}: {e}")
                time.sleep(1)  # sync sleep — _summarize is a regular method

        print(f"[summarizer] All {MAX_RETRIES} attempts failed for segment {segment_id}: {last_error}")
        return {
            "meeting_id": meeting_id, "segment_id": segment_id,
            "t_start": t_start, "t_end": t_end,
            "topic_label": "", "summary_bullets": [], "status": "draft"
        }

    async def summarize_dict(self, body: dict) -> dict:
        start = time.time()
        result = self._summarize(body)
        latency = time.time() - start
        record = {"endpoint": "summarize", "status": "success", "latency": latency}
        if result["status"] == "complete":
            record["summary_length"] = len(
                result.get("topic_label", "") +
                " ".join(result.get("summary_bullets", []))
            )
        if latency > 30.0:
            record["sla_violation"] = "latency_30s"
        self.metrics.record.remote(record)
        return result

    async def __call__(self, request: Request) -> JSONResponse:
        start = time.time()
        try:
            body = await request.json()
            result = self._summarize(body)
            latency = time.time() - start
            record = {"endpoint": "summarize", "status": "success", "latency": latency}
            if result["status"] == "complete":
                record["summary_length"] = len(
                    result.get("topic_label", "") +
                    " ".join(result.get("summary_bullets", []))
                )
            if latency > 30.0:
                record["sla_violation"] = "latency_30s"
            self.metrics.record.remote(record)
            return JSONResponse(content=result)
        except Exception as e:
            self.metrics.record.remote({"endpoint": "summarize", "status": "error"})
            return JSONResponse({"error": str(e)}, status_code=500)


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE: /recap
# ═══════════════════════════════════════════════════════════════════════════════

@serve.deployment(name="recap_pipeline", num_replicas=1)
class RecapPipelineDeployment:
    def __init__(self):
        self.metrics = serve.get_deployment_handle("metrics", app_name="metrics")
        self.segmenter = serve.get_deployment_handle("segmenter", app_name="segmenter")
        #self.summarizer = serve.get_deployment_handle("summarizer", app_name="summarizer")
        #save recap + utterances inside the /recap pipeline after summaries are assembled
        self.summarizer = serve.get_deployment_handle("summarizer", app_name="summarizer")
        self.store = RecapStore()

    def _build_windows(self, utterances, window_size=7):
        windows = []
        half = window_size // 2
        for i in range(len(utterances) - 1):
            start = max(0, i - half)
            end = min(len(utterances), i + half + 1)
            window = utterances[start:end]
            while len(window) < window_size:
                window.append({
                    "position": len(window), "speaker": "",
                    "t_start": 0.0, "t_end": 0.0, "text": ""
                })
            window = [{**u, "position": j} for j, u in enumerate(window)]
            windows.append({
                "transition_index": half,
                "meeting_offset_seconds": window[0]["t_start"],
                "window": window
            })
        return windows

    def _assemble_segments(self, utterances, decisions):
        segments = []
        current = []
        seg_id = 1
        seg_start = utterances[0]["t_start"] if utterances else 0.0
        for i, decision in enumerate(decisions):
            current.append(utterances[i])
            if decision["is_boundary"] or i == len(decisions) - 1:
                segments.append({
                    "segment_id": seg_id,
                    "t_start": seg_start,
                    "t_end": utterances[i]["t_end"],
                    "utterances": current,
                    "total_utterances": len(current)
                })
                seg_id += 1
                seg_start = utterances[i]["t_end"]
                current = []
        return segments
    
    def _validate(self, meeting_id: str, utterances: list):
        """Validate input matching training assumptions exactly

        Rules (from training_assumptions.pdf):
          - 0 utterances OR all under 20 chars → reject 400
          - < 2 utterances → reject 400 "meeting too short for inference"
          - 2–6 utterances → allow, flag "short meeting, low confidence"
          - 7+ utterances → fully valid, no flag
        """
        if not utterances:
            return None, JSONResponse(
                {"error": "Empty transcript — no utterances provided"},
                status_code=400
            )
        # Filter out utterances under 20 chars after cleaning (matches training MIN_CHARS=20)
        import re
        FILLER_RE = re.compile(r"\b(uh+|um+|i mean|you know|like)\b", re.IGNORECASE)
        def clean_text(t):
            t = (t or "").lower()
            t = FILLER_RE.sub("", t)
            return re.sub(r"\s+", " ", t).strip()

        valid = [u for u in utterances if len(clean_text(u.get("text", ""))) >= 20]

        if len(valid) == 0:
            return None, JSONResponse(
                {"error": f"All utterances under 20 chars after cleaning - skipping inference for {meeting_id}"},
                status_code=400
            )

        if len(valid) < 2:
            return None, JSONResponse(
                {"error": f"meeting too short for inference - need at least 2 valid utterances, got {len(valid)}"},
                status_code=400
            )

        # Replace utterances list in-place with cleaned valid ones
        utterances[:] = valid
        warnings = []
        if len(utterances) < 7:
            warnings.append(f"short_meeting_low_confidence: only {len(utterances)} utterances, segmentation may be unreliable")

        speakers = set(u.get("speaker") or "" for u in utterances)
        if len(speakers) == 1:
            warnings.append(f"single_speaker: all utterances from one speaker, boundaries based on content only")
        duration = utterances[-1]["t_end"] - utterances[0]["t_start"]
        if duration < 10:
            warnings.append(f"very_short_meeting: duration {duration:.1f}s < 10s")
        if len(utterances) > 2000:
            warnings.append(f"very_long_meeting: {len(utterances)} utterances, truncating to 2000")
            utterances[:] = utterances[:2000]
        return warnings, None

    async def __call__(self, request: Request) -> JSONResponse:
        start = time.time()
        try:
            body = await request.json()
            meeting_id = body.get("meeting_id", "unknown")
            utterances = body.get("utterances", [])

            #if not utterances or len(utterances) < 2:
            # Robustness: validate before doing any inference
            warnings, err = self._validate(meeting_id, utterances)
            if err:
                self.metrics.record.remote({"endpoint": "recap", "status": "error"})
            #    return JSONResponse({"error": "Need at least 2 utterances"}, status_code=400)
                return err

            # Single utterance edge case — skip segmentation entirely
            if len(utterances) == 1:
                sum_result = await self.summarizer.summarize_dict.remote({
                    "meeting_id": meeting_id, "segment_id": 1,
                    "t_start": utterances[0]["t_start"], "t_end": utterances[0]["t_end"],
                    "utterances": utterances, "total_utterances": 1,
                    "meeting_context": {"total_segments": 1, "segment_index_in_meeting": 1}
                })
                elapsed = time.time() - start
                self.metrics.record.remote({
                    "endpoint": "recap", "status": "success",
                    "latency": elapsed, "recap_duration": elapsed, "recap_segments": 1
                })
                return JSONResponse(content={
                    "meeting_id": meeting_id,
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "total_segments": 1, "processing_time_seconds": round(elapsed, 1),
                    "warnings": warnings, "recap": [sum_result]
                })

            # Stage A — segmentation

            windows = self._build_windows(utterances)
            seg_refs = []
            for w in windows:
                payload = {"meeting_id": meeting_id, **w}
                ref = self.segmenter.predict_single.remote(payload)
                seg_refs.append(ref)

            decisions = []
            for ref in seg_refs:
                result = await ref
                decisions.append(result)

            # Assemble
            segments = self._assemble_segments(utterances, decisions)

            # Stage B
            summaries = []
            for seg in segments:
                sum_payload = {
                    "meeting_id": meeting_id,
                    "segment_id": seg["segment_id"],
                    "t_start": seg["t_start"],
                    "t_end": seg["t_end"],
                    "utterances": seg["utterances"],
                    "total_utterances": seg["total_utterances"],
                    "meeting_context": {
                        "total_segments": len(segments),
                        "segment_index_in_meeting": seg["segment_id"]
                    }
                }
                sum_result = await self.summarizer.summarize_dict.remote(sum_payload)
                summaries.append(sum_result)

            elapsed = time.time() - start
            record = {
                "endpoint": "recap", "status": "success", "latency": elapsed,
                "recap_duration": elapsed, "recap_segments": len(segments)
            }
            if elapsed > 300.0:
                record["sla_violation"] = "latency_300s"
            self.metrics.record.remote(record)

            # Persist recap + utterances for UI and feedback loop
            model_version = os.getenv("MODEL_VERSION", "base")
            self.store.save_recap(meeting_id, model_version, summaries)
            self.store.save_utterances(meeting_id, utterances, decisions)

            return JSONResponse(content={
                "meeting_id": meeting_id,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_segments": len(summaries),
                "processing_time_seconds": round(elapsed, 1),
                "warnings": warnings or [],
                "recap": summaries
            })

        except Exception as e:
            self.metrics.record.remote({"endpoint": "recap", "status": "error"})
            return JSONResponse({"error": str(e)}, status_code=500)

# ═══════════════════════════════════════════════════════════════════════════════
# RECAP API — GET /api/recap/{meeting_id} + POST /api/feedback --Postgres
# ═══════════════════════════════════════════════════════════════════════════════

@serve.deployment(name="recap_api", num_replicas=1)
class RecapAPIDeployment:
    """
    GET  /api/recap/{meeting_id}  → fetch stored recap + utterances
    POST /api/feedback            → submit boundary correction
    GET  /api/feedback/count      → pending corrections count (for retraining trigger)
    GET  /api/feedback/all        → all corrections ( retraining pipeline)
    """
    #def __init__(self):
    #    self.store = RecapStore()

    
    def __init__(self):
        if DATABASE_URL:
            try:
                self.store = PostgresRecapStore()
                print("[recap_api] Using PostgresRecapStore")
            except Exception as e:
                print(f"[recap_api] Postgres init failed ({e}), falling back to RecapStore")
                self.store = RecapStore()
                #self.metrics = serve.get_deployment_handle("metrics", app_name="metrics")
        else:
            print("[recap_api] No DATABASE_URL — using RecapStore (JSONL)")
            self.store = RecapStore()
        self.metrics = serve.get_deployment_handle("metrics", app_name="metrics")


    async def __call__(self, request: Request) -> JSONResponse:
        path = request.url.path
        method = request.method

        # ── GET /api/recap/{meeting_id} ──────────────────────────

        if method == "GET" and "/api/recap/" in path:
            meeting_id = path.split("/api/recap/")[-1].strip("/")
            if not meeting_id:
                return JSONResponse({"error": "meeting_id required"}, status_code=400)

            if isinstance(self.store, PostgresRecapStore):
                # ── Postgres path ────────────────────────────────
                segments   = self.store.get_recap(meeting_id)
                if not segments:
                    return JSONResponse(
                        {"error": f"No recap found for meeting_id '{meeting_id}'"},
                        status_code=404
                    )
                utterances = self.store.get_utterances(meeting_id)

                # utterance_id → utterance_index lookup for start/end mapping
                last_idx = utterances[-1]["utterance_index"] if utterances else 0

                ui_segments = []
                for i, seg in enumerate(segments):
                    start_utt = seg["start_utterance_index"] if seg.get("start_utterance_index") is not None else 0
                    end_utt   = seg["end_utterance_index"]   if seg.get("end_utterance_index")   is not None else last_idx
                    _matched  = next((u for u in utterances if u["utterance_index"] == end_utt), None)
                    conf = _matched.get("boundary_confidence") if _matched else None
                    bullets = seg.get("summary_bullets") or []
                    if isinstance(bullets, str):
                        try:
                            bullets = json.loads(bullets)
                        except Exception:
                            bullets = [bullets]
                    ui_segments.append({
                        "segment_idx":        i,
                        "segment_summary_id": seg["segment_summary_id"],
                        "start_utt":          start_utt,
                        "end_utt":            end_utt,
                        "t_start":            float(seg.get("t_start") or 0),
                        "t_end":              float(seg.get("t_end") or 0),
                        "topic_label":        seg.get("topic_label", ""),
                        "summary":            ". ".join(bullets) or seg.get("topic_label", "No summary"),
                        "boundary_confidence": float(conf) if conf is not None else None,
                    })

                speakers = list({u.get("speaker_label", "") for u in utterances if u.get("speaker_label")})
                t_vals = [float(u["start_time_sec"]) for u in utterances if u.get("start_time_sec") is not None]
                duration_secs = int(max(t_vals) - min(t_vals)) if len(t_vals) > 1 else 0
                duration_str  = f"{duration_secs // 60} min" if duration_secs else ""

                ui_utterances = [
                    {
                        "utterance_idx": u["utterance_index"],
                        "speaker":       u.get("speaker_label", "Speaker"),
                        "text":          u.get("text", ""),
                        "t_start":       float(u.get("start_time_sec") or 0),
                    }
                    for u in utterances
                ]

                return JSONResponse({
                    "recap_id":         f"pg_{meeting_id}",
                    "meeting_id":       meeting_id,
                    "meeting_title":    meeting_id,
                    "meeting_duration": duration_str,
                    "participant_count": len(speakers),
                    "model_version":    segments[0].get("model_version") or f"mlflow@{MODEL_ALIAS}",
                    "created_at":       "",
                    "segments":         ui_segments,
                    "utterances":       ui_utterances,
                })

            else:
                # ── JSONL fallback path ──────────────────────────
                recap = self.store.get_recap(meeting_id)
                if not recap:
                    return JSONResponse(
                        {"error": f"No recap found for meeting_id '{meeting_id}'"},
                        status_code=404
                    )
                utterances = self.store.get_utterances(meeting_id)
                boundary_idxs = [u["utterance_idx"] for u in utterances
                                 if u.get("predicted_label") == 1]
                summaries = recap["segments_json"]
                ui_segments = []
                start_utt = 0
                for i, seg in enumerate(summaries):
                    end_utt = boundary_idxs[i] if i < len(boundary_idxs) else (
                        utterances[-1]["utterance_idx"] if utterances else 0
                    )
                    conf = next(
                        (u.get("boundary_confidence") for u in utterances
                         if u["utterance_idx"] == end_utt), None
                    )
                    ui_segments.append({
                        "segment_idx":        i,
                        "segment_summary_id": None,
                        "start_utt":          start_utt,
                        "end_utt":            end_utt,
                        "t_start":            seg.get("t_start", 0),
                        "t_end":              seg.get("t_end", 0),
                        "topic_label":        seg.get("topic_label", ""),
                        "summary":            ". ".join(seg.get("summary_bullets", []))
                                              or seg.get("topic_label", "No summary"),
                        "boundary_confidence": conf,
                    })
                    start_utt = end_utt + 1
                speakers = list({u.get("speaker", "") for u in utterances if u.get("speaker")})
                t_values = [u.get("t_start", 0) for u in utterances if u.get("t_start") is not None]
                duration_secs = int(max(t_values) - min(t_values)) if len(t_values) > 1 else 0
                duration_str  = f"{duration_secs // 60} min" if duration_secs else ""
                ui_utterances = [
                    {
                        "utterance_idx": u["utterance_idx"],
                        "speaker":       u.get("speaker", "Speaker"),
                        "text":          u.get("text", ""),
                        "t_start":       u.get("t_start", 0),
                    }
                    for u in utterances
                ]
                return JSONResponse({
                    "recap_id":         recap["recap_id"],
                    "meeting_id":       meeting_id,
                    "meeting_title":    meeting_id,
                    "meeting_duration": duration_str,
                    "participant_count": len(speakers),
                    "model_version":    recap["model_version"],
                    "created_at":       recap["created_at"],
                    "segments":         ui_segments,
                    "utterances":       ui_utterances,
                })
            
        # ── POST /api/feedback ───────────────────────────────────
             
        if method == "POST" and path.rstrip("/") == "/api/feedback":
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

            meeting_id         = body.get("meeting_id")
            utterance_idx      = body.get("utterance_idx")
            action             = body.get("action")
            segment_summary_id = body.get("segment_summary_id")  # passed by UI from segment data

            if not meeting_id:
                return JSONResponse({"error": "meeting_id required"}, status_code=400)
            if not action:
                return JSONResponse({"error": "action required"}, status_code=400)

            # ── Overall meeting-level feedback ────────────────────
            if isinstance(action, str) and action.startswith("overall_"):
                if isinstance(self.store, PostgresRecapStore):
                    before = {}
                    after  = {"rating": action.replace("overall_", "")}
                    fid = self.store.save_feedback(meeting_id, None, action, before, after)
                    self.metrics.record.remote({"feedback_action": "overall"})
                    return JSONResponse({"status": "recorded", "action": action,
                                        "feedback_event_id": fid})
                else:
                    correction = self.store.save_feedback(meeting_id, -1, action, -1)
                    self.metrics.record.remote({"feedback_action": "overall"})  
                    return JSONResponse({"status": "recorded", "action": action,
                                        "correction_id": correction["correction_id"]})

            # ── Boundary-level feedback ───────────────────────────
            if utterance_idx is None:
                return JSONResponse({"error": "utterance_idx required"}, status_code=400)
            if action not in ("remove_boundary", "add_boundary"):
                return JSONResponse(
                    {"error": "action must be 'remove_boundary', 'add_boundary', or 'overall_*'"},
                    status_code=400
                )

            if isinstance(self.store, PostgresRecapStore):
                seg = {}
                if segment_summary_id:
                    try:
                        seg = self.store.get_segment_summary(segment_summary_id)
                        bullets = seg.get("summary_bullets") or []
                        if isinstance(bullets, str):
                            seg["summary_bullets"] = json.loads(bullets)
                    except Exception:
                        pass
                seg_snapshot = {k: seg.get(k) for k in
                                ("segment_summary_id", "topic_label", "summary_bullets", "segment_index")}
                before = {"utterance_idx": utterance_idx,
                        "is_boundary": action == "remove_boundary",
                        "segment": seg_snapshot}
                after  = {"utterance_idx": utterance_idx,
                        "is_boundary": action == "add_boundary"}
                fid = self.store.save_feedback(
                    meeting_id, segment_summary_id, action, before, after
                )
                self.metrics.record.remote({"feedback_action": action}) 
                return JSONResponse({
                    "status": "recorded",
                    "feedback_event_id": fid,
                    "meeting_id":    meeting_id,
                    "utterance_idx": utterance_idx,
                    "action":        action,
                })
            else:
                utterances = self.store.get_utterances(meeting_id)
                original_label = 0
                for u in utterances:
                    if u["utterance_idx"] == utterance_idx:
                        original_label = u["predicted_label"]
                        break
                correction = self.store.save_feedback(
                    meeting_id, utterance_idx, action, original_label
                )
                self.metrics.record.remote({"feedback_action": action}) 
                return JSONResponse({
                    "status":          "recorded",
                    "correction_id":   correction["correction_id"],
                    "meeting_id":      meeting_id,
                    "utterance_idx":   utterance_idx,
                    "action":          action,
                    "original_label":  original_label,
                    "corrected_label": correction["corrected_label"],
                })
            
        # ── GET /api/feedback/count ──────────────────────────────
             
        if method == "GET" and path.rstrip("/") == "/api/feedback/count":
            # tracks retraining thresholds via dataset_versions table
            return JSONResponse({"info": "Feedback tracked in Postgres feedback_events"})

        # ── GET /api/feedback/all ────────────────────────────────
        if method == "GET" and path.rstrip("/") == "/api/feedback/all":
            return JSONResponse({"info": "Query feedback_events table directly"})

        # ── GET /api/meetings ─────────────────────────────────────
        if method == "GET" and path.rstrip("/") == "/api/meetings":
            meetings = self.store.list_meetings()
            return JSONResponse({"meetings": meetings, "total": len(meetings)})

        return JSONResponse({"error": "Not found"}, status_code=404)


# ═══════════════════════════════════════════════════════════════════════════════
# RECAP UI — serves recap_ui.html at /ui?meeting_id=<id>
# ═══════════════════════════════════════════════════════════════════════════════

@serve.deployment(name="recap_ui", num_replicas=1)
class RecapUIDeployment:
    def __init__(self):
        self._path = Path(__file__).parent / "recap_ui.html"
        if not self._path.exists():
            print(f"[recap_ui] WARNING: {self._path} not found")
        else:
            print(f"[recap_ui] Will serve UI from {self._path} (disk-read per request)")

    async def __call__(self, request: Request) -> Response:
        if self._path.exists():
            html = self._path.read_text()
        else:
            html = "<h1>recap_ui.html not found</h1>"
        return Response(
            content=html,
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )
    
# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════

@serve.deployment(name="health", num_replicas=1)
class HealthDeployment:
    async def __call__(self, request: Request) -> JSONResponse:
        # Health actor has no num_gpus, so CUDA_VISIBLE_DEVICES="" in this process.
        # Use Ray cluster resources to detect GPU presence, nvidia-smi for details.
        cluster_gpus = ray.cluster_resources().get("GPU", 0)
        gpu_available = cluster_gpus > 0
        gpu_name = "none"
        gpu_mem_gb = 0.0
        if gpu_available:
            try:
                import subprocess
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if r.returncode == 0:
                    parts = r.stdout.strip().split(", ")
                    gpu_name = parts[0]
                    gpu_mem_gb = round(int(parts[1]) / 1024, 1) if len(parts) > 1 else 0.0
            except Exception:
                gpu_name = "unknown"
        return JSONResponse(content={
            "status": "ok",
            "mode": "ray_serve",
            "device": "cuda" if gpu_available else "cpu",
            "gpu": gpu_name,
            "gpu_memory_gb": gpu_mem_gb
        })


# ═══════════════════════════════════════════════════════════════════════════════
# BIND & RUN
# ═══════════════════════════════════════════════════════════════════════════════

ray.init(ignore_reinit_error=True, num_gpus=1, dashboard_host="0.0.0.0")

# Each deployment looks up the shared metrics actor at runtime via
# serve.get_deployment_handle("metrics", app_name="metrics"), so there
# is exactly ONE MetricsDeployment actor that all apps write to.
metrics    = MetricsDeployment.bind()
segmenter  = SegmenterDeployment.bind()   # 0.3 GPU
summarizer = SummarizerDeployment.bind()  # 0.7 GPU  (total = 1.0)
health     = HealthDeployment.bind()
recap      = RecapPipelineDeployment.bind()
recap_api  = RecapAPIDeployment.bind()
recap_ui   = RecapUIDeployment.bind()

#serve.start(http_options={"host": "0.0.0.0", "port": 8000})
# HTML recap UI will run in the browser and make API calls to your server (e.g. fetch('http://192.5.87.115:8000/recap'))
#  Browsers block these "cross-origin" requests by default unless the server explicitly says "yes, other origins can talk to me." That's CORS — Cross-Origin Resource Sharing.


serve.start(http_options={
    "host": "0.0.0.0",
    "port": 8000,
    "middlewares": [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )
    ]
})

# metrics MUST be deployed first — other actors look it up by name on init.
serve.run(metrics,    name="metrics",    route_prefix="/metrics")
serve.run(health,     name="health",     route_prefix="/health")
serve.run(segmenter,  name="segmenter",  route_prefix="/segment")
serve.run(summarizer, name="summarizer", route_prefix="/summarize")
#serve.run(recap,      name="recap",      route_prefix="/recap")
serve.run(recap,      name="recap",      route_prefix="/recap")
serve.run(recap_api,  name="recap_api",  route_prefix="/api")
serve.run(recap_ui,   name="recap_ui",   route_prefix="/ui")

print("=" * 60)
print("Ray Serve running at http://0.0.0.0:8000")
print("Endpoints: /health, /segment, /summarize, /recap, /metrics")
print("=" * 60)

import signal
signal.pause()