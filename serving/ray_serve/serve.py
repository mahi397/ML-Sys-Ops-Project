"""
Ray Serve deployment for Jitsi Meeting Recap pipeline
Replaces FastAPI (app/main.py) with independent model scaling + Prometheus metrics
Matches the EXACT same JSON contract as the FastAPI version
  - SegmentInput / SegmentOutput  (from app/schemas.py)
  - SummarizeInput / SummarizeOutput
  - recap_worker.py pipeline logic

Endpoints:
  GET  /health      → system health
  POST /segment     → Stage A: RoBERTa boundary detection
  POST /summarize   → Stage B: Mistral-7B summarization
  POST /recap       → Full pipeline: segment all → assemble → summarize each
  GET  /metrics     → Prometheus scrape endpoint
"""

import ray
from ray import serve
import torch
import numpy as np
import json
import os
import time
import psutil
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — same env vars as app/config.py
# ═══════════════════════════════════════════════════════════════════════════════

BOUNDARY_THRESHOLD = float(os.getenv("BOUNDARY_THRESHOLD", "0.5"))
MODEL_PATH         = os.getenv("MODEL_PATH", "roberta-base")
LLM_MODEL_PATH     = os.getenv("LLM_MODEL_PATH", "")
MAX_SEGMENT_UTTERANCES = int(os.getenv("MAX_SEGMENT_UTTERANCES", "200"))
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════════════════════════
# PROMETHEUS METRICS
# ═══════════════════════════════════════════════════════════════════════════════

REGISTRY = CollectorRegistry()

REQUEST_COUNT = Counter(
    'jitsi_requests_total', 'Total requests by endpoint and status',
    ['endpoint', 'status'], registry=REGISTRY
)
REQUEST_LATENCY = Histogram(
    'jitsi_request_latency_seconds', 'Request latency in seconds',
    ['endpoint'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
    registry=REGISTRY
)
ACTIVE_REQUESTS = Gauge(
    'jitsi_active_requests', 'Currently processing requests',
    ['endpoint'], registry=REGISTRY
)
BATCH_SIZE = Histogram(
    'jitsi_batch_size', 'Batch sizes for batched inference',
    ['model'], buckets=[1, 2, 4, 8, 16], registry=REGISTRY
)
CONFIDENCE_SCORE = Histogram(
    'jitsi_boundary_confidence', 'Boundary prediction confidence scores',
    [], buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=REGISTRY
)
SEGMENTS_DETECTED = Counter(
    'jitsi_segments_detected_total', 'Total topic boundaries detected',
    registry=REGISTRY
)
SUMMARY_LENGTH = Histogram(
    'jitsi_summary_length_chars', 'Summary length in characters',
    [], buckets=[50, 100, 200, 500, 1000, 2000, 5000], registry=REGISTRY
)
GPU_MEMORY_USED = Gauge(
    'jitsi_gpu_memory_used_mb', 'GPU memory used in MB', registry=REGISTRY
)
GPU_MEMORY_TOTAL = Gauge(
    'jitsi_gpu_memory_total_mb', 'GPU memory total in MB', registry=REGISTRY
)
GPU_UTILIZATION = Gauge(
    'jitsi_gpu_utilization_percent', 'GPU utilization %', registry=REGISTRY
)
CPU_UTILIZATION = Gauge(
    'jitsi_cpu_utilization_percent', 'CPU utilization %', registry=REGISTRY
)
RAM_USED = Gauge(
    'jitsi_ram_used_mb', 'RAM used in MB', registry=REGISTRY
)
MODEL_LOADED = Gauge(
    'jitsi_model_loaded', 'Whether model is loaded (1=yes, 0=no)',
    ['model_name'], registry=REGISTRY
)
SLA_VIOLATIONS = Counter(
    'jitsi_sla_violations_total', 'SLA violations by endpoint and type',
    ['endpoint', 'sla_type'], registry=REGISTRY
)
RECAP_SEGMENTS_PER_MEETING = Histogram(
    'jitsi_recap_segments_per_meeting', 'Number of segments per meeting recap',
    [], buckets=[1, 2, 3, 5, 8, 10, 15, 20], registry=REGISTRY
)
RECAP_DURATION = Histogram(
    'jitsi_recap_duration_seconds', 'Total recap pipeline duration',
    [], buckets=[5, 10, 30, 60, 120, 180, 300, 600], registry=REGISTRY
)
MODEL_INFO = Info('jitsi_model', 'Model version info', registry=REGISTRY)


def update_system_metrics():
    """Update GPU/CPU/RAM gauges."""
    try:
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated(0) / 1024 / 1024
            mem_total = torch.cuda.get_device_properties(0).total_mem / 1024 / 1024
            GPU_MEMORY_USED.set(mem_used)
            GPU_MEMORY_TOTAL.set(mem_total)
            GPU_UTILIZATION.set((mem_used / mem_total) * 100 if mem_total > 0 else 0)
    except Exception:
        pass
    CPU_UTILIZATION.set(psutil.cpu_percent())
    RAM_USED.set(psutil.virtual_memory().used / 1024 / 1024)


# ═══════════════════════════════════════════════════════════════════════════════
# TOKENIZATION — same as app/tokenize.py
# ═══════════════════════════════════════════════════════════════════════════════

def format_window_for_roberta(window: list) -> str:
    """Same format as app/tokenize.py — matches training data format."""
    sorted_window = sorted(window, key=lambda u: u["position"])
    return " ".join(f"[SPEAKER_{u['speaker']}]: {u['text']}" for u in sorted_window)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE A: RoBERTa SEGMENTER
# ═══════════════════════════════════════════════════════════════════════════════

@serve.deployment(
    name="segmenter",
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.3},
    max_ongoing_requests=3,
)
class SegmenterDeployment:
    def __init__(self):
        self.device = DEVICE
        self.threshold = BOUNDARY_THRESHOLD
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        if os.path.exists(MODEL_PATH):
            self.model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
            print(f"[segmenter] Loaded fine-tuned model from {MODEL_PATH}")
        else:
            self.model = RobertaForSequenceClassification.from_pretrained(
                "roberta-base", num_labels=2
            )
            print(f"[segmenter] WARNING: No model at {MODEL_PATH}, using base weights")

        self.model.to(self.device)
        self.model.eval()
        MODEL_LOADED.labels(model_name="roberta_segmenter").set(1)
        MODEL_INFO.info({
            'segmenter_path': MODEL_PATH,
            'segmenter_device': self.device,
            'threshold': str(self.threshold)
        })
        print(f"[segmenter] Ready on {self.device}, threshold={self.threshold}")

    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.05)
    async def batch_predict(self, requests: list) -> list:
        """
        Native Ray Serve batching — groups up to 8 concurrent requests
        into a single GPU forward pass.
        """
        BATCH_SIZE.labels(model="segmenter").observe(len(requests))

        texts = []
        metadata = []
        for req in requests:
            texts.append(format_window_for_roberta(req["window"]))
            ti = req["transition_index"]
            metadata.append({
                "meeting_id": req["meeting_id"],
                "transition_index": ti,
                "meeting_offset_seconds": req["meeting_offset_seconds"],
                "t_boundary": req["window"][ti]["t_end"]
            })

        inputs = self.tokenizer(
            texts, return_tensors="pt", truncation=True,
            max_length=512, padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            boundary_probs = probs[:, 1].cpu().tolist()

        # Build responses — EXACT same format as SegmentOutput in schemas.py
        results = []
        for prob, meta in zip(boundary_probs, metadata):
            is_boundary = prob >= self.threshold
            CONFIDENCE_SCORE.observe(prob)
            if is_boundary:
                SEGMENTS_DETECTED.inc()
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
        """Called by RecapPipeline via DeploymentHandle."""
        result = await self.batch_predict(body)
        return result

    async def __call__(self, request: Request) -> JSONResponse:
        """HTTP endpoint: POST /segment — same contract as FastAPI."""
        start = time.time()
        ACTIVE_REQUESTS.labels(endpoint="segment").inc()
        try:
            body = await request.json()
            if "window" not in body:
                REQUEST_COUNT.labels(endpoint="segment", status="error").inc()
                return JSONResponse({"error": "Missing 'window' field"}, status_code=400)

            result = await self.predict_single(body)
            latency = time.time() - start
            REQUEST_LATENCY.labels(endpoint="segment").observe(latency)
            REQUEST_COUNT.labels(endpoint="segment", status="success").inc()
            if latency > 2.0:
                SLA_VIOLATIONS.labels(endpoint="segment", sla_type="latency_2s").inc()
            update_system_metrics()
            return JSONResponse(content=result)
        except Exception as e:
            REQUEST_COUNT.labels(endpoint="segment", status="error").inc()
            return JSONResponse({"error": str(e)}, status_code=500)
        finally:
            ACTIVE_REQUESTS.labels(endpoint="segment").dec()


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE B: LLM SUMMARIZER
# ═══════════════════════════════════════════════════════════════════════════════

@serve.deployment(
    name="summarizer",
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.7},
    max_ongoing_requests=3,
)
class SummarizerDeployment:
    def __init__(self):
        self.llm = None
        if LLM_MODEL_PATH and os.path.exists(LLM_MODEL_PATH):
            from llama_cpp import Llama
            self.llm = Llama(
                model_path=LLM_MODEL_PATH,
                n_gpu_layers=-1,
                n_ctx=4096,
                verbose=False
            )
            MODEL_LOADED.labels(model_name="mistral_summarizer").set(1)
            print(f"[summarizer] LLM loaded from {LLM_MODEL_PATH}")
        else:
            MODEL_LOADED.labels(model_name="mistral_summarizer").set(0)
            print(f"[summarizer] No LLM at '{LLM_MODEL_PATH}' — draft mode")

    def _summarize(self, body: dict) -> dict:
        """Core summarization logic — same as app/llm.py + recap_worker.py contract."""
        meeting_id = body.get("meeting_id", "unknown")
        segment_id = body.get("segment_id", 0)
        t_start = body.get("t_start", 0)
        t_end = body.get("t_end", 0)

        # No LLM → draft response (same as FastAPI fallback)
        if self.llm is None:
            return {
                "meeting_id": meeting_id, "segment_id": segment_id,
                "t_start": t_start, "t_end": t_end,
                "topic_label": "", "summary_bullets": [], "status": "draft"
            }

        try:
            utterances = body.get("utterances", [])[:MAX_SEGMENT_UTTERANCES]
            transcript = "\n".join(
                f"[SPEAKER_{u['speaker']}]: {u['text']}" for u in utterances
            )
            seg_ctx = body.get("meeting_context", {})
            prompt = f"""Summarize this meeting segment. Respond with JSON only, no other text.

Segment {seg_ctx.get('segment_index_in_meeting', 1)} of {seg_ctx.get('total_segments', 1)}.

Transcript:
{transcript}

JSON format:
{{"topic_label": "2-5 word label", "summary_bullets": ["point 1", "point 2", "point 3"]}}"""

            response = self.llm(prompt, max_tokens=300, temperature=0.1, stop=["```"])
            text = response["choices"][0]["text"].strip()
            start_idx = text.find("{")
            end_idx = text.rfind("}") + 1
            parsed = json.loads(text[start_idx:end_idx])

            return {
                "meeting_id": meeting_id, "segment_id": segment_id,
                "t_start": t_start, "t_end": t_end,
                "topic_label": parsed["topic_label"],
                "summary_bullets": parsed["summary_bullets"],
                "status": "complete"
            }
        except Exception as e:
            print(f"[summarizer] Failed for segment {segment_id}: {e}")
            return {
                "meeting_id": meeting_id, "segment_id": segment_id,
                "t_start": t_start, "t_end": t_end,
                "topic_label": "", "summary_bullets": [], "status": "draft"
            }

    async def summarize_dict(self, body: dict) -> dict:
        """Called by RecapPipeline via DeploymentHandle — takes dict, returns dict."""
        start = time.time()
        result = self._summarize(body)
        latency = time.time() - start
        REQUEST_LATENCY.labels(endpoint="summarize").observe(latency)
        status = "success" if result["status"] != "error" else "error"
        REQUEST_COUNT.labels(endpoint="summarize", status=status).inc()
        if result["status"] == "complete":
            SUMMARY_LENGTH.observe(
                len(result.get("topic_label", "")) +
                len(" ".join(result.get("summary_bullets", [])))
            )
        if latency > 30.0:
            SLA_VIOLATIONS.labels(endpoint="summarize", sla_type="latency_30s").inc()
        update_system_metrics()
        return result

    async def __call__(self, request: Request) -> JSONResponse:
        """HTTP endpoint: POST /summarize — same contract as FastAPI."""
        start = time.time()
        ACTIVE_REQUESTS.labels(endpoint="summarize").inc()
        try:
            body = await request.json()
            result = self._summarize(body)
            latency = time.time() - start
            REQUEST_LATENCY.labels(endpoint="summarize").observe(latency)
            status = "success" if result["status"] != "error" else "error"
            REQUEST_COUNT.labels(endpoint="summarize", status=status).inc()
            if result["status"] == "complete":
                SUMMARY_LENGTH.observe(
                    len(result.get("topic_label", "")) +
                    len(" ".join(result.get("summary_bullets", [])))
                )
            if latency > 30.0:
                SLA_VIOLATIONS.labels(endpoint="summarize", sla_type="latency_30s").inc()
            update_system_metrics()
            return JSONResponse(content=result)
        except Exception as e:
            REQUEST_COUNT.labels(endpoint="summarize", status="error").inc()
            return JSONResponse({"error": str(e)}, status_code=500)
        finally:
            ACTIVE_REQUESTS.labels(endpoint="summarize").dec()


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE: /recap
# Same logic as worker/recap_worker.py but via Ray Serve DeploymentHandles
# ═══════════════════════════════════════════════════════════════════════════════

@serve.deployment(
    name="recap_pipeline",
    num_replicas=1,
    max_ongoing_requests=2,
)
class RecapPipelineDeployment:
    def __init__(self, segmenter_handle, summarizer_handle):
        self.segmenter = segmenter_handle
        self.summarizer = summarizer_handle

    def _build_windows(self, utterances, window_size=7):
        """Same as recap_worker.py build_windows()."""
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
        """Same as recap_worker.py assemble_segments()."""
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

    async def __call__(self, request: Request) -> JSONResponse:
        """POST /recap — full meeting recap pipeline."""
        start = time.time()
        ACTIVE_REQUESTS.labels(endpoint="recap").inc()
        try:
            body = await request.json()
            meeting_id = body.get("meeting_id", "unknown")
            utterances = body.get("utterances", [])

            if not utterances or len(utterances) < 2:
                REQUEST_COUNT.labels(endpoint="recap", status="error").inc()
                return JSONResponse({"error": "Need at least 2 utterances"}, status_code=400)

            # ── Stage A: segment all windows ─────────────────────────────
            windows = self._build_windows(utterances)

            # Fire all segmentation requests via DeploymentHandle
            seg_refs = []
            for w in windows:
                payload = {"meeting_id": meeting_id, **w}
                ref = self.segmenter.predict_single.remote(payload)
                seg_refs.append(ref)

            # Gather results
            decisions = []
            for ref in seg_refs:
                result = await ref
                decisions.append(result)

            # ── Assemble segments ────────────────────────────────────────
            segments = self._assemble_segments(utterances, decisions)
            RECAP_SEGMENTS_PER_MEETING.observe(len(segments))
            print(f"[recap] {meeting_id}: {len(segments)} segments from {len(utterances)} utterances")

            # ── Stage B: summarize each segment ──────────────────────────
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
                # Call summarizer via DeploymentHandle (dict→dict, no HTTP)
                sum_result = await self.summarizer.summarize_dict.remote(sum_payload)
                summaries.append(sum_result)

            elapsed = time.time() - start
            REQUEST_LATENCY.labels(endpoint="recap").observe(elapsed)
            REQUEST_COUNT.labels(endpoint="recap", status="success").inc()
            RECAP_DURATION.observe(elapsed)
            if elapsed > 300.0:
                SLA_VIOLATIONS.labels(endpoint="recap", sla_type="latency_300s").inc()
            update_system_metrics()

            # Same output format as recap_worker.py
            return JSONResponse(content={
                "meeting_id": meeting_id,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_segments": len(summaries),
                "processing_time_seconds": round(elapsed, 1),
                "recap": summaries
            })

        except Exception as e:
            REQUEST_COUNT.labels(endpoint="recap", status="error").inc()
            return JSONResponse({"error": str(e)}, status_code=500)
        finally:
            ACTIVE_REQUESTS.labels(endpoint="recap").dec()


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS ENDPOINT (Prometheus scrapes this)
# ═══════════════════════════════════════════════════════════════════════════════

@serve.deployment(name="metrics", num_replicas=1)
class MetricsDeployment:
    async def __call__(self, request: Request) -> Response:
        update_system_metrics()
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST
        )


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════

@serve.deployment(name="health", num_replicas=1)
class HealthDeployment:
    async def __call__(self, request: Request) -> JSONResponse:
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "none"
        gpu_mem_gb = (torch.cuda.get_device_properties(0).total_mem / 1024**3
                      if gpu_available else 0)
        return JSONResponse(content={
            "status": "ok",
            "mode": "ray_serve",
            "device": "cuda" if gpu_available else "cpu",
            "gpu": gpu_name,
            "gpu_memory_gb": round(gpu_mem_gb, 1)
        })


# ═══════════════════════════════════════════════════════════════════════════════
# BIND & RUN
# ═══════════════════════════════════════════════════════════════════════════════

ray.init(ignore_reinit_error=True)

segmenter   = SegmenterDeployment.bind()
summarizer  = SummarizerDeployment.bind()
health      = HealthDeployment.bind()
metrics     = MetricsDeployment.bind()
recap       = RecapPipelineDeployment.bind(segmenter, summarizer)

serve.run(
    {
        "/health":     health,
        "/segment":    segmenter,
        "/summarize":  summarizer,
        "/recap":      recap,
        "/metrics":    metrics,
    },
    name="jitsi_recap",
    route_prefix="/",
    host="0.0.0.0",
    port=8000,
)

print("=" * 60)
print("Ray Serve running at http://0.0.0.0:8000")
print("Endpoints: /health, /segment, /summarize, /recap, /metrics")
print("Ray Dashboard: http://0.0.0.0:8265")
print("=" * 60)

import signal
signal.pause()