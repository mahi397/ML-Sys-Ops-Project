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
import os
import time
import psutil
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

BOUNDARY_THRESHOLD = float(os.getenv("BOUNDARY_THRESHOLD", "0.5"))
MODEL_PATH         = os.getenv("MODEL_PATH", "roberta-base")
LLM_MODEL_PATH     = os.getenv("LLM_MODEL_PATH", "")
MAX_SEGMENT_UTTERANCES = int(os.getenv("MAX_SEGMENT_UTTERANCES", "200"))
DEVICE             = "cuda"  # overridden per-actor in __init__


def format_window_for_roberta(window: list) -> str:
    sorted_window = sorted(window, key=lambda u: u["position"])
    return " ".join(f"[SPEAKER_{u['speaker']}]: {u['text']}" for u in sorted_window)


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
            ['model_name'], registry=self.registry
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

    def _update_system(self):
        try:
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated(0) / 1024 / 1024
                mem_total = torch.cuda.get_device_properties(0).total_mem / 1024 / 1024
                self.gpu_mem_used.set(mem_used)
                self.gpu_mem_total.set(mem_total)
        except:
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
            self.model_loaded.labels(model_name=data["model_name"]).set(
                1 if data["model_loaded"] else 0
            )
        if "recap_duration" in data:
            self.recap_duration.observe(data["recap_duration"])
        if "recap_segments" in data:
            self.recap_segments.observe(data["recap_segments"])

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
)
class SegmenterDeployment:
    def __init__(self):
        self.metrics = serve.get_deployment_handle("metrics", app_name="metrics")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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

        # Report model loaded
        self.metrics.record.remote({
            "endpoint": "segment", "model_loaded": True,
            "model_name": "roberta_segmenter"
        })
        print(f"[segmenter] Ready on {self.device}, threshold={self.threshold}")

    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.05)
    async def batch_predict(self, requests: list) -> list:
        batch_size = len(requests)

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

    async def __call__(self, request: Request) -> JSONResponse:
        start = time.time()
        try:
            body = await request.json()
            if "window" not in body:
                self.metrics.record.remote({"endpoint": "segment", "status": "error"})
                return JSONResponse({"error": "Missing 'window'"}, status_code=400)

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
            self.metrics.record.remote({
                "endpoint": "summarize", "model_loaded": True,
                "model_name": "mistral_summarizer"
            })
            print(f"[summarizer] LLM loaded from {LLM_MODEL_PATH}")
        else:
            self.metrics.record.remote({
                "endpoint": "summarize", "model_loaded": False,
                "model_name": "mistral_summarizer"
            })
            print(f"[summarizer] No LLM at '{LLM_MODEL_PATH}' — draft mode")

    def _summarize(self, body: dict) -> dict:
        meeting_id = body.get("meeting_id", "unknown")
        segment_id = body.get("segment_id", 0)
        t_start = body.get("t_start", 0)
        t_end = body.get("t_end", 0)

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
        self.summarizer = serve.get_deployment_handle("summarizer", app_name="summarizer")

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

    async def __call__(self, request: Request) -> JSONResponse:
        start = time.time()
        try:
            body = await request.json()
            meeting_id = body.get("meeting_id", "unknown")
            utterances = body.get("utterances", [])

            if not utterances or len(utterances) < 2:
                self.metrics.record.remote({"endpoint": "recap", "status": "error"})
                return JSONResponse({"error": "Need at least 2 utterances"}, status_code=400)

            # Stage A
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

            return JSONResponse(content={
                "meeting_id": meeting_id,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_segments": len(summaries),
                "processing_time_seconds": round(elapsed, 1),
                "recap": summaries
            })

        except Exception as e:
            self.metrics.record.remote({"endpoint": "recap", "status": "error"})
            return JSONResponse({"error": str(e)}, status_code=500)


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

serve.start(http_options={"host": "0.0.0.0", "port": 8000})

# metrics MUST be deployed first — other actors look it up by name on init.
serve.run(metrics,    name="metrics",    route_prefix="/metrics")
serve.run(health,     name="health",     route_prefix="/health")
serve.run(segmenter,  name="segmenter",  route_prefix="/segment")
serve.run(summarizer, name="summarizer", route_prefix="/summarize")
serve.run(recap,      name="recap",      route_prefix="/recap")

print("=" * 60)
print("Ray Serve running at http://0.0.0.0:8000")
print("Endpoints: /health, /segment, /summarize, /recap, /metrics")
print("=" * 60)

import signal
signal.pause()