"""
Prometheus metrics server for Jitsi Ray Serve monitoring.
Runs alongside serve.py inside the same container.

- Exposes /metrics on port 9091 for Prometheus to scrape
- Polls Ray Serve /segment endpoint every 10s to track real request metrics
- Reads GPU stats from nvidia-smi (works from any process, no CUDA context needed)
- Tracks: request count, latency, confidence, GPU/CPU/RAM, SLA violations
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
from prometheus_client import (
    Counter, Histogram, Gauge, CollectorRegistry, generate_latest
)
import psutil
import subprocess
import threading
import time
import json

REGISTRY = CollectorRegistry()

# ── Request metrics ──────────────────────────────────────────────
REQUEST_COUNT = Counter(
    'jitsi_requests_total', 'Total requests by endpoint and status',
    ['endpoint', 'status'], registry=REGISTRY
)
REQUEST_LATENCY = Histogram(
    'jitsi_request_latency_seconds', 'Request latency in seconds',
    ['endpoint'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY
)
ACTIVE_REQUESTS = Gauge(
    'jitsi_active_requests', 'Currently processing requests',
    ['endpoint'], registry=REGISTRY
)

# ── Model metrics ────────────────────────────────────────────────
CONFIDENCE = Histogram(
    'jitsi_boundary_confidence', 'Boundary prediction confidence scores',
    [],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=REGISTRY
)
SEGMENTS_DETECTED = Counter(
    'jitsi_segments_detected_total', 'Total topic boundaries detected',
    registry=REGISTRY
)
BATCH_SIZE = Histogram(
    'jitsi_batch_size', 'Batch sizes for inference',
    ['model'], buckets=[1, 2, 4, 8, 16], registry=REGISTRY
)
MODEL_LOADED = Gauge(
    'jitsi_model_loaded', 'Whether model is loaded (1=yes)',
    ['model_name'], registry=REGISTRY
)
SUMMARY_LENGTH = Histogram(
    'jitsi_summary_length_chars', 'Summary length in characters',
    [], buckets=[50, 100, 200, 500, 1000, 2000, 5000], registry=REGISTRY
)

# ── System resource metrics (GPU via nvidia-smi) ─────────────────
GPU_MEMORY_USED = Gauge(
    'jitsi_gpu_memory_used_mb', 'GPU memory used in MB',
    registry=REGISTRY
)
GPU_MEMORY_TOTAL = Gauge(
    'jitsi_gpu_memory_total_mb', 'GPU memory total in MB',
    registry=REGISTRY
)
GPU_UTILIZATION = Gauge(
    'jitsi_gpu_utilization_percent', 'GPU utilization percentage',
    registry=REGISTRY
)
CPU_UTILIZATION = Gauge(
    'jitsi_cpu_utilization_percent', 'CPU utilization percentage',
    registry=REGISTRY
)
RAM_USED = Gauge(
    'jitsi_ram_used_mb', 'RAM used in MB',
    registry=REGISTRY
)

# ── SLA tracking ─────────────────────────────────────────────────
SLA_VIOLATIONS = Counter(
    'jitsi_sla_violations_total', 'SLA violations',
    ['endpoint', 'sla_type'], registry=REGISTRY
)

# ── Recap metrics ────────────────────────────────────────────────
RECAP_DURATION = Histogram(
    'jitsi_recap_duration_seconds', 'Full recap pipeline duration',
    [], buckets=[5, 10, 30, 60, 120, 180, 300, 600], registry=REGISTRY
)
RECAP_SEGMENTS = Histogram(
    'jitsi_recap_segments_per_meeting', 'Segments per meeting recap',
    [], buckets=[1, 2, 3, 5, 8, 10, 15, 20], registry=REGISTRY
)


# ═══════════════════════════════════════════════════════════════════
# SYSTEM METRICS — reads real GPU data from nvidia-smi
# ═══════════════════════════════════════════════════════════════════

def update_system_metrics():
    """Read GPU stats from nvidia-smi + CPU/RAM from psutil."""
    # GPU metrics via nvidia-smi (works from any process in the container)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            GPU_MEMORY_USED.set(float(parts[0]))    # MB
            GPU_MEMORY_TOTAL.set(float(parts[1]))    # MB
            GPU_UTILIZATION.set(float(parts[2]))     # %
    except Exception:
        pass

    # CPU and RAM
    CPU_UTILIZATION.set(psutil.cpu_percent())
    RAM_USED.set(psutil.virtual_memory().used / 1024 / 1024)


# ═══════════════════════════════════════════════════════════════════
# PROBE LOOP — sends real requests to Ray Serve to populate metrics
# ═══════════════════════════════════════════════════════════════════

SAMPLE_SEGMENT = {
    "meeting_id": "PROBE",
    "window": [
        {"position": 0, "speaker": "A", "t_start": 0.0, "t_end": 10.0,
         "text": "we need to finalize the interface before the next sprint"},
        {"position": 1, "speaker": "B", "t_start": 10.0, "t_end": 20.0,
         "text": "agreed the api contract should be locked down first"},
        {"position": 2, "speaker": "C", "t_start": 20.0, "t_end": 30.0,
         "text": "i can have a draft ready by thursday if that works"},
        {"position": 3, "speaker": "A", "t_start": 30.0, "t_end": 40.0,
         "text": "thursday works should we also loop in the frontend team"},
        {"position": 4, "speaker": "B", "t_start": 40.0, "t_end": 50.0,
         "text": "actually before that can we revisit the budget numbers"},
        {"position": 5, "speaker": "C", "t_start": 50.0, "t_end": 60.0,
         "text": "yes the q3 projections changed significantly last week"},
        {"position": 6, "speaker": "A", "t_start": 60.0, "t_end": 70.0,
         "text": "right we should update the forecast before the board meeting"}
    ],
    "transition_index": 3,
    "meeting_offset_seconds": 0.0
}


def probe_loop():
    """
    Periodically sends real requests to Ray Serve endpoints.
    This populates Prometheus metrics with actual latency, confidence,
    throughput data — visible in Grafana dashboards.
    """
    import requests as req

    # Wait for Ray Serve to finish loading models
    print("[metrics] Waiting 30s for Ray Serve to start...")
    time.sleep(30)

    while True:
        try:
            # ── Probe /health ────────────────────────────────────
            r = req.get("http://localhost:8000/health", timeout=5)
            if r.status_code == 200:
                data = r.json()
                MODEL_LOADED.labels(model_name="ray_serve").set(1)
                # Check if GPU is reported
                if data.get("gpu", "none") != "none":
                    MODEL_LOADED.labels(model_name="gpu_available").set(1)

            # ── Probe /segment with a real request ───────────────
            t0 = time.time()
            r = req.post(
                "http://localhost:8000/segment",
                json=SAMPLE_SEGMENT,
                timeout=10
            )
            latency = time.time() - t0

            if r.status_code == 200:
                result = r.json()

                # Record success
                REQUEST_COUNT.labels(endpoint="segment", status="success").inc()
                REQUEST_LATENCY.labels(endpoint="segment").observe(latency)
                BATCH_SIZE.labels(model="segmenter").observe(1)

                # Record confidence
                conf = result.get("boundary_probability", 0)
                CONFIDENCE.observe(conf)

                # Record boundary detection
                if result.get("is_boundary", False):
                    SEGMENTS_DETECTED.inc()

                # SLA check: segmentation p95 should be < 2s
                if latency > 2.0:
                    SLA_VIOLATIONS.labels(
                        endpoint="segment", sla_type="latency_2s"
                    ).inc()

                # Model is working
                MODEL_LOADED.labels(model_name="roberta_segmenter").set(1)
            else:
                REQUEST_COUNT.labels(endpoint="segment", status="error").inc()

        except req.exceptions.ConnectionError:
            # Ray Serve not ready yet
            print("[metrics] Ray Serve not reachable, retrying...")
        except Exception as e:
            print(f"[metrics] Probe error: {e}")

        # Update system metrics every loop
        update_system_metrics()

        # Probe every 10 seconds
        time.sleep(10)


# ═══════════════════════════════════════════════════════════════════
# HTTP SERVER — Prometheus scrapes this
# ═══════════════════════════════════════════════════════════════════

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        update_system_metrics()
        data = generate_latest(REGISTRY)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        # Suppress request logs to keep output clean
        pass


if __name__ == "__main__":
    # Start probe loop in background thread
    threading.Thread(target=probe_loop, daemon=True).start()

    print("=" * 50)
    print("[metrics] Prometheus metrics server on :9091")
    print("[metrics] Probe polling Ray Serve every 10s")
    print("=" * 50)

    HTTPServer(("0.0.0.0", 9091), MetricsHandler).serve_forever()