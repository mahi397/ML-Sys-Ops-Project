# Jitsi ML Serving — NeuralOps

End-to-end ML serving system for **topic segmentation + meeting summarization** built on Ray Serve 2.9.3 with GPU acceleration. Deployed on Chameleon Cloud (h100, Quadro RTX 6000, 24 GB VRAM(Ray serve)).

---

## Quick Access

| Service | URL | Credentials |
|---|---|---|
| API / Health | http://192.5.87.115:8000/health | — |
| Recap UI | http://192.5.87.115:8000/ui | — |
| Grafana | http://192.5.87.115:3000 | `admin` / `admin` |
| Prometheus | http://192.5.87.115:9090 | — |
| Alertmanager | http://192.5.87.115:9093 | — |
| Ray Dashboard | http://192.5.87.115:8265 | — |

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [End-to-End Data Flow](#end-to-end-data-flow)
3. [Directory Structure](#directory-structure)
4. [Ray Serve (Primary)](#ray-serve-primary)
   - [Deployments](#deployments)
   - [API Endpoints](#api-endpoints)
   - [Request Flow](#request-flow)
   - [MLflow Integration](#mlflow-integration)
   - [Metrics & Monitoring](#metrics--monitoring)
5. [Database & Persistence](#database--persistence)
6. [FastAPI Baseline](#fastapi-baseline)
7. [Docker Compose Services](#docker-compose-services)
8. [Container Build](#container-build)
9. [Setup & Deployment](#setup--deployment)
10. [Environment Variables](#environment-variables)
11. [Benchmarking](#benchmarking)
12. [Edge Cases & Validation](#edge-cases--validation)
13. [Monitoring Stack](#monitoring-stack)
14. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
                        ┌─────────────────────────────────────────────┐
                        │              Ray Serve (Port 8000)           │
                        │                                              │
  Client / Jitsi  ───►  │  /recap ──► RecapPipelineDeployment         │
                        │               │                              │
                        │               ├──► SegmenterDeployment       │
                        │               │    (RoBERTa, GPU 0.25*2)        │
                        │               │    @serve.batch (8 req/pass) │
                        │               │                              │
                        │               └──► SummarizerDeployment      │
                        │                    (Mistral-7B, GPU 0.5)     │
                        │                                              │
                        │  /segment ──► SegmenterDeployment (raw)      │
                        │  /summarize ► SummarizerDeployment (raw)     │
                        │  /metrics  ► MetricsDeployment               │
                        │  /health   ► HealthDeployment                │
                        │  /ui       ► RecapUIDeployment               │
                        │ /retrain, /rollback                          │
                        └──────────────────────┬──────────────────────┘
                                               │  save_recap()
                                               │  save_utterances()
                                               ▼
                                  ┌────────────────────────┐
                                  │   SQLite Database       │
                                  │   recaps.db             │
                                  │   (RecapStore)          │
                                  └────────────┬───────────┘
                                               │  read via RecapDeployment
                                               ▼
                                  ┌────────────────────────┐
                                  │   /api/* + /ui          │
                                  │   (browser recap viewer)│
                                  └────────────────────────┘
                                          │
                        ┌─────────────────▼──────────────────┐
                        │   Prometheus (Port 9090)            │
                        │   Scrapes ray-serve:8000/metrics    │
                        └─────────────────┬──────────────────┘
                                          │
                        ┌─────────────────▼──────────────────┐
                        │   Grafana (Port 3000)               │
                        │   Jitsi ML Serving Dashboard        │
                        └────────────────────────────────────┘
```

**Two ML stages in sequence:**
- **Stage A — RoBERTa** (`roberta-base` fine-tuned): Detects topic boundaries between utterance pairs
- **Stage B — Mistral-7B** (`Q4_K_M` quantized GGUF via `llama-cpp-python`): Generates topic labels + bullet-point summaries per segment

---

**GPU time-sharing summary:**

| Stage | Deployment | GPU Fraction | Concurrency |
|---|---|---|---|
| A — Boundary detection | SegmenterDeployment | 0.25*2 | up to 10 (batched 8/pass) |
| B — Summarization | SummarizerDeployment | 0.5 | up to 3 |
| Total | — | 1.0 (exactly fills RTX 6000) | — |

Stages A and B share the single Quadro RTX 6000 via Ray's fractional GPU allocation. Stage A windows run in parallel (fanned out via `predict_single.remote()`); Stage B segments run sequentially because Mistral holds 70% of VRAM and cannot safely batch multiple 7B inference calls simultaneously.

---

## Directory Structure

```
serving/
├── ray_serve/                   # PRIMARY: Ray Serve distributed system
│   ├── serve.py                 # All deployments, endpoints, MLflow integration
│   ├── storage.py               # SQLite-backed recap & utterance store (RecapStore)
│   ├── metrics_server.py        # Standalone metrics probe (port 9091)
│   ├── benchmark_ray.py         # Load testing for Ray endpoints
│   ├── recap_ui.html            # Browser UI for viewing meeting recaps
│   ├── Dockerfile.ray           # Container build (PyTorch 2.1 + CUDA 12.1)
│   └── requirements_ray.txt     # Ray-specific Python dependencies
│
├── app/                         # BASELINE: FastAPI serving (non-distributed)
│   ├── main.py                  # FastAPI entry point
│   ├── model.py                 # RoBERTa PyTorch loader
│   ├── model_onnx.py            # ONNX Runtime backend
│   ├── llm.py                   # Mistral loader (llama-cpp-python)
│   ├── schemas.py               # Pydantic request/response models
│   ├── tokenize.py              # Shared window formatter
│   └── config.py                # Serving mode + threshold config
│
├── monitoring/
│   ├── prometheus.yml           # Scrape config (ray-serve:8000)
│   ├── alerts.yml               # Alert rules (SLA, model health, resources)
│   ├── alertmanager.yml         # Alertmanager config
│   └── grafana/
│       └── provisioning/
│           ├── dashboards/      # Auto-provisioned dashboard JSON
│           └── datasources/     # Prometheus datasource config
│
├── worker/
│   └── recap_worker.py          # Async background recap processor
│
├── benchmark/
│   ├── benchmark.py             # FastAPI latency/throughput tests
│   └── benchmark_triton.py      # Triton backend benchmarks
│
├── triton_models/
│   └── roberta_segmenter/
│       ├── config.pbtxt         # Triton config (batch 256, ONNX backend)
│       └── 1/                   # Model version directory
│
├── scripts/
│   └── export_onnx.py           # PyTorch → ONNX model export
│
├── inputs/                      # Sample request payloads for testing
├── outputs/                     # Recap JSON output directory
├── models/                      # Downloaded model weights (gitignored)
│   ├── roberta-seg/             # Fine-tuned RoBERTa checkpoint
│   └── mistral-7b-instruct-v0.2.Q4_K_M.gguf
│
├── docker-compose.yml           # Full stack orchestration
├── setup.sh                     # Chameleon Cloud bootstrap script
├── Dockerfile                   # FastAPI baseline container
└── requirements.txt             # FastAPI baseline dependencies
```

---



# Serving — Ray Serve (Shruti Pangare)

Ray Serve deployment for the Jitsi Meeting Recap pipeline. Hosts the
two-stage inference system (RoBERTa segmenter + Mistral-7B summarizer),
exposes REST endpoints consumed by the data pipeline and the recap UI,
and handles MLflow-driven hot-reload, automated rollback, and
Prometheus/Grafana observability.

---

## Contents

```
serving/
├── ray_serve/
│   ├── serve.py              # All Ray Serve deployments (single entry point)
│   ├── storage.py            # JSONL + Postgres recap/feedback stores
│   ├── metrics_server.py     # Standalone Prometheus probe (sidecar)
│   ├── Dockerfile.ray        # Container image for the ray-serve service
│   └── requirements_ray.txt  # Python dependencies
├── monitoring/
│   ├── prometheus.yml        # Scrape config (ray-serve :9091, node-exporter :9100)
│   ├── alerts.yml            # Alerting rules (SLA, error rate, rollback trigger)
│   ├── alertmanager.yml      # Routes critical alerts → /rollback webhook
│   └── dashboards/           # Grafana JSON dashboards
├── benchmark/
│   └── benchmark_ray.py      # Load-testing script (segment, summarize, full meeting)
├── app/                      # FastAPI baseline (benchmarking only, not production)
├── recap_ui/                 # Static browser recap viewer
└── models/                   # Downloaded model weights (gitignored)
```

---

## Architecture

### Two-stage pipeline

```
                        POST /segment
Jitsi transcript ──────────────────────► SegmenterDeployment
(800 windows/meeting)                     RoBERTa-base fine-tuned
                                          0.25 GPU × up to 2 replicas
                                          @serve.batch  (max 8, 50ms wait)
                                          ~8 ms/window
                                               │
                                               │ boundary detected
                                               ▼
                        POST /summarize   SummarizerDeployment
                        (8 segments)      Mistral-7B (4-bit GGUF or HF)
                                          0.5 GPU × 1 replica
                                          ~4 s/segment
                                               │
                                               ▼
                        POST /recap       RecapPipelineDeployment
                                          orchestrates A→B, writes recap
```

### Deployment map

| Deployment | Route | GPU | Replicas | Notes |
|---|---|---|---|---|
| `SegmenterDeployment` | `/segment` | 0.25 (auto-scales to 0.5) | 1–2 | `@serve.batch` batching |
| `SummarizerDeployment` | `/summarize` | 0.5 | 1 | Mistral-7B, falls back to stub |
| `RecapPipelineDeployment` | `/recap` | — | 1 | Orchestrates A→B pipeline |
| `RecapAPIDeployment` | `/api/*` | — | 1 | meetings, utterances, feedback |
| `RecapUIDeployment` | `/ui` | — | 1 | Static recap browser |
| `HealthDeployment` | `/health` | — | 1 | System status + GPU info |
| `MetricsDeployment` | `/metrics` | — | 1 | Prometheus scrape endpoint |
| `RollbackDeployment` | `/rollback` | — | 1 | Alertmanager webhook |
| `RetrainDeployment` | `/retrain` | — | 1 | Alertmanager webhook (log only) |

Total GPU budget: 0.25×2 (segmenter) + 0.5 (summarizer) = 1.0 GPU (Quadro RTX 6000)

### Why Ray Serve (not FastAPI + Triton)

The two-stage pipeline needs independent scaling: Stage A is ~500× faster
than Stage B but handles ~100× more requests per meeting. A single FastAPI
process cannot scale them independently, and Triton requires ONNX conversion
plus a separate server. Ray Serve solves both:

- **Fractional GPU allocation** — `ray_actor_options={"num_gpus": 0.25/0.5}`
  gives each model dedicated VRAM with no contention.
- **`@serve.batch`** — groups concurrent `/segment` calls into a single GPU
  forward pass (same benefit as Triton's dynamic batching, no ONNX required).
- **Deployment handles** — `RecapPipelineDeployment` calls the segmenter and
  summarizer via `serve.get_deployment_handle(...)` without extra HTTP hops.
- **Built-in autoscaling** — segmenter scales 1→2 replicas when concurrent
  requests exceed 4 (`target_ongoing_requests=4`).

See [`RAY_SERVE_JUSTIFICATION.md`](ray_serve/RAY_SERVE_JUSTIFICATION.md) for
the full comparison table against FastAPI and Triton.

---

## API Reference

### `GET /health`

Returns system status including GPU info and currently loaded model version.

```json
{
  "status": "ok",
  "model_version": "mlflow@production_v7",
  "threshold": 0.38,
  "device": "cuda",
  "gpu": "Quadro RTX 6000",
  "gpu_memory_gb": 24.0
}
```

### `POST /segment`

Stage A — RoBERTa boundary detection.

**Request:**
```json
{
  "meeting_id": "ES2002a",
  "window": [
    {"position": 0, "speaker": "A", "t_start": 98.3, "t_end": 109.1, "text": "..."},
    {"position": 3, "speaker": "B", "t_start": 135.1, "t_end": 147.9, "text": "..."}
  ],
  "transition_index": 3,
  "meeting_offset_seconds": 98.3
}
```

**Batch format** (send multiple windows in one call):
```json
{
  "meeting_id": "ES2002a",
  "requests": [
    {"request_id": "t0", "window": [...], "transition_index": 3, "meeting_offset_seconds": 0.0},
    {"request_id": "t1", "window": [...], "transition_index": 5, "meeting_offset_seconds": 45.2}
  ]
}
```

**Response:**
```json
{
  "meeting_id": "ES2002a",
  "is_boundary": true,
  "boundary_probability": 0.812,
  "transition_after_position": 3,
  "t_boundary": 147.9,
  "segment_so_far": {"t_start": 98.3, "t_end": 147.9}
}
```

SLA: p95 < 2 000 ms. Target: ~8 ms per window under normal load.

### `POST /summarize`

Stage B — Mistral-7B summarization.

**Request:**
```json
{
  "meeting_id": "ES2002a",
  "segment_id": 1,
  "t_start": 98.3,
  "t_end": 204.8,
  "utterances": [
    {"speaker": "A", "text": "we need to finalize the interface..."},
    {"speaker": "B", "text": "agreed the api contract should be locked down first"}
  ],
  "total_utterances": 5,
  "meeting_context": {"total_segments": 3, "segment_index_in_meeting": 1}
}
```

**Response:**
```json
{
  "meeting_id": "ES2002a",
  "segment_id": 1,
  "topic_label": "API Interface Finalization",
  "summary_bullets": [
    "Team agreed to lock down API contract before frontend work begins.",
    "Draft interface to be ready by Thursday.",
    "Frontend team to be looped in after contract is finalized."
  ],
  "model_version": "mlflow@production_v7"
}
```

SLA: p95 < 30 s per segment.

### `POST /recap`

Full pipeline: segments all utterances, summarizes each segment, writes
recap to store, returns full structured recap.

**Request:**
```json
{
  "meeting_id": "ES2002a",
  "utterances": [
    {"speaker": "A", "t_start": 0.0, "t_end": 10.0, "text": "..."},
    ...
  ]
}
```

**Response:**
```json
{
  "meeting_id": "ES2002a",
  "total_segments": 3,
  "processing_time_seconds": 14.2,
  "recap": [
    {
      "segment_id": 1,
      "topic_label": "API Interface Finalization",
      "t_start": 0.0,
      "t_end": 147.9,
      "summary_bullets": ["..."]
    }
  ]
}
```

SLA: full 800-utterance meeting < 300 s.

### `GET  /api/meetings`
### `GET  /api/recap/{meeting_id}`
### `GET  /api/utterances/{meeting_id}`
### `POST /api/feedback`

Feedback payload:
```json
{
  "meeting_id": "ES2002a",
  "segment_summary_id": 42,
  "action": "remove_boundary",
  "before_payload": {"..."},
  "after_payload": {"..."}
}
```

Actions: `remove_boundary`, `add_boundary`, `overall_positive`, `overall_negative`.
Each feedback event is written to `feedback_events` in Postgres and counted
toward the retraining threshold watermark (see Training README).

### `GET /metrics`

Prometheus text format. Scraped by Prometheus at `:9091` (sidecar) and
directly at `:8000/metrics` (Ray Serve deployment).

### `POST /rollback`

Alertmanager webhook. Swaps `@production` ↔ `@fallback` aliases in MLflow.
Hot-reload thread picks up the new alias within `MODEL_RELOAD_INTERVAL_SECONDS`
(default 300 s). 5-minute cooldown prevents repeat fires.

### `POST /retrain`

Alertmanager webhook. Logs the alert event and reports pending feedback count.
Actual retraining is delegated to `retrain_watcher.py` (Training service).

---

## MLflow Hot-Reload

`SegmenterDeployment` runs a background thread (`_reload_loop`) that polls
the MLflow registry every `MODEL_RELOAD_INTERVAL_SECONDS` (default 300 s):

```
Poll MLflow @production alias
    │
    ├── Same version as current → skip
    │
    ├── New version detected
    │       ├── Download via --serve-artifacts proxy
    │       ├── Serving-side quality gate:
    │       │     test_pk >= MIN_TEST_PK (default 0.10)
    │       │     test_pk not worse than current by > 0.05
    │       ├── PASS → swap model under _model_lock (zero downtime)
    │       └── FAIL → skip version, log, wait for next poll
    │
    └── Download fails → mark version as _last_failed_version, skip next poll
```

Model load priority at startup:

1. MLflow `@production` alias (via artifact proxy)
2. MLflow `@fallback` alias (if production fails)
3. Local fine-tuned weights at `MODEL_PATH`
4. Base `roberta-base` weights (warning logged)

The decision threshold (`best_threshold`) is read from the model version's
MLflow tags on every load — it is never hardcoded in serving code.

---

## Data Stores

Two stores co-exist depending on `DATABASE_URL`:

| Store | When used | What it holds |
|---|---|---|
| `PostgresRecapStore` | `DATABASE_URL` is set | Meetings, utterances, segment summaries, feedback events |
| `RecapStore` (JSONL) | No DB / standalone | Same data in `/data/*.jsonl` files — drop-in fallback |

`storage.py` exposes the same method signatures for both. Swapping to
Postgres is a one-line env change — no API surface changes.

---

## Monitoring

### Prometheus metrics (collected in `MetricsDeployment`)

| Metric | Type | Description |
|---|---|---|
| `jitsi_requests_total{endpoint,status}` | Counter | Request count by endpoint and status |
| `jitsi_request_latency_seconds{endpoint}` | Histogram | Latency distribution per endpoint |
| `jitsi_active_requests{endpoint}` | Gauge | In-flight requests |
| `jitsi_boundary_confidence` | Histogram | RoBERTa output probability distribution |
| `jitsi_segments_detected_total` | Counter | Topic boundaries detected |
| `jitsi_summary_length_chars` | Histogram | Mistral output length |
| `jitsi_gpu_memory_used_mb` | Gauge | VRAM used (pynvml → torch fallback) |
| `jitsi_gpu_memory_total_mb` | Gauge | VRAM total |
| `jitsi_gpu_utilization_percent` | Gauge | GPU utilization % |
| `jitsi_cpu_utilization_percent` | Gauge | CPU utilization % |
| `jitsi_ram_used_mb` | Gauge | RAM used |
| `jitsi_model_loaded{model_name,model_version}` | Gauge | 1 when model is active |
| `jitsi_sla_violations_total{endpoint,sla_type}` | Counter | SLA breaches |
| `jitsi_recap_duration_seconds` | Histogram | Full /recap pipeline duration |
| `jitsi_recap_segments_per_meeting` | Histogram | Segments per recap |
| `jitsi_feedback_corrections_total{action}` | Counter | User corrections by type |
| `jitsi_feedback_events_total` | Gauge | Total feedback rows in Postgres |
| `jitsi_feedback_pending` | Gauge | Corrections above last retrain watermark |
| `jitsi_retrain_triggered_total` | Gauge | Retrain trigger events |
| `jitsi_retrain_completed_total` | Gauge | Completed retrain runs |
| `jitsi_retrain_passed_total` | Gauge | Runs that passed quality gates |
| `jitsi_retrain_last_f1` | Gauge | F1 from last completed retrain |
| `jitsi_retrain_last_pk` | Gauge | Pk from last completed retrain |

### Alerting rules → automated rollback

Alertmanager routes **critical** alerts to `POST /rollback`:

| Alert | Condition | Severity | Action |
|---|---|---|---|
| `HighErrorRate` | Error rate > 10% for 5 min | critical | → `/rollback` |
| `SegmentSLAViolation` | p95 latency > 2 s | warning | page only |
| `SummarizeSLAViolation` | p95 latency > 30 s | warning | page only |
| `RetrainingThresholdReached` | pending corrections ≥ threshold | info | → `/retrain` (log) |
| `GPUMemoryHigh` | VRAM > 90% for 10 min | warning | page only |
| `ModelNotLoaded` | `jitsi_model_loaded == 0` for 5 min | critical | → `/rollback` |

### Grafana dashboards

- **Serving overview** — request rate, latency percentiles, error rate, active requests
- **Model health** — boundary confidence distribution, segments detected, model version in use
- **Infrastructure** — GPU VRAM, GPU utilization, CPU, RAM
- **Retraining pipeline** — feedback accumulation, pending corrections, retrain pass/fail, last F1/Pk
- **SLA tracker** — real-time SLA violation counters per endpoint

Grafana: `http://<FLOATING_IP>:3000` (admin / admin)

---

## Safeguarding (Serving Layer)

The serving layer implements the following safeguarding principles within
its own scope. The full cross-team safeguarding plan is in the root README.

### Fairness
- Confidence scores (`boundary_probability`) are returned on every `/segment`
  response and stored in `utterance_transitions.pred_boundary_prob`.
  The Grafana confidence histogram exposes model calibration drift over time.
- Alerting is triggered if the confidence distribution shifts significantly
  (high proportion of predictions near 0.5 indicates model uncertainty).
- The decision threshold is read from the MLflow model version tag
  (`best_threshold`) — it is set by the training pipeline using a
  threshold-sweep on the validation set, ensuring fairness across meeting
  sizes and speaker counts, not a fixed value tuned by feel.

### Transparency
- Every API response includes `model_version` (e.g. `mlflow@production_v7`),
  so clients and logs always know which model produced a prediction.
- `/health` exposes the currently loaded version and threshold.
- Model load events (initial load, hot-reload, fallback activation) are
  printed to container logs and reflected in `jitsi_model_loaded` Grafana panel.
- The recap UI shows the model version alongside each recap.

### Explainability
- `/segment` returns `boundary_probability` alongside the binary `is_boundary`
  decision, giving downstream systems and human reviewers the raw model
  confidence — not just a yes/no verdict.
- The recap UI surfaces the confidence score on each segment boundary, making
  low-confidence splits visible to users before they give feedback.

### Accountability
- All user feedback events are written to Postgres `feedback_events` with
  `before_payload` and `after_payload`, creating an auditable correction trail.
- Rollback events are logged: the `/rollback` endpoint records which alerts
  fired, which model version was replaced, and which version it rolled back to.
- `jitsi_retrain_triggered_total`, `jitsi_retrain_completed_total`,
  `jitsi_retrain_passed_total`, and `jitsi_retrain_failed_total` make the
  full retraining lifecycle visible in Grafana.

### Robustness
- **Three-layer model fallback at startup**: MLflow `@production` →
  MLflow `@fallback` → local fine-tuned weights → base `roberta-base`.
  The server never starts without a model.
- **Serving-side quality gate on hot-reload**: a new model version is only
  swapped in if `test_pk >= MIN_TEST_PK` AND `test_pk` is not more than
  0.05 worse than the currently serving model. This prevents a regression
  introduced by a bad retrain from reaching production even if it passed
  training-side gates.
- **Rollback cooldown**: 5-minute cooldown on `/rollback` prevents alert
  storms from triggering repeated alias swaps.
- **`_last_failed_version` tracking**: if a model version download fails,
  serving skips that version on the next poll rather than retrying
  indefinitely and blocking future promotions.
- **Thread-safe model swap**: `_model_lock` ensures in-flight requests
  complete against the old model while the new one loads — no torn reads.

### Privacy
- Speaker identity is abstracted to `[SPEAKER_A/B/C]` tokens by
  `format_window_for_roberta()` before any text reaches the model.
  Raw speaker names or participant IDs never appear in model inputs.
- Feedback payloads stored in Postgres contain utterance indices and
  boundary labels — not participant identity.

---

## Serving-Side Monitoring Responsibilities

As per the project guidelines, the serving role owns:

1. **Model output monitoring** — `jitsi_boundary_confidence` histogram
   tracks distribution drift; `jitsi_segments_detected_total` tracks
   whether the model is predicting boundaries at a realistic rate.

2. **Operational metrics** — latency histograms, error rate counters,
   active request gauges, GPU/CPU/RAM gauges, SLA violation counters.

3. **User feedback monitoring** — `jitsi_feedback_corrections_total`
   by action type; `jitsi_feedback_pending` tracks how many corrections
   are waiting above the last retrain watermark.

4. **Rollback trigger** — Alertmanager fires `POST /rollback` on
   `HighErrorRate` or `ModelNotLoaded`. The rollback swaps `@production`
   ↔ `@fallback` in MLflow; hot-reload picks up the new alias within
   5 minutes without a container restart.

5. **Promotion trigger** — manual promotion from `candidate` → `production`
   in MLflow UI causes hot-reload to pick up the new version automatically.
   Automated promotion occurs when `retrain_watcher.py` (Training service)
   registers a model that passes all quality gates and sets the `candidate`
   alias; the serving hot-reload then validates it again with the
   serving-side quality gate before swapping.

---

## Running the Stack

### Start (via docker compose)

```bash
docker compose up -d ray-serve prometheus grafana alertmanager node-exporter
```

The `ray-serve` container runs `serve.py` as entrypoint and
`metrics_server.py` as a sidecar process.

### Verify

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics   # Prometheus text
```

### Benchmark

```bash
# 200 /segment requests, sequential
python3 benchmark/benchmark_ray.py --url http://localhost:8000 --n 200

# 200 /segment requests, 5 concurrent
python3 benchmark/benchmark_ray.py --url http://localhost:8000 --n 200 --concurrency 5

# Also benchmark /summarize
python3 benchmark/benchmark_ray.py --url http://localhost:8000 --n 50 --summarize

# Full 800-window meeting simulation (SLA: < 300 s)
python3 benchmark/benchmark_ray.py --url http://localhost:8000 --full-meeting

# Full /recap pipeline (segment + summarize)
python3 benchmark/benchmark_ray.py --url http://localhost:8000 --recap
```

### Force a rollback manually

```bash
curl -X POST http://localhost:8000/rollback \
  -H "Content-Type: application/json" \
  -d '{"alerts": [{"labels": {"alertname": "manual"}}]}'
```

### Reload model from MLflow (without restart)

Update the `@production` alias in MLflow UI → wait up to 5 minutes, or
reduce `MODEL_RELOAD_INTERVAL_SECONDS` env var for faster polling during
testing.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `roberta-base` | Local model weights path (fallback) |
| `LLM_MODEL_PATH` | `""` | Mistral weights path |
| `BOUNDARY_THRESHOLD` | `0.35` | Default boundary threshold (overridden by MLflow tag) |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server |
| `MODEL_ALIAS` | `production` | MLflow alias to serve |
| `MODEL_RELOAD_INTERVAL_SECONDS` | `300` | Hot-reload poll interval |
| `MIN_TEST_PK` | `0.10` | Minimum Pk score for serving-side quality gate |
| `DATABASE_URL` | `""` | Postgres connection string (uses JSONL fallback if unset) |
| `DATA_DIR` | `/data` | JSONL store directory |

---

## SLA Summary

| Endpoint | SLA | Alert threshold |
|---|---|---|
| `/segment` | p95 < 2 000 ms | warning at 2 s, critical rollback at 10% error rate |
| `/summarize` | p95 < 30 s | warning at 30 s |
| `/recap` (800 utterances) | < 300 s | checked in benchmark, not auto-alerted |
| Model hot-reload | < 5 min after alias change | `ModelNotLoaded` alert at 5 min |