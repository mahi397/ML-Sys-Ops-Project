# Jitsi ML Serving — NeuralOps

End-to-end ML serving system for **topic segmentation + meeting summarization** built on Ray Serve 2.9.3 with GPU acceleration. Deployed on Chameleon Cloud (h100, Quadro RTX 6000, 24 GB VRAM(Ray serve)).

---

## Quick Access

| Service | URL | Credentials |
|---|---|---|
| API / Health | http://192.5.87.115:8000/health | — |
| Recap UI | http://192.5.87.115:8000/ui | — |
| Grafana | http://192.5.87.115:3000 | `admin` / `jitsi2026` |
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
                        │               │    (RoBERTa, GPU 0.3)        │
                        │               │    @serve.batch (8 req/pass) │
                        │               │                              │
                        │               └──► SummarizerDeployment      │
                        │                    (Mistral-7B, GPU 0.7)     │
                        │                                              │
                        │  /segment ──► SegmenterDeployment (raw)      │
                        │  /summarize ► SummarizerDeployment (raw)     │
                        │  /metrics  ► MetricsDeployment               │
                        │  /health   ► HealthDeployment                │
                        │  /ui       ► RecapUIDeployment               │
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
| A — Boundary detection | SegmenterDeployment | 0.3 | up to 10 (batched 8/pass) |
| B — Summarization | SummarizerDeployment | 0.7 | up to 3 |
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

## Ray Serve (Primary)

### Deployments

`serve.py` defines **4 active Ray Serve deployments** wired together via deployment handles:

#### 1. `SegmenterDeployment` — `/segment`
RoBERTa-based topic boundary detector.

| Config | Value |
|---|---|
| GPU fraction | 0.3 |
| Replicas | 1 |
| Max concurrent requests | 10 |
| Batching | `@serve.batch(max_batch_size=8, batch_wait_timeout_s=0.05)` |
| Model | `RobertaForSequenceClassification` (2-class) |
| Input format | 7-utterance sliding window, formatted as `[SPEAKER_X]: text` |
| Threshold | `BOUNDARY_THRESHOLD` env var (default: 0.5) |

**Model load order (startup):**
1. `models:/jitsi-topic-segmenter@production` from MLflow registry
2. `models:/jitsi-topic-segmenter@fallback` if production fails
3. `MODEL_PATH` env var — local fine-tuned checkpoint (default: `roberta-base`)
4. `roberta-base` pretrained weights as last resort

**Hot-reload:** Polls MLflow every `MODEL_RELOAD_INTERVAL_SECONDS` (default 300s). Swaps model in-place under a threading lock without restarting the actor.

**Single window input:**
```json
{
  "meeting_id": "meeting_123",
  "window": [
    {"position": 0, "speaker": "A", "text": "...", "t_start": 0.0, "t_end": 5.0},
    {"position": 1, "speaker": "B", "text": "...", "t_start": 5.0, "t_end": 10.0}
  ],
  "transition_index": 3,
  "meeting_offset_seconds": 0.0
}
```

**Output:**
```json
{
  "meeting_id": "meeting_123",
  "transition_after_position": 3,
  "boundary_probability": 0.87,
  "is_boundary": true,
  "t_boundary": 30.0,
  "segment_so_far": { "t_start": 0.0, "t_end": 30.0 }
}
```

#### 2. `SummarizerDeployment` — `/summarize`
Mistral-7B-Instruct meeting summarizer via `llama-cpp-python` (CUDA build, `CMAKE_ARGS="-DLLAMA_CUBLAS=on"`).

| Config | Value |
|---|---|
| GPU fraction | 0.7 |
| Replicas | 1 |
| Max concurrent requests | 3 |
| Model | Mistral-7B-Instruct Q4_K_M GGUF |
| Context window | 4096 tokens (`n_ctx`) |
| GPU layers | All (`n_gpu_layers=-1`) |
| Max tokens | 300 |
| Temperature | 0.1 |
| JSON extraction | Finds first `{…}` block in response; up to 10 retries |

**Input:**
```json
{
  "meeting_id": "meeting_123",
  "segment_id": 1,
  "t_start": 0.0,
  "t_end": 30.0,
  "utterances": [
    {"speaker": "A", "text": "The quarterly results are looking very positive"},
    {"speaker": "B", "text": "Revenue grew by twenty percent this quarter"}
  ],
  "meeting_context": {
    "total_segments": 3,
    "segment_index_in_meeting": 1
  }
}
```

**Output:**
```json
{
  "meeting_id": "meeting_123",
  "segment_id": 1,
  "t_start": 0.0,
  "t_end": 30.0,
  "topic_label": "Q2 Sales Review",
  "summary_bullets": [
    "Revenue up 20% vs previous quarter",
    "Customer acquisition costs declining",
    "Cloud infrastructure driving cost increases"
  ],
  "status": "complete"
}
```

> `status: "draft"` is returned if Mistral is not loaded (`LLM_MODEL_PATH` missing or file absent). All other response fields will be empty strings/lists.

#### 3. `HealthDeployment` — `/health`

```json
{
  "status": "ok",
  "mode": "ray_serve",
  "device": "cuda"
}
```

#### 4. `RecapPipelineDeployment` — `/recap`
Full pipeline orchestrator. Takes raw utterances, runs both stages, persists to SQLite, returns consolidated recap.

**Input validation rules:**
- 0 utterances → 400 error
- All utterances under 20 chars after cleaning → 400 error
- Fewer than 2 valid utterances → 400 error
- 2–6 utterances → allowed with `short_meeting_low_confidence` warning
- 7+ utterances → fully valid
- >2000 utterances → truncated to 2000 with warning

**Input:**
```json
{
  "meeting_id": "meeting_123",
  "utterances": [
    {"speaker": "A", "text": "Good morning, let's start the quarterly review.", "t_start": 0.0, "t_end": 5.0},
    {"speaker": "B", "text": "Sure, I have the latest numbers ready to share.", "t_start": 5.0, "t_end": 10.0}
  ]
}
```

**Output:**
```json
{
  "meeting_id": "meeting_123",
  "generated_at": "2026-04-20T10:00:00Z",
  "total_segments": 3,
  "processing_time_seconds": 1.4,
  "warnings": [],
  "recap": [
    {
      "meeting_id": "meeting_123",
      "segment_id": 1,
      "t_start": 0.0,
      "t_end": 25.0,
      "topic_label": "Q2 Sales Review",
      "summary_bullets": ["Revenue up 20%", "Costs declining", "Cloud driving spend"],
      "status": "complete"
    }
  ]
}
```

---

### API Endpoints

| Method | Path | Handler | Description |
|---|---|---|---|
| `GET` | `/health` | HealthDeployment | System health + device info |
| `POST` | `/segment` | SegmenterDeployment | Raw boundary detection (single window) |
| `POST` | `/summarize` | SummarizerDeployment | Raw segment summarization |
| `POST` | `/recap` | RecapPipelineDeployment | Full pipeline: utterances → segmented recap + DB write |
| `GET` | `/metrics` | MetricsDeployment | Prometheus metrics |
| `GET` | `/ui` | RecapUIDeployment | Browser recap viewer (reads from DB) |
| `GET` | `/api/meetings` | RecapDeployment | List all processed meetings from DB |
| `GET` | `/api/recap/{meeting_id}` | RecapDeployment | Get recap for a meeting from DB |
| `POST` | `/api/feedback` | RecapDeployment | Submit boundary correction (written to DB) |

---

### Request Flow

```
POST /recap
  │
  ├─ _validate(utterances)
  │    ├─ empty → 400
  │    ├─ all < 20 chars → 400
  │    └─ < 2 valid → 400
  │
  ├─ _build_windows(utterances, window_size=7)
  │    └─ Sliding 7-utterance windows centred on each transition point
  │       "[SPEAKER_X]: text ..." formatted string per window
  │
  ├─ [parallel] segmenter.predict_single.remote(window) × N-1 windows
  │    └─ @serve.batch groups up to 8 simultaneous calls → 1 GPU forward pass
  │       Returns: { is_boundary, boundary_probability, t_boundary }
  │
  ├─ _assemble_segments(utterances, decisions)
  │    └─ Splits utterance list at every is_boundary=true position
  │       Produces M segments with t_start, t_end, utterances[]
  │
  ├─ [sequential] summarizer.__call__.remote(segment) × M segments
  │    └─ One Mistral call per segment (JSON extraction, up to 10 retries)
  │       Returns: { topic_label, summary_bullets[], status }
  │
  ├─ store.save_recap(meeting_id, recap)    → SQLite recaps table
  ├─ store.save_utterances(meeting_id, ..)  → SQLite utterances table
  │
  └─ JSONResponse: { meeting_id, generated_at, total_segments,
                     processing_time_seconds, warnings[], recap[] }
```

---

### MLflow Integration

**Registry:** `http://192.5.86.182:5000`
**Model name:** `jitsi-topic-segmenter`
**Aliases:** `@production` (V1, threshold 0.40), `@fallback` (V2, threshold 0.35)

**Environment variables:**
```bash
MLFLOW_TRACKING_URI=http://192.5.86.182:5000
MLFLOW_MODEL_NAME=jitsi-topic-segmenter
MODEL_ALIAS=production
MODEL_RELOAD_INTERVAL_SECONDS=300
```

**Fallback chain on startup:**
```
MLflow @production → MLflow @fallback → /models/roberta-seg (local) → roberta-base (HuggingFace)
```

**Hot-reload:** Background thread wakes every 300s, checks MLflow version, swaps model under `_model_lock` without actor restart.

---

### Metrics & Monitoring

Prometheus scrapes `ray-serve:8000/metrics` every 5 seconds.

**Grafana Dashboard panels:**

| Section | Panel | Metric |
|---|---|---|
| System Overview | Total Requests | `jitsi_requests_total` |
| System Overview | Success Rate | success/total ratio |
| System Overview | GPU Memory | `jitsi_gpu_memory_used_mb` |
| System Overview | SLA Violations | `jitsi_sla_violations_total` |
| Latency | Segmentation p50/p95/p99 | `jitsi_request_latency_seconds{endpoint="segment"}` |
| Latency | Summarization p50/p95/p99 | `jitsi_request_latency_seconds{endpoint="summarize"}` |
| Throughput | req/s by endpoint | `rate(jitsi_requests_total[1m])` |
| Model Metrics | Confidence Distribution | `jitsi_boundary_confidence` |
| Model Metrics | Avg Batch Size | `jitsi_batch_size` |
| Recap Pipeline | Recap Duration | `jitsi_recap_duration_seconds` |
| Recap Pipeline | Segments per Meeting | `jitsi_recap_segments_per_meeting` |
| System Resources | GPU Memory Over Time | `jitsi_gpu_memory_used_mb` |
| System Resources | CPU & RAM | `jitsi_cpu_utilization_percent`, `jitsi_ram_used_mb` |

**Alert rules:**

| Alert | Condition | Severity |
|---|---|---|
| `SegmentationLatencyHigh` | Segmentation p95 > 2s for 2 min | critical + rollback |
| `SummarizationLatencyHigh` | Summarization p95 > 30s for 2 min | warning |
| `RecapLatencyHigh` | Full recap p95 > 5 min for 2 min | critical + rollback |
| `HighErrorRate` | Error rate > 5% for 3 min | critical + rollback |
| `ServiceDown` | `jitsi_requests_total` metric absent for 10 min | critical |
| `GPUMemoryHigh` | GPU memory > 21504 MB (87.5% of 24 GB) for 5 min | warning |
| `CPUHigh` | CPU > 85% for 5 min | warning |
| `ModelNotLoaded` | `jitsi_model_loaded == 0` for 1 min | critical |
| `LowConfidenceScores` | Median boundary confidence < 0.3 for 3 min | warning + investigate |
| `RetrainingThresholdReached` | 500 corrections accumulated for 5 min | warning + retrain |
| `HighBoundaryCorrections` | 10+ boundary removals in 1 hour | warning + investigate |
| `FeedbackDriftSpiking` | >30% of recap segments manually corrected for 5 min | critical + investigate |

---

## Database & Persistence

### Overview

All meeting data is persisted in a **SQLite database** managed by `RecapStore` in `ray_serve/storage.py`. The database file lives inside the container at the path configured by `DB_PATH` (default: `recaps.db` in the working directory `/app`). To survive container restarts, mount it as a volume (see below).

### Schema

```
recaps
┌──────────────┬──────────────┬───────────────────────┐
│ meeting_id   │ data         │ created_at             │
│ TEXT (PK)    │ TEXT (JSON)  │ TEXT (ISO timestamp)   │
└──────────────┴──────────────┴───────────────────────┘

utterances
┌──────────────┬─────────┬──────────────┬─────────┬───────┐
│ meeting_id   │ speaker │ text         │ t_start │ t_end │
│ TEXT         │ TEXT    │ TEXT         │ REAL    │ REAL  │
└──────────────┴─────────┴──────────────┴─────────┴───────┘
```

### RecapStore operations

| Method | Description |
|---|---|
| `save_recap(meeting_id, recap_json)` | Upserts full recap blob. Duplicate `meeting_id` overwrites. |
| `save_utterances(meeting_id, utterances)` | Inserts all utterances for a meeting. |
| `get_recap(meeting_id)` | Returns the recap JSON for a given meeting, or `None`. |
| `list_meetings()` | Returns all `meeting_id` values with their `created_at` timestamps. |

### Who reads and writes the DB

```
RecapPipelineDeployment (/recap)
    └── WRITES → save_recap() + save_utterances()   on every successful recap

RecapDeployment (/api/*)
    └── READS  → get_recap(), list_meetings()
    └── WRITES → feedback corrections via /api/feedback

RecapUIDeployment (/ui)
    └── serves recap_ui.html; browser JS fetches /api/recap/{id} to display data
```

### Persisting the database across container restarts

By default `recaps.db` lives inside the container and is lost on restart. To persist it, add a bind-mount in `docker-compose.yml`:

```yaml
ray-serve:
  volumes:
    - ./models:/models
    - ./recaps.db:/app/recaps.db    # add this line
```

Or point to a custom path via environment variable:

```bash
DB_PATH=/data/recaps.db
```

### Connecting to the database directly

```bash
# Open an interactive SQLite shell inside the running container
docker exec -it jitsi-ray-serve sqlite3 recaps.db

# Useful queries
.tables
SELECT meeting_id, created_at FROM recaps ORDER BY created_at DESC LIMIT 10;
SELECT COUNT(*) FROM utterances WHERE meeting_id = 'meeting_123';
SELECT data FROM recaps WHERE meeting_id = 'meeting_123';
```

---

## What We Tried & Why We Use Ray Serve

### Approach 1 — FastAPI (baseline)

We started with a plain **FastAPI + PyTorch** server (`app/`). It handles `/segment` and `/summarize` as independent stateless HTTP endpoints. We also exported the RoBERTa model to **ONNX** and benchmarked all four variants: PyTorch CPU, PyTorch GPU, ONNX CPU, ONNX GPU. FastAPI is simple and easy to reason about, but every request runs serially — concurrent Jitsi meetings would queue up behind each other with no way to share a single GPU pass across requests.

### Approach 2 — Triton Inference Server

We configured **Triton** (`triton_models/roberta_segmenter/config.pbtxt`) with ONNX backend, dynamic batching (preferred sizes 64/128/256, 5ms queue delay), and a max batch of 256. Triton solved the batching problem well for RoBERTa, but it only handles the segmenter — Mistral-7B cannot run inside Triton (no GGUF/llama.cpp support). That meant we still needed a separate FastAPI process for summarization, leaving us with two independent services and no clean way to chain Stage A → Stage B in a single request without a wrapper orchestrator.

### Why We Chose Ray Serve

Ray Serve solves all of this in one framework:

| Requirement | FastAPI | Triton | Ray Serve |
|---|---|---|---|
| GPU dynamic batching for RoBERTa | ❌ | ✅ | ✅ (`@serve.batch`) |
| Mistral-7B summarization | ✅ | ❌ | ✅ |
| Stage A → B pipeline in one request | ❌ | ❌ | ✅ (deployment handles) |
| Fractional GPU sharing (0.3 + 0.7) | ❌ | ❌ | ✅ |
| MLflow model registry + hot-reload | ❌ | ❌ | ✅ |
| Built-in Prometheus metrics | ❌ | Partial | ✅ |
| SQLite persistence | ❌ | ❌ | ✅ |
| Single container, single port | ❌ | ❌ | ✅ |

---

## FastAPI Baseline

Located in `app/`. Stateless single-process serving kept for benchmarking comparison against Ray Serve.

**Serving modes** (via `SERVING_MODE` env var):

| Mode | Backend | Use Case |
|---|---|---|
| `pytorch` | Native PyTorch | GPU reference |
| `onnx_cpu` | ONNX Runtime CPU | CPU-only deployment |
| `onnx_gpu` | ONNX Runtime CUDA | GPU with ONNX optimization |

Key difference from Ray Serve: no batching, no deployment-to-deployment calls, no MLflow integration, no persistence.

---

## Docker Compose Services

```bash
# Start the full Ray Serve stack
cd serving
docker compose up -d ray-serve prometheus grafana alertmanager node-exporter

# FastAPI baselines for comparison
docker compose up -d api_pytorch_gpu          # Port 8000
docker compose up -d api_onnx_cpu             # Port 8001
docker compose up -d api_pytorch_cpu          # Port 8002
docker compose up -d api_pytorch_multiworker  # Port 8004 (4 uvicorn workers)

# With Triton (optional — requires --profile triton)
docker compose --profile triton up -d
```

**Port map:**

| Service | Port | Description |
|---|---|---|
| `ray-serve` | 8000 | Ray Serve (primary) |
| `ray-dashboard` | 8265 | Ray cluster dashboard |
| `prometheus` | 9090 | Metrics collector |
| `grafana` | 3000 | Dashboard UI |
| `alertmanager` | 9093 | Alert routing |
| `node-exporter` | 9100 | Host system metrics |
| `api_pytorch_gpu` | 8000 | FastAPI PyTorch GPU |
| `api_pytorch_cpu` | 8002 | FastAPI PyTorch CPU (CUDA disabled) |
| `api_onnx_cpu` | 8001 | FastAPI ONNX CPU |
| `api_onnx_gpu` | 8003 | FastAPI ONNX GPU |
| `api_pytorch_multiworker` | 8004 | FastAPI 4-worker GPU |
| `triton` | 8100–8102 | Triton Inference Server |

**Service notes:**
- `api_pytorch_cpu` — forces `CUDA_VISIBLE_DEVICES=""` to disable GPU
- `api_pytorch_multiworker` — runs `uvicorn --workers 4` for parallel GPU utilisation testing
- `worker` — async background recap processor (`recap_worker.py`), depends on `api_pytorch`
- `triton` — only starts with `--profile triton`; uses `nvcr.io/nvidia/tritonserver:23.10-py3`

---

## Container Build

The Ray Serve container is built from `ray_serve/Dockerfile.ray`:

```
Base image:  pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
```

**Build steps:**
1. System packages: `curl`, `git`, `build-essential`, `cmake` (required to compile llama-cpp with CUDA)
2. `pip install -r requirements_ray.txt`
3. `llama-cpp-python==0.2.56` compiled with `CMAKE_ARGS="-DLLAMA_CUBLAS=on"` (enables full GPU offload)
4. Copies `ray_serve/` into `/app/ray_serve/`
5. Exposes port `8000`
6. Entrypoint: `python3 ray_serve/serve.py`

> `llama-cpp-python` compiles from source during build (~5 min). The `cmake` + `build-essential` packages are required for this step — do not remove them from the Dockerfile.

**Rebuild after code changes:**
```bash
cd ML-Sys-Ops-Project
git pull
cd serving
docker compose build ray-serve
docker compose up -d --force-recreate ray-serve
```

---

## Setup & Deployment

### First-time setup on Chameleon Cloud

```bash
git clone <repo-url>
cd ML-Sys-Ops-Project/serving
bash setup.sh
```

`setup.sh` installs Docker, NVIDIA Container Toolkit, downloads model weights, and starts all containers.

**Chameleon-specific prerequisites (bare-metal only):**
```bash
# python3-pip is not pre-installed on Chameleon bare-metal images
sudo apt install python3-pip -y

# setup.sh step [5] (model download) uses pip — must be on PATH first
# After install, re-run setup.sh or run the download step manually:
pip3 install transformers torch --quiet
python3 -c "from transformers import RobertaTokenizer; RobertaTokenizer.from_pretrained('roberta-base')"
```

### Verify deployment

```bash
curl http://192.5.87.115:8000/health
curl http://192.5.87.115:9090/api/v1/targets   # Prometheus targets
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://192.5.86.182:5000` | MLflow server URL |
| `MLFLOW_MODEL_NAME` | `jitsi-topic-segmenter` | Registered model name |
| `MODEL_ALIAS` | `production` | MLflow alias to load (`production` or `fallback`) |
| `MODEL_RELOAD_INTERVAL_SECONDS` | `300` | Hot-reload poll interval (seconds) |
| `MODEL_PATH` | `roberta-base` | Local path to fine-tuned RoBERTa checkpoint |
| `BOUNDARY_THRESHOLD` | `0.5` | RoBERTa boundary decision threshold |
| `LLM_MODEL_PATH` | `/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf` | Mistral GGUF path |
| `MAX_SEGMENT_UTTERANCES` | `200` | Max utterances per segment sent to LLM |
| `DB_PATH` | `recaps.db` | SQLite database file path |
| `SERVING_MODE` | `pytorch` | FastAPI only: `pytorch` / `onnx_cpu` / `onnx_gpu` |
| `FLOATING_IP` | `` | Public IP of Chameleon serving node |

---

## Benchmarking

```bash
# Ray Serve load test (100 requests, 10 concurrent)
docker exec jitsi-ray-serve python3 benchmark_ray.py \
  --url http://localhost:8000/segment \
  --n 100 --concurrency 10

# FastAPI baseline
python3 benchmark/benchmark.py \
  --url http://localhost:8000/segment \
  --label pytorch_gpu --n 200
```

**Observed performance — Quadro RTX 6000 (measured on Chameleon node `192.5.87.115`):**

| Metric | Concurrency 1 | Concurrency 5 |
|---|---|---|
| Segmentation p50 | 96.4 ms | 157.8 ms |
| Segmentation p95 | 100.3 ms | 173.2 ms |
| Throughput | 10.34 req/s | 31.40 req/s |
| Full `/recap` (12 utterances) | ~1.2 s | — |
| GPU memory (total, nvidia-smi) | ~6.1 GB | — |
| SLA (2s for segmentation) | Never breached | Never breached |

> **Note:** `jitsi_gpu_memory_used_mb` currently reads 0 in Grafana. This is a known issue — `torch.cuda.memory_allocated()` returns 0 in MetricsDeployment because it runs in a separate Ray worker process. Actual GPU usage is visible via `nvidia-smi` (~6.1 GB). Fix requires switching to `pynvml` for system-wide GPU queries (not yet applied).

---

## Edge Cases & Validation

| Input | Behaviour |
|---|---|
| Empty utterances list | `400` — `"Empty transcript — no utterances provided"` |
| All utterances < 20 chars | `400` — `"All utterances under 20 chars after cleaning"` |
| Only 1 valid utterance | `400` — `"meeting too short for inference"` |
| 2–6 utterances | `200` + `warnings: ["short_meeting_low_confidence"]` |
| Single speaker | `200` + `warnings: ["single_speaker"]` |
| Duration < 10s | `200` + `warnings: ["very_short_meeting"]` |
| > 2000 utterances | Truncated to 2000 + warning |
| Unicode / emoji | Cleaned before inference, no crash |
| Duplicate `meeting_id` | Overwrites previous entry in SQLite |
| Missing `text` field | Treated as empty string, filtered out |

---

## Monitoring Stack

```
monitoring/
├── prometheus.yml        # Scrapes: ray-serve:8000, node-exporter:9100
├── alerts.yml            # 12 alert rules across 4 groups
├── alertmanager.yml      # Alert routing config
└── grafana/
    └── provisioning/
        ├── dashboards/
        │   └── jitsi-serving.json    # Dashboard definition (auto-loaded)
        └── datasources/
            └── prometheus.yml        # Datasource: http://prometheus:9090
```


## Troubleshooting

### Container restart loop
```bash
docker logs jitsi-ray-serve 2>&1 | tail -50
```
Common cause: `max_ongoing_requests` in serve.py — Ray 2.9.3 uses `max_concurrent_queries` instead.

### MLflow model not loading
```bash
docker exec jitsi-ray-serve env | grep MLFLOW
# Must show: MLFLOW_TRACKING_URI=http://192.5.86.182:5000
docker exec jitsi-ray-serve curl -s http://192.5.86.182:5000/health
docker logs jitsi-ray-serve 2>&1 | grep -E "MLflow|production|fallback|Ready on"
```

### GPU not used for inference
```bash
nvidia-smi
# Look for: ray::ServeReplica:segmenter and ray::ServeReplica:summarizer with GPU memory
docker logs jitsi-ray-serve 2>&1 | grep "Ready on"
# Must show: [segmenter] Ready on cuda
```

### Grafana showing "No data"
```bash
# Verify Prometheus scrapes port 8000 (not 9091)
curl -s http://192.5.87.115:9090/api/v1/targets | python3 -c "
import json,sys; t=json.load(sys.stdin)
[print(tg['scrapeUrl'], tg['health']) for tg in t['data']['activeTargets']]
"
# Fix if wrong:
sed -i 's/ray-serve:9091/ray-serve:8000/' monitoring/prometheus.yml
docker compose restart prometheus
```

### Database issues
```bash
# Inspect the SQLite database directly
docker exec -it jitsi-ray-serve sqlite3 recaps.db ".tables"
docker exec -it jitsi-ray-serve sqlite3 recaps.db \
  "SELECT meeting_id, created_at FROM recaps ORDER BY created_at DESC LIMIT 5;"

# Database lost after restart? Add a volume mount in docker-compose.yml:
#   volumes:
#     - ./models:/models
#     - ./recaps.db:/app/recaps.db

# Check which DB path the container is using
docker exec jitsi-ray-serve env | grep DB_PATH
```

### UI showing "Failed to fetch" or wrong IP
```bash
grep "API_BASE" ray_serve/recap_ui.html
# Must show: const API_BASE = '';  (relative URL, no hardcoded IP)
```

### LLM returning draft status
```bash
# Check model file exists
docker exec jitsi-ray-serve ls -lh /models/*.gguf
# If missing, re-download:
bash setup.sh
# Verify path matches env var
docker exec jitsi-ray-serve env | grep LLM_MODEL_PATH
```

### Models directory empty after setup
```bash
ls -lh models/
bash setup.sh   # Re-run bootstrap to re-download models
```