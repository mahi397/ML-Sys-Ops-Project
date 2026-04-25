# Jitsi Topic Segmentation & Recap — NeuralOps (proj07)

End-to-end ML system that automatically segments Jitsi meeting transcripts by topic and generates per-segment summaries, with automated model retraining driven by user feedback corrections.

**Team:** Aneesh Mokashi (Data) · Mahima Sachdeva (Training) · Shruti Pangare (Serving)   
**Infrastructure:** Chameleon Cloud GPU node (CHI@UC, Quadro RTX 6000)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Jitsi Meet Stack                             │
│  (jitsi-deployment/)  Web · Prosody · JVB · Jigasi · Vosk · Portal  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ transcript upload (HTTP)
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Data Pipeline  (data/)                        │
│  jitsi_transcript_receiver → stage1_payload → stage1_forward        │
│                            → stage2_input   → stage2_forward        │
│  user_summary_materialize · retraining_dataset_service              │
│  production_drift_monitor                                           │
└──────────────┬──────────────────────────────────┬───────────────────┘
               │  inference requests              │  feedback events
               ▼                                  ▼
┌──────────────────────────┐       ┌───────────────────────────────────┐
│   Serving  (serving/)    │       │    Training  (train/)             │
│                          │       │                                   │
│  Ray Serve  :8000        │◄──────│  retrain_watcher  (always-on)     │
│  RoBERTa segmenter (0.3 GPU)     │  retrain.py  (Ray Train, GPU)     │
│  Mistral-7B summarizer (0.7 GPU) │  online_eval  (hourly)            │
│  MLflow hot-reload       │       │  offline_eval  (on demand)        │
│  /recap · /segment       │       └──────────────┬────────────────────┘
│  /summarize · /feedback  │                      │ registers candidate
└──────────────────────────┘                      ▼
                                   ┌───────────────────────────────────┐
                                   │  MLflow Registry  :5000           │
                                   │  jitsi-topic-segmenter            │
                                   │  aliases: production · candidate  │
                                   └───────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────┐
│                     Shared Platform                                  │
│  Postgres :5432 · MinIO :9000/:9001 · Adminer :5050                  │
│  Prometheus :9090 · Grafana :3000 · Alertmanager :9093               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Service Map

| Service | Port | Description |
|---------|------|-------------|
| Ray Serve API | `:8000` | `/health` `/recap` `/segment` `/summarize` `/feedback` |
| Ray Dashboard | `:8265` | Ray cluster dashboard |
| MLflow | `:5000` | Experiment tracking + model registry |
| Grafana | `:3000` | Serving + infrastructure dashboards |
| Prometheus | `:9090` | Metrics collector |
| Alertmanager | `:9093` | Alert routing |
| Postgres | `:5432` | Application + MLflow database |
| MinIO | `:9000` | Ray checkpoint object store |
| MinIO Console | `:9001` | MinIO UI |
| Adminer | `:5050` | Postgres admin UI |
| Jitsi Meet | `:8443` | HTTPS meeting portal |
| Jitsi HTTP | `:8088` | HTTP (redirects to HTTPS) |

---

## Repository Structure

```
ML-Sys-Ops-Project/
├── docker-compose.yml          # Single compose entry point for platform, data, serving, training, monitoring
├── setup.sh                    # One-command bootstrap for a fresh Chameleon GPU node
├── .env.example                # Global environment template — copy to .env
│
├── serving/                    # Ray Serve, RoBERTa + Mistral, Prometheus/Grafana
│   ├── ray_serve/              # serve.py, storage.py, Dockerfile.ray
│   ├── monitoring/             # prometheus.yml, alerts.yml, Grafana dashboards
│   └── models/                 # Downloaded model weights (gitignored)
│
├── train/                      # PyTorch fine-tuning, Ray Train, MLflow
│   ├── retrain.py              # Main training (fault-tolerant) + evaluation + MLflow registration
│   ├── retrain_watcher.py      # Feedback watcher daemon
│   ├── online_eval.py          # Hourly correction-rate monitoring
│   ├── offline_eval.py         # On-demand test-set evaluation
│   └── Dockerfile              # Training container image
│
├── data/                       # ingest, workflow workers, dataset pipeline
│   ├── proj07-runtime/         # Production service package and Docker image context
│   ├── proj07-db/              # Postgres schema and migrations
│   └── initial_implementation/ # Archived independent standalone scripts
│
│
└── jitsi-deployment/           # Jitsi Meet + meeting portal + transcript uploader
    ├── install-jitsi-vm.sh     # Automated Jitsi installer
    └── compose/                # Custom service definitions (portal, uploader, vosk)
```

---

## End-to-End Flow

1. **Meeting** — A Jitsi session is transcribed live by Vosk (speech-to-text). The transcript-uploader POSTs the finished transcript to the ingest API at `:9099`.

2. **Ingest** — `jitsi_transcript_receiver` saves the transcript, builds Stage 1 inference payloads (7-utterance sliding windows), and forwards each window to the serving API at `:8000/segment`.

3. **Serving** — `SegmenterDeployment` (RoBERTa, 0.3 GPU) predicts topic boundaries. Stage 2 payloads are assembled from boundary decisions and forwarded to `:8000/summarize`. `SummarizerDeployment` (Mistral-7B, 0.7 GPU) generates topic labels + bullet summaries per segment.

4. **Recap UI** — Results are persisted to Postgres and displayed in the Jitsi meeting portal at `:8000/ui`.

5. **Feedback** — Users correct boundaries in the recap UI. Each correction writes a `merge_segments` or `split_segment` event to `feedback_events` in Postgres.

6. **Retraining trigger** — `retrain_watcher` polls the table every 5 minutes. When accumulated corrections above the watermark reach the threshold (5 for demo, 500 for prod), it fires the retrain pipeline.

7. **Dataset build** — `retraining_dataset_service` assembles `feedback_examples.jsonl` from corrected meeting windows, runs a drift quality gate, and uploads to chi.tacc object storage. The new version is registered in `dataset_versions`.

8. **Training** — `retrain.py` warm-starts from the current `production` model, trains with Ray Train (fault-tolerant, MinIO checkpoints), then runs evaluation: aggregate metrics (Pk, F1, WindowDiff), fairness slice evaluation, and failure mode tests. Metrics and a model card are logged to MLflow.

9. **Registration** — If all quality gates pass, the model is registered in MLflow with the `candidate` alias. A team member promotes it to `production` via the MLflow UI. The serving layer hot-reloads within 5 minutes.

10. **Monitoring** — Prometheus scrapes Ray Serve metrics every 5s. `online_eval.py` logs correction rate, FPR, FNR to MLflow hourly. Grafana dashboards and Alertmanager alert rules watch for latency spikes, high error rates, and model degradation.

---

## Quick Start

### Prerequisites

- Chameleon Cloud GPU node with a floating IP and attached block volume at `/mnt/block`
- `~/.config/rclone/rclone.conf` with an `rclone_s3` remote (CHI@TACC S3)
- `setup.sh` reads object-storage credentials from that rclone remote and fills the MLflow/boto3 env vars when they are blank.

### 1. Clone and configure

```bash
git clone https://github.com/mahi397/ML-Sys-Ops-Project.git
cd ML-Sys-Ops-Project
cp .env.example .env
# Edit .env: set FLOATING_IP, POSTGRES_PASSWORD, MINIO_PASSWORD, BUCKET_NAME
nano .env
```

### 2. Bootstrap

```bash
chmod +x setup.sh
bash setup.sh
# For Jitsi as well:
DEPLOY_JITSI=true bash setup.sh
```

`setup.sh` handles: Docker + NVIDIA toolkit install, block storage layout, rclone validation, RoBERTa + Mistral model downloads, Postgres schema init + migrations, MLflow startup + model registry restore, and root `docker compose up -d` for all default services. The traffic generator is the only manual profile service.

### 3. Verify

```bash
curl http://<FLOATING_IP>:8000/health
# → {"status": "ok", "mode": "ray_serve", "device": "cuda"}

docker compose ps
# All services should show "running" or "healthy"
```

### 4. Access services

| URL | What |
|-----|------|
| `http://<FLOATING_IP>:5000` | MLflow |
| `http://<FLOATING_IP>:3000` | Grafana (admin / see .env) |
| `http://<FLOATING_IP>:9001` | MinIO console |
| `http://<FLOATING_IP>:5050` | Adminer (Postgres) |
| `https://<FLOATING_IP>:8443` | Jitsi Meet |

---

## Environment Variables

See [`.env.example`](.env.example) for the full list. Key variables:

| Variable | Description |
|----------|-------------|
| `FLOATING_IP` | Public IP of the Chameleon node |
| `POSTGRES_PASSWORD` | Postgres password |
| `MINIO_PASSWORD` | MinIO root password |
| `AWS_ACCESS_KEY_ID` | Optional; auto-filled from `rclone_s3` for MLflow/boto3 when blank |
| `AWS_SECRET_ACCESS_KEY` | Optional; auto-filled from `rclone_s3` for MLflow/boto3 when blank |
| `BUCKET_NAME` | MLflow artifact bucket (default: `proj07-mlflow-artifacts`) |
| `RETRAIN_THRESHOLD` | Feedback events to trigger retraining (default: `5` for demo) |
| `GRAFANA_PASSWORD` |admin  |

## Manual Operations

### Trigger a retrain manually

```bash
docker compose exec retrain-watcher python /app/retrain.py
```

### Force a dataset rebuild

```bash
docker compose exec retraining_dataset_service \
  python -m proj07_services.workers.retraining_dataset_service --once --force-run
```

### Promote a candidate model to production

1. Open MLflow at `http://<FLOATING_IP>:5000`
2. Go to **Models → jitsi-topic-segmenter**
3. Find the version with alias `candidate`, review its model card artifact
4. Set alias `production` — serving hot-reloads within 5 minutes

### Run online evaluation

```bash
docker compose exec retrain-watcher python /app/online_eval.py --days 7
```

### Reload Prometheus config

```bash
curl -X POST http://<FLOATING_IP>:9090/-/reload
```

---

## Key Design Decisions

- **Ray Serve** for serving: fractional GPU sharing (RoBERTa 0.3 + Mistral 0.7), dynamic batching, deployment handles for pipeline chaining, MLflow hot-reload — all in one container. See [serving/ray_serve/RAY_SERVE_JUSTIFICATION.md](serving/ray_serve/RAY_SERVE_JUSTIFICATION.md).
- **Ray Train** for retraining: `FailureConfig(max_failures=2)` makes unattended overnight retraining fault-tolerant. Checkpoints go to MinIO so a VM restart mid-epoch resumes from the last saved state.
- **MLflow `--serve-artifacts`**: The MLflow server proxies all artifact uploads/downloads. Training containers authenticate to chi.tacc S3 through the proxy — no S3 credentials needed in client code.
- **Strict meeting-ID splits**: Train/val/test splits are fixed by `meeting_id` and never reassigned. New production feedback meetings only enter val/test; the previous dataset version rolls forward into train.
- **Watermark-based retraining**: The watcher tracks the highest consumed `feedback_event_id`. Corrections are never double-counted across runs even if the watcher restarts.

---

## Safeguarding

| Principle | Implementation |
|-----------|---------------|
| **Fairness** | Slice evaluation on 6 slices (meeting size × speaker count). Fairness gate blocks registration if any slice Pk > 0.40. |
| **Robustness** | Three failure-mode tests on every run: very short meetings, single-topic meetings, speaker relabeling invariance. |
| **Transparency** | Model card JSON artifact logged to MLflow on every run — train data provenance, metrics, thresholds, fairness results, limitations. |
| **Accountability** | `audit_log` + `retrain_log` tables. Promotion from `candidate` → `production` is a manual human step. |
| **Privacy** | Speaker identity abstracted to `[SPEAKER_A/B/C]` tokens before any training data is created. No participant names or IDs in datasets. |
| **Threshold** | `best_threshold` tagged on each registered model version. Serving reads from registry — never hardcoded. |

---

## Sub-system READMEs

- [serving/README.md](serving/README.md) — Ray Serve architecture, API reference, monitoring, benchmarks
- [train/README.md](train/README.md) — Training pipeline, quality gates, evaluation test suite, safeguarding
- [data/README.md](data/README.md) — Data pipeline, ingest flow, dataset versioning, drift control
- [data/proj07-runtime/README.md](data/proj07-runtime/README.md) — Runtime service bundle
- [jitsi-deployment/README.md](jitsi-deployment/README.md) — Jitsi Meet installer and configuration
