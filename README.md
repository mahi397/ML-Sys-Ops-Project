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
# Edit .env: set FLOATING_IP, POSTGRES_PASSWORD,
# and for full mode also MINIO_PASSWORD / GRAFANA_PASSWORD.
# If MLflow is on another host, set MLFLOW_TRACKING_URI too.
nano .env
```

For a VM that only runs data + Jitsi and forwards inference to another serving VM, set:

```bash
SETUP_MODE=data-jitsi
STAGE1_FORWARD_URL=http://<SERVING_FLOATING_IP>:8000/segment
STAGE2_FORWARD_URL=http://<SERVING_FLOATING_IP>:8000/summarize
```

You do not need to hand-build `DATABASE_URL`, `MEETING_PORTAL_DATABASE_URL`, or
`JITSI_TRANSCRIPT_INGEST_URL` in the root `.env`. The setup/installer scripts
derive those from `POSTGRES_*`, `INGEST_PORT`, and the Jitsi host-access target.
For same-host Jitsi + data deployments, that target defaults to
`host.docker.internal`. For split deployments, override
`JITSI_TRANSCRIPT_INGEST_URL` and `MEETING_PORTAL_DATABASE_URL`.

### 2. Bootstrap

```bash
chmod +x setup.sh
bash setup.sh
# For Jitsi as well:
DEPLOY_JITSI=true bash setup.sh
# For data + Jitsi only:
SETUP_MODE=data-jitsi DEPLOY_JITSI=true bash setup.sh
```

`setup.sh` handles: Docker + NVIDIA toolkit install, block storage layout, rclone validation, Postgres schema init + migrations, and selected service startup from the single root `docker-compose.yml`. Full mode also downloads local RoBERTa + Mistral models and starts serving/training/monitoring. `data-jitsi` mode skips those heavy CUDA/model services. The traffic generator is the only manual profile service.

Manual compose profiles are available when needed:

```bash
docker compose --profile mlflow up -d minio minio-create-buckets mlflow
docker compose --profile mlflow --profile serving up -d serving-api
docker compose --profile mlflow --profile serving --profile training up -d retrain-watcher online-eval
docker compose --profile serving --profile monitoring up -d prometheus grafana alertmanager node-exporter
docker compose --profile emulated-traffic up -d traffic-generator
```

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

The root [`.env.example`](.env.example) is intentionally short. Most worker
settings, Jitsi auth defaults, meeting-portal DB wiring, and generated secrets
are now derived by `setup.sh` and `install-jitsi-vm.sh`. The main values you
typically edit are:

| Variable | Description |
|----------|-------------|
| `FLOATING_IP` | Public IP of the data/Jitsi VM, or the single full-stack VM |
| `POSTGRES_PASSWORD` | Shared Postgres password for the runtime DB and meeting portal |
| `OBJECT_BUCKET` | Main object-storage bucket used by the data pipeline |
| `MLFLOW_TRACKING_URI` | Optional MLflow server URI when it lives on a different host than this VM |
| `MINIO_PASSWORD` | MinIO root password for full mode |
| `AWS_ACCESS_KEY_ID` | Optional; auto-filled from `rclone_s3` for MLflow/boto3 when blank |
| `AWS_SECRET_ACCESS_KEY` | Optional; auto-filled from `rclone_s3` for MLflow/boto3 when blank |
| `BUCKET_NAME` | MLflow artifact bucket; this can differ from `OBJECT_BUCKET` |
| `STAGE1_FORWARD_URL` | Stage 1 serving endpoint; set this to the serving VM in `data-jitsi` mode |
| `STAGE2_FORWARD_URL` | Stage 2 serving endpoint; set this to the serving VM in `data-jitsi` mode |
| `RETRAIN_THRESHOLD` | Feedback events to trigger retraining (default: `5` for demo) |
| `GRAFANA_PASSWORD` | Grafana admin password in full mode |

## Manual Operations

### Root Compose Services

Run these from the repository root:

| Service | Manual run | Logs | Stop |
|---------|------------|------|------|
| `postgres` | `docker compose up -d postgres` | `docker compose logs -f --since 15m postgres` | `docker compose stop postgres` |
| `adminer` | `docker compose up -d adminer` | `docker compose logs -f --since 15m adminer` | `docker compose stop adminer` |
| `jitsi_transcript_receiver` | `docker compose up -d jitsi_transcript_receiver` | `docker compose logs -f --since 15m jitsi_transcript_receiver` | `docker compose stop jitsi_transcript_receiver` |
| `db_task_worker` | `docker compose up -d db_task_worker` | `docker compose logs -f --since 15m db_task_worker` | `docker compose stop db_task_worker` |
| `stage1_payload_service` | `docker compose up -d stage1_payload_service` | `docker compose logs -f --since 15m stage1_payload_service` | `docker compose stop stage1_payload_service` |
| `stage1_forward_service` | `docker compose up -d stage1_forward_service` | `docker compose logs -f --since 15m stage1_forward_service` | `docker compose stop stage1_forward_service` |
| `stage2_input_service` | `docker compose up -d stage2_input_service` | `docker compose logs -f --since 15m stage2_input_service` | `docker compose stop stage2_input_service` |
| `stage2_forward_service` | `docker compose up -d stage2_forward_service` | `docker compose logs -f --since 15m stage2_forward_service` | `docker compose stop stage2_forward_service` |
| `user_summary_materialize_service` | `docker compose up -d user_summary_materialize_service` | `docker compose logs -f --since 15m user_summary_materialize_service` | `docker compose stop user_summary_materialize_service` |
| `retraining_dataset_service` | `docker compose up -d retraining_dataset_service` | `docker compose logs -f --since 15m retraining_dataset_service` | `docker compose stop retraining_dataset_service` |
| `production_drift_monitor` | `docker compose up -d production_drift_monitor` | `docker compose logs -f --since 15m production_drift_monitor` | `docker compose stop production_drift_monitor` |
| `minio` | `docker compose --profile mlflow up -d minio` | `docker compose --profile mlflow logs -f --since 15m minio` | `docker compose --profile mlflow stop minio` |
| `minio-create-buckets` | `docker compose --profile mlflow up minio-create-buckets` | `docker compose --profile mlflow logs --since 15m minio-create-buckets` | `docker compose --profile mlflow stop minio-create-buckets` |
| `mlflow` | `docker compose --profile mlflow up -d mlflow` | `docker compose --profile mlflow logs -f --since 15m mlflow` | `docker compose --profile mlflow stop mlflow` |
| `serving-api` | `docker compose --profile mlflow --profile serving up -d serving-api` | `docker compose --profile mlflow --profile serving logs -f --since 15m serving-api` | `docker compose --profile mlflow --profile serving stop serving-api` |
| `retrain-watcher` | `docker compose --profile mlflow --profile serving --profile training up -d retrain-watcher` | `docker compose --profile mlflow --profile serving --profile training logs -f --since 15m retrain-watcher` | `docker compose --profile mlflow --profile serving --profile training stop retrain-watcher` |
| `online-eval` | `docker compose --profile mlflow --profile training up -d online-eval` | `docker compose --profile mlflow --profile training logs -f --since 15m online-eval` | `docker compose --profile mlflow --profile training stop online-eval` |
| `traffic-generator` | `docker compose --profile emulated-traffic up -d traffic-generator` | `docker compose --profile emulated-traffic logs -f --since 15m traffic-generator` | `docker compose --profile emulated-traffic stop traffic-generator` |
| `prometheus` | `docker compose --profile serving --profile monitoring up -d prometheus` | `docker compose --profile serving --profile monitoring logs -f --since 15m prometheus` | `docker compose --profile serving --profile monitoring stop prometheus` |
| `grafana` | `docker compose --profile serving --profile monitoring up -d grafana` | `docker compose --profile serving --profile monitoring logs -f --since 15m grafana` | `docker compose --profile serving --profile monitoring stop grafana` |
| `alertmanager` | `docker compose --profile monitoring up -d alertmanager` | `docker compose --profile monitoring logs -f --since 15m alertmanager` | `docker compose --profile monitoring stop alertmanager` |
| `node-exporter` | `docker compose --profile monitoring up -d node-exporter` | `docker compose --profile monitoring logs -f --since 15m node-exporter` | `docker compose --profile monitoring stop node-exporter` |

### Jitsi Compose Services

After Jitsi is installed, run these from `/mnt/block/jitsi/jitsi-docker-jitsi-meet`:

| Service | Manual run | Logs | Stop |
|---------|------------|------|------|
| `web` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml up -d web` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml logs -f --since 15m web` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml stop web` |
| `prosody` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml up -d prosody` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml logs -f --since 15m prosody` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml stop prosody` |
| `jicofo` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml up -d jicofo` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml logs -f --since 15m jicofo` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml stop jicofo` |
| `jvb` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml up -d jvb` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml logs -f --since 15m jvb` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml stop jvb` |
| `jigasi` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml up -d jigasi` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml logs -f --since 15m jigasi` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml stop jigasi` |
| `transcriber` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml up -d transcriber` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml logs -f --since 15m transcriber` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml stop transcriber` |
| `meeting-portal-app` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml up -d meeting-portal-app` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml logs -f --since 15m meeting-portal-app` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml stop meeting-portal-app` |
| `transcript-uploader` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml up -d transcript-uploader` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml logs -f --since 15m transcript-uploader` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml stop transcript-uploader` |
| `vosk` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml up -d vosk` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml logs -f --since 15m vosk` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml stop vosk` |

The same Jitsi commands are also documented in [jitsi-deployment/README.md](jitsi-deployment/README.md).

### Trigger a retrain manually

```bash
docker compose exec retrain-watcher python /app/retrain.py
```

### Force a dataset rebuild

```bash
docker compose exec retraining_dataset_service \
  python -m proj07_services.workers.retraining_dataset_service --once --force-run
```

### Manual worker runs with CLI args

Check retraining thresholds and candidate meetings without building artifacts:

```bash
docker compose exec retraining_dataset_service \
  python -m proj07_services.workers.retraining_dataset_service --once --dry-run
```

Force a one-shot retraining dataset cycle even if thresholds are not met:

```bash
docker compose exec retraining_dataset_service \
  python -m proj07_services.workers.retraining_dataset_service --once --force-run
```

Run a single production drift monitoring cycle on demand:

```bash
docker compose exec production_drift_monitor \
  python -m proj07_services.workers.production_drift_monitor --once
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
