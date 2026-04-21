# Data Runtime Bundle

This `data/` folder now has an explicit split between the original standalone runtime bundle and the final integrated stack that was built on top of it after April 7, 2026.

## Assumptions

This bundle assumes:

- you are running on a Chameleon VM
- `/mnt/block` is attached and writable
- `rclone` is already configured on the VM
- Docker and the Docker Compose plugin are installed
- the setup step can download the raw AMI dataset and upload it into Chameleon object storage
- you want Postgres data to live on block storage under `/mnt/block/postgres-data`

## Top-level layout

- `proj07-db/`
  - source-only schema and migration assets for the final integration
  - owns `init_sql/` and the canonical database-side SQL history
- `proj07-runtime/`
  - canonical final runnable stack
  - owns the one compose file that launches Postgres, Adminer, transcript ingest, workflow dispatch, Stage 1 / Stage 2 workers, user-summary materialization, retraining-dataset monitoring, and production drift monitoring
- `initial_implementation/`
  - preserved April 6 standalone runtime bundle
  - contains the original `external_data_training_runtime/`, `endpoint_replay_runtime/`, `online_inference_workflow_runtime/`, `retraining_dataset_runtime/`, and `mock_jitsi_meet/` folders
  - now also includes its own `setup.sh` for archived-only bootstrap
- `requirements.txt`
  - shared Python dependencies for this bundle
- `.env.example`
  - canonical shared environment template for the whole `data/` bundle
  - copy it to `.env` and use that one file as the main place to update DB, object-store, Stage 1, and Stage 2 settings
- `setup.sh`
  - global bootstrap for the modern stack
  - creates the Python environment, copies the shared `.env` template if needed, prepares block-storage folders, uploads the raw AMI corpus to object storage if needed, bootstraps synthetic Stage 1 inputs if missing, checks for existing runtime containers and Postgres data, applies idempotent DB migrations, starts the integrated runtime stack only when needed, backfills AMI meetings into Postgres through the modern `proj07-runtime` pipeline when the object-store corpus exists but the DB is missing fully-ingested AMI rows, and replays stored Stage 1 dataset lineage into block storage and Postgres when historical dataset versions still exist

## Lineage map

- `initial_implementation/online_inference_workflow_runtime/ingest_jitsi_transcript.py`
  - evolved into `proj07-runtime/proj07_services/api/jitsi_transcript_receiver.py` plus `proj07-runtime/proj07_services/pipeline/ingest_saved_jitsi_transcript.py`
- `initial_implementation/online_inference_workflow_runtime/build_online_inference_payloads.py`
  - evolved into `proj07-runtime/proj07_services/pipeline/build_online_inference_payloads.py`
- `initial_implementation/endpoint_replay_runtime/replay_to_hypothetical_endpoints.py`
  - was split and reproduced as `proj07-runtime/proj07_services/workers/stage1_forward_service.py` and `proj07-runtime/proj07_services/workers/stage2_forward_service.py`
- `initial_implementation/online_inference_workflow_runtime/materialize_corrected_recap.py`
  - was reproduced as `proj07-runtime/proj07_services/workers/user_summary_materialize_service.py`
- `initial_implementation/*/feedback_common.py`
  - was directly reused as `proj07-runtime/proj07_services/common/feedback_common.py`
- `proj07-runtime/proj07_services/workers/db_task_worker.py`, `proj07-runtime/proj07_services/common/task_service_common.py`, `proj07-runtime/proj07_services/common/workflow_task_common.py`, `proj07-runtime/proj07_services/workers/stage1_payload_service.py`, and `proj07-runtime/proj07_services/workers/stage2_input_service.py`
  - are integration-layer additions introduced during the final service migration
- `proj07-db/init_sql/`
  - keeps the final schema and migration-style bootstrap files that the integrated runtime stack mounts into Postgres

## Data control plane

The modern `data/` bundle now includes an explicit data-quality control layer on top of the ingest and retraining flow.

- At ingestion time, malformed or incomplete Jitsi transcripts are rejected before they are treated as valid inputs.
- During runtime orchestration, `meetings.is_valid` is only flipped on when the expected raw, parsed, Stage 1, Stage 2, and summary artifacts exist.
- During retraining dataset publication, candidate `roberta_stage1_feedback_pool/vN` and `roberta_stage1/vN` builds are profiled, compared against the latest approved reference profile, and quarantined if drift exceeds the configured threshold instead of being published silently.
- During live production monitoring, a scheduled drift monitor compares a rolling window of valid Jitsi meetings against the latest approved Stage 1 dataset profile and persists a report for investigation.

The control-plane state is stored in Postgres:

- `dataset_versions`
  - approved, published dataset manifests
- `dataset_quality_reports`
  - persisted drift and quality-gate results for feedback-pool builds, retraining snapshots, and live production windows
- `retrain_log` and `audit_log`
  - retraining lifecycle history and operational audit trail

The main artifacts produced by this control plane are:

- approved dataset profiles at `proj07-runtime` dataset roots such as `/mnt/block/roberta_stage1/vN/profile.json`
- per-run drift reports under `/mnt/block/staging/feedback_loop/.../quality_report.json`
- quarantined failed candidates under dataset-root `_quarantine/` folders

## Recommended flow

1. From inside `data/`, run:

```bash
cp .env.example .env
bash setup.sh
```

2. The global setup already starts the full stack. To start it manually later:

```bash
cd proj07-runtime
docker compose up -d
cd ..
```

3. Activate the environment:

```bash
source .venv/bin/activate
```

4. Use `proj07-runtime/README.md` as the primary runtime guide.

5. Use `proj07-db/README.md` when you only need the database schema/bootstrap source.

6. Use `initial_implementation/setup.sh` and `initial_implementation/README.md` only when you need the archived standalone scripts for lineage tracing, comparison, or isolated batch reruns.

7. If you tune drift thresholds or monitoring cadence, update `data/.env` and restart the relevant `proj07-runtime` services so the new control-plane settings take effect.

## Runtime service inventory

The modern runtime under `proj07-runtime/` exposes these Docker Compose services:

- `postgres`
  - runtime database
- `adminer`
  - lightweight DB inspection UI
- `jitsi_transcript_receiver`
  - transcript ingest API
- `db_task_worker`
  - workflow-task dispatcher and validity refresher
- `stage1_payload_service`
  - Stage 1 request builder
- `stage1_forward_service`
  - Stage 1 endpoint caller
- `stage2_input_service`
  - Stage 2 input builder from Stage 1 outputs
- `stage2_forward_service`
  - Stage 2 summarization caller
- `user_summary_materialize_service`
  - user-summary artifact materializer
- `retraining_dataset_service`
  - feedback-pool and retraining-snapshot publisher
- `production_drift_monitor`
  - live data-drift monitor

List them directly from Compose:

```bash
cd proj07-runtime
docker compose config --services
```

## Common runtime operations

Start the full runtime stack:

```bash
cd proj07-runtime
docker compose up -d
```

Start only selected services:

```bash
cd proj07-runtime

# generic pattern
docker compose up -d postgres SERVICE

# examples
docker compose up -d postgres jitsi_transcript_receiver
docker compose up -d postgres db_task_worker
docker compose up -d postgres stage1_payload_service
docker compose up -d postgres stage1_forward_service
docker compose up -d postgres stage2_input_service
docker compose up -d postgres stage2_forward_service
docker compose up -d postgres user_summary_materialize_service
docker compose up -d postgres retraining_dataset_service
docker compose up -d postgres production_drift_monitor
docker compose up -d adminer
```

Inspect service status:

```bash
cd proj07-runtime
docker compose ps
```

Check logs for any individual service:

```bash
cd proj07-runtime

# generic pattern
docker compose logs -f SERVICE_NAME

# examples
docker compose logs -f postgres
docker compose logs -f adminer
docker compose logs -f jitsi_transcript_receiver
docker compose logs -f db_task_worker
docker compose logs -f stage1_payload_service
docker compose logs -f stage1_forward_service
docker compose logs -f stage2_input_service
docker compose logs -f stage2_forward_service
docker compose logs -f user_summary_materialize_service
docker compose logs -f retraining_dataset_service
docker compose logs -f production_drift_monitor
```

Show the last 200 lines and continue following:

```bash
cd proj07-runtime
docker compose logs -f --tail=200 SERVICE_NAME
```

Follow several worker services at once:

```bash
cd proj07-runtime
docker compose logs -f \
  db_task_worker \
  stage1_payload_service \
  stage1_forward_service \
  stage2_input_service \
  stage2_forward_service \
  user_summary_materialize_service \
  retraining_dataset_service \
  production_drift_monitor
```

The services also write host-side logs under `/mnt/block/ingest_logs/`:

- `jitsi_transcript_receiver`
  - `/mnt/block/ingest_logs/jitsi_transcript`
- `db_task_worker`
  - `/mnt/block/ingest_logs/db_task_worker`
- `stage1_payload_service`
  - `/mnt/block/ingest_logs/stage1_payload_service`
- `stage1_forward_service`
  - `/mnt/block/ingest_logs/stage1_forward_service`
- `stage2_input_service`
  - `/mnt/block/ingest_logs/stage2_input_service`
- `stage2_forward_service`
  - `/mnt/block/ingest_logs/stage2_forward_service`
- `user_summary_materialize_service`
  - `/mnt/block/ingest_logs/user_summary_materialize_service`
- `retraining_dataset_service`
  - `/mnt/block/ingest_logs/retraining_dataset_service`
- `production_drift_monitor`
  - `/mnt/block/ingest_logs/production_drift_monitor`

Examples:

```bash
tail -f /mnt/block/ingest_logs/retraining_dataset_service/*
tail -f /mnt/block/ingest_logs/production_drift_monitor/*
tail -f /mnt/block/ingest_logs/stage1_payload_service/*
tail -f /mnt/block/ingest_logs/jitsi_transcript/*
```

## Data workflows and one-off jobs

### Generate synthetic Stage 1 bootstrap data

`data/setup.sh` automatically generates synthetic Stage 1 bootstrap artifacts when
`BOOTSTRAP_SYNTHETIC_STAGE1_ENABLED=true` and the expected manifest is missing. To rerun
that flow manually:

```bash
cd data
source .venv/bin/activate
set -a
source .env
set +a

python initial_implementation/endpoint_replay_runtime/generate_synthetic_endpoint_inputs.py \
  --output-root "${BLOCK_ROOT:-/mnt/block}/user-behaviour/inference_requests/stage1" \
  --version "${BOOTSTRAP_SYNTHETIC_STAGE1_VERSION:-1}" \
  --meeting-count "${BOOTSTRAP_SYNTHETIC_STAGE1_MEETING_COUNT:-3}" \
  --seed "${BOOTSTRAP_SYNTHETIC_STAGE1_SEED:-42}" \
  --upload-artifacts \
  --rclone-remote "${RCLONE_REMOTE:-rclone_s3}" \
  --bucket "${OBJECT_BUCKET:-objstore-proj07}" \
  --stage1-object-prefix "${STAGE1_OBJECT_PREFIX:-production/inference_requests/stage1}" \
  --log-file "${BLOCK_ROOT:-/mnt/block}/ingest_logs/synthetic_stage1_bootstrap.log"
```

Watch the synthetic-data log:

```bash
tail -f /mnt/block/ingest_logs/synthetic_stage1_bootstrap.log
```

### Force one retraining dataset cycle

This service builds the next `roberta_stage1_feedback_pool/vN` and `roberta_stage1/vN`
when eligible new meetings and feedback have accumulated. To force one evaluation cycle:

```bash
cd proj07-runtime
docker compose up -d postgres retraining_dataset_service
docker compose exec retraining_dataset_service \
  python -m proj07_services.workers.retraining_dataset_service --once --force-run
```

Inspect outputs:

```bash
ls -1 /mnt/block/staging/feedback_loop/datasets/roberta_stage1_feedback_pool
ls -1 /mnt/block/roberta_stage1
```

### Force one production drift check

```bash
cd proj07-runtime
docker compose up -d postgres production_drift_monitor
docker compose exec production_drift_monitor \
  python -m proj07_services.workers.production_drift_monitor --once
```

Reports are written under:

```text
/mnt/block/staging/feedback_loop/production_drift_reports
```

### Bootstrap AMI meetings into Postgres

If the raw AMI corpus already exists in object storage and Postgres is missing fully ingested
AMI meetings, rerun the bootstrap pipeline directly:

```bash
cd data
source .venv/bin/activate
set -a
source .env
set +a

cd proj07-runtime
python -m proj07_services.pipeline.bootstrap_ami_corpus \
  --rclone-remote "${RCLONE_REMOTE:-rclone_s3}" \
  --bucket "${OBJECT_BUCKET:-objstore-proj07}" \
  --prefix "${AMI_OBJECT_PREFIX:-ami_public_manual_1.6.2}" \
  --raw-root "${BLOCK_ROOT:-/mnt/block}/staging/current_job/raw" \
  --processed-root "${BLOCK_ROOT:-/mnt/block}/staging/current_job/processed" \
  --log-file "${BLOCK_ROOT:-/mnt/block}/ingest_logs/ami_corpus_bootstrap.log"
```

### Restore stored dataset lineage

If historical `roberta_stage1_feedback_pool/vN` or `roberta_stage1/vN` artifacts already exist
in block storage or object storage, rerun the lineage restore flow:

```bash
cd data
source .venv/bin/activate
set -a
source .env
set +a

cd proj07-runtime
python -m proj07_services.retraining.restore_dataset_lineage \
  --rclone-remote "${RCLONE_REMOTE:-rclone_s3}" \
  --bucket "${OBJECT_BUCKET:-objstore-proj07}" \
  --log-file "${BLOCK_ROOT:-/mnt/block}/ingest_logs/retraining_dataset_lineage_restore.log"
```

### Legacy standalone retraining batch

The original batch-style retraining runtime is preserved under
`initial_implementation/retraining_dataset_runtime/`. Use it only when you explicitly need the
archived April 6 workflow instead of the modern long-running service:

```bash
cd data/initial_implementation/retraining_dataset_runtime
bash run_retraining_dataset_batch.sh
```

## Important note about first-time DB initialization

The SQL files under `proj07-db/init_sql/` are mounted by `proj07-runtime/docker-compose.yml`. On a fresh Postgres data volume they are applied automatically during container initialization, and `data/setup.sh` also reapplies the idempotent post-bootstrap migrations so existing volumes can pick up newer schema changes such as `meetings.is_valid` and `dataset_quality_reports`.

When you rerun `data/setup.sh`, it now detects whether the runtime services are already running and whether `${POSTGRES_DATA_DIR:-/mnt/block/postgres-data}` already contains a Postgres data directory, so repeated setup runs are more predictable.

If the AMI raw corpus is already present in object storage but Postgres is empty or only partially populated for `source_type = 'ami'`, `data/setup.sh` now runs the modern `proj07-runtime` AMI bootstrap pipeline automatically and lets that flow fill or repair any missing AMI meetings before setup finishes.

If historical `roberta_stage1_feedback_pool/vN` or `roberta_stage1/vN` artifacts still exist in block storage or object storage, `data/setup.sh` now also restores that lineage before finishing so retraining does not accidentally reuse the wrong base version or restart version numbering from `v1`. When those stored dataset versions reference historical `jitsi_*` meetings that are missing from Postgres, the restore flow now tries to recover those meetings from their stored parsed-transcript payloads and verified Stage 1 / Stage 2 artifacts before replaying the lineage. If you intentionally want a clean-room test run, either clear those stored dataset prefixes first or set `BOOTSTRAP_DATASET_LINEAGE_ENABLED=false` in `data/.env`.

`data/.env` is the canonical shared configuration file. `setup.sh` keeps `proj07-runtime/.env` pointed at that shared file so that updating `data/.env` and then running `docker compose` from `proj07-runtime/` uses the same values.

So this bundle works most cleanly when:

- `/mnt/block/postgres-data` is empty on first startup

If you reuse an old Postgres data directory from a previous schema version, the init SQL will not re-run automatically.
