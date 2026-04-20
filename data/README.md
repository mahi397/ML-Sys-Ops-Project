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
  - owns the one compose file that launches Postgres, Adminer, transcript ingest, workflow dispatch, Stage 1 / Stage 2 workers, user-summary materialization, and retraining-dataset monitoring
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
  - creates the Python environment, copies the shared `.env` template if needed, prepares block-storage folders, uploads the raw AMI corpus to object storage if needed, bootstraps synthetic Stage 1 inputs if missing, checks for existing runtime containers and Postgres data, applies idempotent DB migrations, and starts the integrated runtime stack only when needed

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

## Important note about first-time DB initialization

The SQL files under `proj07-db/init_sql/` are mounted by `proj07-runtime/docker-compose.yml`. On a fresh Postgres data volume they are applied automatically during container initialization, and `data/setup.sh` also reapplies the idempotent post-bootstrap migrations so existing volumes can pick up newer schema changes.

When you rerun `data/setup.sh`, it now detects whether the runtime services are already running and whether `${POSTGRES_DATA_DIR:-/mnt/block/postgres-data}` already contains a Postgres data directory, so repeated setup runs are more predictable.

`data/.env` is the canonical shared configuration file. `setup.sh` keeps `proj07-runtime/.env` pointed at that shared file so that updating `data/.env` and then running `docker compose` from `proj07-runtime/` uses the same values.

So this bundle works most cleanly when:

- `/mnt/block/postgres-data` is empty on first startup

If you reuse an old Postgres data directory from a previous schema version, the init SQL will not re-run automatically.
