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
  - database-only assets for the final integration
  - owns the Postgres bootstrap compose file, `.env.example`, and `init_sql/`
- `proj07-services/`
  - canonical final integration for ingest and workflow services
  - owns transcript ingest, workflow dispatch, Stage 1 / Stage 2 workers, user-summary materialization, and the full integrated compose file
- `initial_implementation/`
  - preserved April 6 standalone runtime bundle
  - contains the original `external_data_training_runtime/`, `endpoint_replay_runtime/`, `online_inference_workflow_runtime/`, `retraining_dataset_runtime/`, and `mock_jitsi_meet/` folders
- `requirements.txt`
  - shared Python dependencies for this bundle
- `setup.sh`
  - creates the Python environment, installs dependencies, copies env templates, prepares block-storage folders, uploads the raw AMI corpus to object storage if needed, and seeds the mock Jitsi transcripts

## Lineage map

- `initial_implementation/online_inference_workflow_runtime/ingest_jitsi_transcript.py`
  - evolved into `proj07-services/proj07_services/api/jitsi_transcript_receiver.py` plus `proj07-services/proj07_services/pipeline/ingest_saved_jitsi_transcript.py`
- `initial_implementation/online_inference_workflow_runtime/build_online_inference_payloads.py`
  - evolved into `proj07-services/proj07_services/pipeline/build_online_inference_payloads.py`
- `initial_implementation/endpoint_replay_runtime/replay_to_hypothetical_endpoints.py`
  - was split and reproduced as `proj07-services/proj07_services/workers/stage1_forward_service.py` and `proj07-services/proj07_services/workers/stage2_forward_service.py`
- `initial_implementation/online_inference_workflow_runtime/materialize_corrected_recap.py`
  - was reproduced as `proj07-services/proj07_services/workers/user_summary_materialize_service.py`
- `initial_implementation/*/feedback_common.py`
  - was directly reused as `proj07-services/proj07_services/common/feedback_common.py`
- `proj07-services/proj07_services/workers/db_task_worker.py`, `proj07-services/proj07_services/common/task_service_common.py`, `proj07-services/proj07_services/common/workflow_task_common.py`, `proj07-services/proj07_services/workers/stage1_payload_service.py`, and `proj07-services/proj07_services/workers/stage2_input_service.py`
  - are integration-layer additions introduced during the final service migration
- `proj07-db/init_sql/`
  - keeps the final schema and migration-style bootstrap files that both the DB-only and full-service compose files mount into Postgres

## Recommended flow

1. From inside `data/`, run:

```bash
bash setup.sh
```

2. Start the integrated service stack:

```bash
cd proj07-services
docker compose up -d
cd ..
```

3. Activate the environment:

```bash
source .venv/bin/activate
```

4. Use `proj07-services/README.md` as the primary runtime guide.

5. Use `proj07-db/README.md` when you only need the database bootstrap and schema details.

6. Use `initial_implementation/README.md` only when you need the original standalone scripts for lineage tracing, comparison, or isolated batch reruns.

## Important note about first-time DB initialization

The SQL files under `proj07-db/init_sql/` are applied automatically by Postgres only when the database volume is initialized for the first time, including when Postgres is started through `proj07-services/docker-compose.yml`.

So this bundle works most cleanly when:

- `/mnt/block/postgres-data` is empty on first startup

If you reuse an old Postgres data directory from a previous schema version, the init SQL will not re-run automatically.
