# Data Runtime Bundle

This `data/` folder is the Chameleon-facing package for the data-team workflows.

## Assumptions

This bundle assumes:

- you are running on a Chameleon VM
- `/mnt/block` is attached and writable
- `rclone` is already configured on the VM
- Docker and the Docker Compose plugin are installed
- the raw AMI dataset is already present in Chameleon object storage
- you want Postgres data to live on block storage under `/mnt/block/postgres-data`

## What is inside

- `proj07-db/`
  - `docker-compose.yml`
  - `init_sql/001_init_postgres_schema.sql`
  - `init_sql/002_feedback_loop_schema.sql`
  - `.env.example`
- `external_data_training_runtime/`
  - AMI ingest + Stage 1 v1/v2 dataset generation
- `endpoint_replay_runtime/`
  - hypothetical Stage 1 / Stage 2 endpoint replay from existing request artifacts
- `online_inference_workflow_runtime/`
  - Jitsi ingest + online feature computation + mock serving + corrected production-like state
- `retraining_dataset_runtime/`
  - rolling feedback-pool and retraining-snapshot compilation
- `mock_jitsi_meet/`
  - Jitsi-style sample transcripts for the online inference workflow
- `requirements.txt`
  - shared Python dependencies for this bundle
- `setup.sh`
  - creates the Python environment, installs dependencies, copies env templates, prepares block-storage folders, and seeds the mock Jitsi transcripts

## Recommended flow

1. From inside `data/`, run:

```bash
bash setup.sh
```

2. Start the database:

```bash
cd proj07-db
docker compose up -d
cd ..
```

3. Activate the environment:

```bash
source .venv/bin/activate
```

4. Run the runtimes in order:

```bash
bash external_data_training_runtime/run_external_data_training_batch.sh
bash endpoint_replay_runtime/run_endpoint_replay_batch.sh
bash online_inference_workflow_runtime/run_online_inference_workflow_batch.sh
bash retraining_dataset_runtime/run_retraining_dataset_batch.sh
```

Why this order:

- `endpoint_replay_runtime` is now independently runnable in synthetic mode
- `online_inference_workflow_runtime` is the separate raw-transcript-to-online-feature path

## Important note about first-time DB initialization

The SQL files under `proj07-db/init_sql/` are applied automatically by Postgres only when the database volume is initialized for the first time.

So this bundle works most cleanly when:

- `/mnt/block/postgres-data` is empty on first startup

If you reuse an old Postgres data directory from a previous schema version, the init SQL will not re-run automatically.
