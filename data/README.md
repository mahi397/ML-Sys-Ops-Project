# Data Runtime

The data services are part of the root system stack. The only Compose entry point for this repository is now:

```bash
cd ..
docker compose up -d
```

Use the root `.env` as the single global environment file. `data/setup.sh` reads that file, prepares the data runtime dependencies, applies schema migrations, and starts the runtime services through the root `docker-compose.yml`.

## Services

The root compose starts these data services by default:

- `postgres`
- `adminer`
- `jitsi_transcript_receiver`
- `db_task_worker`
- `stage1_payload_service`
- `stage1_forward_service`
- `stage2_input_service`
- `stage2_forward_service`
- `user_summary_materialize_service`
- `retraining_dataset_service`

`traffic-generator` and `production_drift_monitor` are manual profile services. Start them explicitly when needed:

```bash
docker compose --profile emulated-traffic up -d traffic-generator
docker compose logs -f traffic-generator
docker compose --profile emulated-traffic stop traffic-generator

docker compose --profile drift-monitor up -d production_drift_monitor
docker compose logs -f production_drift_monitor
docker compose --profile drift-monitor stop production_drift_monitor
```

The archived `initial_implementation/` tree is independent reference material. The active runtime and setup flow do not depend on it.

## One-Off Jobs

Bootstrap AMI corpus data into Postgres:

```bash
cd data/proj07-runtime
python -m proj07_services.pipeline.bootstrap_ami_corpus \
  --rclone-remote "${RCLONE_REMOTE:-rclone_s3}" \
  --bucket "${OBJECT_BUCKET:-${BUCKET:-objstore-proj07}}" \
  --prefix "${AMI_OBJECT_PREFIX:-ami_public_manual_1.6.2}" \
  --raw-root "${BLOCK_ROOT:-/mnt/block}/staging/current_job/raw" \
  --processed-root "${BLOCK_ROOT:-/mnt/block}/staging/current_job/processed" \
  --log-file "${BLOCK_ROOT:-/mnt/block}/ingest_logs/ami_corpus_bootstrap.log"
```

Add `--meeting ES2002a` to ingest a single AMI meeting instead of the full corpus. This command expects the raw AMI corpus to already be present in object storage; `./data/setup.sh` can stage it first.

Generate synthetic Stage 1 bootstrap data:

```bash
cd data/proj07-runtime
python -m proj07_services.pipeline.generate_synthetic_stage1_inputs \
  --output-root "${BLOCK_ROOT:-/mnt/block}/user-behaviour/inference_requests/stage1" \
  --version "${BOOTSTRAP_SYNTHETIC_STAGE1_VERSION:-1}" \
  --meeting-count "${BOOTSTRAP_SYNTHETIC_STAGE1_MEETING_COUNT:-3}" \
  --seed "${BOOTSTRAP_SYNTHETIC_STAGE1_SEED:-42}" \
  --upload-artifacts \
  --rclone-remote "${RCLONE_REMOTE:-rclone_s3}" \
  --bucket "${OBJECT_BUCKET:-${BUCKET:-objstore-proj07}}"
```

Force one retraining dataset cycle:

```bash
docker compose exec retraining_dataset_service \
  python -m proj07_services.workers.retraining_dataset_service --once --force-run
```

Dry-run the retraining dataset worker without writing artifacts:

```bash
docker compose exec retraining_dataset_service \
  python -m proj07_services.workers.retraining_dataset_service --once --dry-run
```

Force one production drift check:

```bash
docker compose --profile drift-monitor up -d production_drift_monitor
docker compose exec production_drift_monitor \
  python -m proj07_services.workers.production_drift_monitor --once
docker compose --profile drift-monitor stop production_drift_monitor
```

`bash setup.sh` no longer runs retraining dataset lineage restore. Any future lineage/restore workflow should be run manually instead.
