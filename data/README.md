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
- `production_drift_monitor`

`traffic-generator` is the only manual service. Start it explicitly when you want emulated production uploads:

```bash
docker compose --profile emulated-traffic up -d traffic-generator
docker compose logs -f traffic-generator
```

The archived `initial_implementation/` tree is independent reference material. The active runtime and setup flow do not depend on it.

## One-Off Jobs

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

Force one production drift check:

```bash
docker compose exec production_drift_monitor \
  python -m proj07_services.workers.production_drift_monitor --once
```

Restore stored dataset lineage:

```bash
cd data/proj07-runtime
python -m proj07_services.retraining.restore_dataset_lineage \
  --rclone-remote "${RCLONE_REMOTE:-rclone_s3}" \
  --bucket "${OBJECT_BUCKET:-${BUCKET:-objstore-proj07}}" \
  --log-file "${BLOCK_ROOT:-/mnt/block}/ingest_logs/retraining_dataset_lineage_restore.log"
```
