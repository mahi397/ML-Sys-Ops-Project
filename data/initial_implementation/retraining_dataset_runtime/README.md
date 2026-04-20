# Retraining Dataset Runtime

This folder now lives under `data/initial_implementation/` because it preserves the original standalone runtime from April 6, 2026. The final integrated layout now uses `../../proj07-db/` for schema/bootstrap assets and `../../proj07-runtime/` for the running stack.

This folder is a self-contained runtime bundle for compiling rolling retraining datasets:

- start from online-inference workflow meetings that already have corrected production-like state in Postgres
- only select meetings whose `meetings.is_valid = TRUE`
- only select meetings whose `meetings.dataset_version` is still `NULL`
- compile the next versioned Stage 1 feedback pool automatically
- build the next versioned `roberta_stage1` snapshot automatically where:
  - the latest existing dataset version is rolled forward into the new train split
  - only newly-arrived production meetings are assigned to validation and test
- stamp those newly-consumed meetings with the new `dataset_version` and split in Postgres

In other words:

- if the latest dataset is `v3`, the next run will create `v4`
- if the latest dataset is `v4`, the next run will create `v5`
- meetings already stamped with a non-null `dataset_version` will not be treated as new again

## Expected inputs

This runtime assumes these already exist:

1. Online-inference workflow meetings already processed into Postgres with:
   - generated recap output
   - feedback events already created
   - corrected recap state already materialized
   - `topic_segments.segment_type = 'user_corrected'` for structurally corrected meetings
   - `meetings.is_valid = TRUE` so incomplete ingest / Stage 1 / Stage 2 runs are excluded
   - `meetings.dataset_version IS NULL` for meetings that have not been consumed yet

2. Historical base dataset already present locally at:
   - `/mnt/block/roberta_stage1/vN/train.jsonl`
   - `/mnt/block/roberta_stage1/vN/val.jsonl`
   - `/mnt/block/roberta_stage1/vN/test.jsonl`

## Files

- `feedback_common.py`: shared DB/object-store helpers
- `discover_retraining_dataset_meetings.py`: lists only structurally corrected Jitsi meetings that have not yet been consumed into a dataset snapshot
- `build_feedback_pool.py`: compiles `roberta_stage1_feedback_pool/vN`
- `build_retraining_snapshot.py`: builds the new `roberta_stage1/vN` train/val/test snapshot
- `run_retraining_dataset_batch.sh`: batch runner for the whole runtime flow
- `retraining_dataset.env.example`: environment template
- `requirements.txt`: minimal Python dependency list

## VM setup

1. Copy this folder to the VM.
2. Copy `retraining_dataset.env.example` to `retraining_dataset.env`.
3. Update values if your DB, bucket, or local dataset roots differ.
4. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

5. Make sure your database has the updated base and feedback schemas applied, because this runtime relies on:
   - `../../proj07-db/init_sql/001_init_postgres_schema.sql`
   - `../../proj07-db/init_sql/002_feedback_loop_schema.sql`
   - `meetings.dataset_version`
   - `meetings.dataset_split`

## Run all discovered eligible meetings

```bash
bash run_retraining_dataset_batch.sh
```

## Run only selected meetings

```bash
bash run_retraining_dataset_batch.sh \
  jitsi_20260401T201408Z_9f8732d1 \
  jitsi_20260403T180944Z_61459c2e
```

## Local outputs

- feedback pool:
  - `/mnt/block/staging/feedback_loop/datasets/roberta_stage1_feedback_pool/vN/`
- final retraining snapshot:
  - `/mnt/block/roberta_stage1/vN/`

## Object-store outputs

- `datasets/roberta_stage1_feedback_pool/vN/`
- `datasets/roberta_stage1/vN/`
