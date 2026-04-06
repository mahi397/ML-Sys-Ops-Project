# External Data Training Runtime

This folder is a self-contained runtime bundle for external-data ingestion and training-set preparation:

- ingest external AMI data from Chameleon object storage
- store durable processed artifacts back in object storage
- load canonical meeting data into Postgres
- build the Stage 1 train/val/test dataset as `roberta_stage1/v1`
- build the augmented Stage 1 dataset as `roberta_stage1/v2`

It uses these scripts:

- `run_ingest.py`: stages one or more AMI meetings locally and launches per-meeting ingest
- `ingest_one_meeting.py`: parses a single AMI meeting, uploads processed artifacts, and loads normalized rows into Postgres
- `build_st1_db.py`: compiles the initial Stage 1 dataset from canonical Postgres data
- `augment_stage1_train.py`: conservatively augments the `train` split into a new `v2` dataset
- `run_external_data_training_batch.sh`: orchestrates the whole runtime flow
- `001_init_postgres_schema.sql`: base schema needed for this objective

One important detail:

- when `build_st1_db.py` creates `roberta_stage1/v1`, it also stamps `meetings.dataset_version = 1`
- it writes `meetings.dataset_split = train | val | test`
- the later `v2` augmentation keeps the same AMI meeting membership and only adds synthetic training rows

## Expected inputs

This runtime expects:

- Dockerized Postgres running and reachable as the configured container
- `rclone` configured for the Chameleon object store
- AMI raw files already present in object storage under:
  - `ami_public_manual_1.6.2/corpusResources/`
  - `ami_public_manual_1.6.2/ontologies/`
  - `ami_public_manual_1.6.2/words/`
  - `ami_public_manual_1.6.2/segments/`
  - `ami_public_manual_1.6.2/topics/`
  - `ami_public_manual_1.6.2/abstractive/`

## VM setup

1. Copy this folder to the VM.
2. Copy `external_data_training.env.example` to `external_data_training.env`.
3. Update values if your VM paths, bucket, or DB settings differ.
4. The scripts use only the Python standard library, so `requirements.txt` is informational.

## Run the default AMI subset

```bash
bash run_external_data_training_batch.sh
```

## Run selected meetings

```bash
bash run_external_data_training_batch.sh ES2011c ES2014a TS3005c
```

## Local outputs

- processed per-meeting artifacts:
  - `/mnt/block/staging/current_job/processed/<meeting_id>/`
- initial Stage 1 dataset:
  - `/mnt/block/roberta_stage1/v1/`
- augmented Stage 1 dataset:
  - `/mnt/block/roberta_stage1/v2/`

## Object-store outputs

- processed AMI artifacts:
  - `processed/ami/v1/transcripts/<meeting_id>.json`
  - `processed/ami/v1/summaries/<meeting_id>.json`
  - `processed/ami/v1/manifests/<meeting_id>.json`
- initial Stage 1 dataset:
  - `datasets/roberta_stage1/v1/`
- augmented Stage 1 dataset:
  - `datasets/roberta_stage1/v2/`
