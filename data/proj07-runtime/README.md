# proj07-runtime

This folder is the canonical final runnable stack under `data/`.

It supersedes the original standalone runtime bundle now grouped under `../initial_implementation/` and contains the Jitsi transcript ingest service plus the async workers that build Stage 1 requests, forward Stage 1 inference, derive Stage 2 segment inputs, materialize user-summary state, and monitor whether enough new valid meetings / feedback have accumulated to build the next retraining dataset snapshot.

Database-specific source assets stay under `../proj07-db/`, but this folder owns the runnable stack. Its compose file launches Postgres, Adminer, and the current ingest/workflow services together, while mounting `../proj07-db/init_sql/` into Postgres during bootstrap.

## Package layout

- `proj07_services/api/`
  - HTTP entrypoints such as `jitsi_transcript_receiver.py`
- `proj07_services/pipeline/`
  - ingest and payload-building flows, including the AMI corpus bootstrap modules that backfill canonical AMI meetings into Postgres from object storage
- `proj07_services/workers/`
  - long-running task workers
- `proj07_services/common/`
  - shared DB, logging, artifact, and workflow helpers

## Start the full stack

```bash
cd ..
cp .env.example .env
bash setup.sh
cd proj07-runtime
docker compose up -d
```

The canonical shared config file is `../.env`. `../setup.sh` keeps this folder's `.env` pointed at that shared file, so updating `data/.env` is enough for DB settings, object-store settings, and Stage 1 / Stage 2 inference URLs.

## Relationship to the initial implementation and DB layer

- `proj07_services/common/feedback_common.py`
  - directly reused from the original runtime bundle
- `proj07_services/pipeline/build_online_inference_payloads.py` and `proj07_services/pipeline/ingest_saved_jitsi_transcript.py`
  - service-oriented evolutions of the original online-inference scripts
- `proj07_services/pipeline/bootstrap_ami_corpus.py` and `proj07_services/pipeline/ingest_ami_meeting.py`
  - runtime-owned reproduction of the archived AMI ingest flow, updated to write the current `proj07-db` schema instead of relying on the legacy bundle
- `proj07_services/workers/stage1_forward_service.py`, `proj07_services/workers/stage2_forward_service.py`, and `proj07_services/workers/user_summary_materialize_service.py`
  - reproduce the older batch flow as durable worker services
- `proj07_services/workers/db_task_worker.py`, `proj07_services/common/task_service_common.py`, `proj07_services/common/workflow_task_common.py`, `proj07_services/workers/stage1_payload_service.py`, and `proj07_services/workers/stage2_input_service.py`
  - integration-specific additions introduced after the original bundle
- `../proj07-db/init_sql/` contains the schema/bootstrap SQL that these services depend on.
- `docker-compose.yml` in this folder owns the Postgres and Adminer containers that these services talk to.

## Current flow

1. `jitsi_transcript_receiver.py` accepts `POST /ingest/jitsi-transcript`, validates the upload, and saves the raw transcript locally.
2. It launches `ingest_saved_jitsi_transcript.py`.
3. The ingester parses the transcript, uploads raw and parsed transcript artifacts, and inserts the meeting into Postgres.
4. `db_task_worker.py` scans Postgres and upserts operational work into `workflow_tasks`.
5. `stage1_payload_service.py` claims `stage1_build` tasks from Postgres and runs `build_online_inference_payloads.py`.
6. The builder writes `stage1_requests.jsonl`, `stage1_requests.json`, `model_utterances.json`, and `manifest.json`, optionally uploads them, and upserts their references into `meeting_artifacts`.
7. `db_task_worker.py` or the builder success path upserts a `stage1_forward` task.
8. `stage1_forward_service.py` claims that task from Postgres, posts the Stage 1 payload to the configured HTTP endpoint, saves the response locally, optionally uploads it, and upserts response artifact rows into `meeting_artifacts`.
9. `stage1_forward_service.py` enqueues `stage2_build` when row-wise Stage 1 responses are available.
10. `stage2_input_service.py` loads `model_utterances.json` plus `stage1_responses.jsonl`, reconstructs predicted segments, writes Stage 2 input artifacts, and upserts their references into `meeting_artifacts`.
11. `stage2_input_service.py` enqueues `stage2_forward`.
12. `stage2_forward_service.py` posts `stage2_inputs.json` to `POST /summarize`, saves the Stage 2 response locally, optionally uploads it, and upserts response artifact rows into `meeting_artifacts`.
13. `db_task_worker.py` continuously refreshes `meetings.is_valid` so retraining can exclude incomplete Jitsi meetings whose ingest or Stage 1 / Stage 2 artifact chain is missing.
14. `retraining_dataset_service.py` watches for valid unconsumed meetings and structural feedback volume, then builds the next `roberta_stage1_feedback_pool/vN` and `roberta_stage1/vN` snapshot when thresholds are crossed.
15. `production_drift_monitor.py` compares rolling live Jitsi meeting features against the latest approved Stage 1 dataset profile and writes a persisted quality report for drift investigation.

The key split is:
- `meeting_artifacts` describes durable data outputs
- `workflow_tasks` and `workflow_task_attempts` describe operational state and retry history

## AMI bootstrap

The runtime also owns the AMI historical-data bootstrap path used by `../setup.sh`.

- `bootstrap_ami_corpus.py`
  - discovers AMI meetings from object storage, compares them against fully-ingested `source_type = 'ami'` rows in Postgres, stages missing raw XML locally, and invokes the per-meeting ingester only when coverage is incomplete
- `ingest_ami_meeting.py`
  - parses one staged AMI meeting, uploads processed transcript/summary artifacts, inserts `meetings`, `users`, `meeting_participants`, `meeting_speakers`, `utterances`, `topic_segments`, `utterance_transitions`, `meeting_artifacts`, and `summaries`, and repairs partial AMI rows by replacing the existing meeting when needed
- `restore_dataset_lineage.py`
  - restores historical `roberta_stage1_feedback_pool/vN` and `roberta_stage1/vN` directories from block storage or object storage, upserts their `dataset_versions` manifests, attempts to recover any lineage-referenced historical `jitsi_*` meetings from stored parsed transcript payloads plus verified Stage 1 / Stage 2 artifacts, and then replays per-meeting `dataset_version` / `dataset_split` stamps

Manual trigger:

```bash
cd proj07-runtime
python -m proj07_services.pipeline.bootstrap_ami_corpus
python -m proj07_services.retraining.restore_dataset_lineage
```

`restore_dataset_lineage.py` is intentionally conservative. It will only recover missing historical Jitsi meetings when the stored parsed transcript payload and any claimed artifact files actually exist in object storage; otherwise it still fails instead of silently restamping the wrong lineage. For a truly fresh environment, clear those stored dataset prefixes or disable `BOOTSTRAP_DATASET_LINEAGE_ENABLED` before running `../setup.sh`.

## Core DB tables

### `meetings`

Stores one row per ingested meeting.

Common columns used in this flow:
- `meeting_id`
- `source_type`
- `source_name`
- `started_at`
- `ended_at`
- `raw_folder_prefix`
- `is_valid`

### `meeting_artifacts`

Stores pointers to durable artifacts, usually either object-storage keys or temporary `local://...` pointers during upload handoff.

Common columns used in this flow:
- `meeting_id`
- `artifact_type`
- `object_key`
- `content_type`
- `artifact_version`
- `created_at`

### `utterances`

Stores parsed meeting utterances.

Common columns used in this flow:
- `utterance_id`
- `meeting_id`
- `meeting_speaker_id`
- `utterance_index`
- `start_time_sec`
- `end_time_sec`
- `raw_text`
- `clean_text`

### `utterance_transitions`

Stores placeholder source-level adjacency rows between consecutive utterances.

Common columns used in this flow:
- `meeting_id`
- `left_utterance_id`
- `right_utterance_id`
- `transition_index`
- `gold_boundary_label`
- `pred_boundary_prob`
- `pred_boundary_label`

Important note:
- Stage 1 request building does not use `utterance_transitions`
- Stage 1 currently operates on derived model utterances, not directly on source adjacency rows

### `workflow_tasks`

Stores durable operational state for background work.

Columns:
- `task_id`
- `task_type`
- `meeting_id`
- `artifact_version`
- `status`
- `payload_json`
- `attempt_count`
- `max_attempts`
- `next_attempt_at`
- `locked_by`
- `locked_at`
- `heartbeat_at`
- `last_error`
- `created_at`
- `updated_at`

Statuses:
- `pending`
- `running`
- `retry_scheduled`
- `succeeded`
- `failed_permanent`
- `cancelled`

Uniqueness:
- `(task_type, meeting_id, artifact_version)` is unique

Current task types:
- `stage1_build`
- `stage1_forward`
- `stage2_build`
- `stage2_forward`

### `dataset_versions`

Stores versioned dataset manifests produced by feedback-pool and retraining snapshot builds.

Common columns used in this flow:
- `dataset_name`
- `stage`
- `source_type`
- `object_key`
- `manifest_json`
- `created_at`

### `workflow_task_attempts`

Stores one row per task attempt for audit and debugging.

Columns:
- `attempt_id`
- `task_id`
- `attempt_number`
- `worker_id`
- `started_at`
- `finished_at`
- `outcome`
- `error_summary`
- `stderr_tail`
- `duration_ms`

## Services

### `postgres`

Role:
- Stores transcript data
- Stores artifact references
- Stores workflow task state

### `adminer`

Role:
- DB inspection UI only

### `retraining_dataset_service.py`

Role:
- Polls Postgres for valid Jitsi meetings whose `dataset_version` is still `NULL`
- Checks whether the threshold has been crossed for unconsumed valid meetings or structural feedback events
- Builds the next Stage 1 feedback pool and rolling retraining snapshot when thresholds are met
- Marks meetings consumed by the snapshot with the new `dataset_version` and `dataset_split`
- Profiles each candidate feedback pool / snapshot, compares it against the latest approved reference profile, and quarantines candidates whose drift exceeds the configured threshold

Manual trigger:

```bash
docker compose exec retraining_dataset_service \
  python -m proj07_services.workers.retraining_dataset_service --once --force-run
```

### `production_drift_monitor.py`

Role:
- Polls recent valid Jitsi meetings from Postgres on a schedule
- Rebuilds inference-style feature distributions from live utterances
- Compares the live window against the latest approved `roberta_stage1` profile
- Persists a `dataset_quality_reports` row plus a JSON report under `/mnt/block/staging/feedback_loop/production_drift_reports`

Manual trigger:

```bash
docker compose exec production_drift_monitor \
  python -m proj07_services.workers.production_drift_monitor --once
```

### `jitsi_transcript_receiver.py`

Role:
- FastAPI upload endpoint: `POST /ingest/jitsi-transcript`
- Validates the transcript upload
- Saves the raw transcript under `/mnt/block/user-behaviour/received_transcripts`
- Launches the ingester subprocess
- Returns `400` for invalid parsed transcripts such as missing metadata or zero spoken utterances

DB writes:
- none directly

Local files written:
- raw transcript `.txt`
- upload metadata sidecar `.meta.json`

### `ingest_saved_jitsi_transcript.py`

Role:
- Parses a saved Jitsi transcript
- Uploads raw and parsed transcript artifacts
- Inserts the meeting and transcript rows into Postgres
- Can still run Stage 1 inline if enabled, but the default architecture is async via `workflow_tasks`

DB writes:

#### `users`
- inserts or updates `user_id`
- updates `display_name`
- inserts `email` as `NULL`

#### `meetings`
- inserts `meeting_id`
- inserts `source_type`
- inserts `source_name`
- inserts `started_at`
- inserts `ended_at`
- inserts `raw_folder_prefix`

#### `meeting_participants`
- inserts `meeting_id`
- inserts `user_id`
- inserts `role`
- inserts `can_view_summary`
- inserts `can_edit_summary`
- inserts `joined_at`
- inserts `left_at`

#### `meeting_speakers`
- inserts `meeting_id`
- inserts `user_id` as `NULL`
- inserts `speaker_label`
- inserts `display_name`
- inserts `role`

#### `meeting_artifacts`
- upserts `raw_transcript`
- upserts `parsed_transcript`

Written columns:
- `meeting_id`
- `artifact_type`
- `object_key`
- `content_type`
- `artifact_version`

#### `utterances`
- inserts parsed utterances

Written columns:
- `meeting_id`
- `meeting_speaker_id`
- `utterance_index`
- `start_time_sec`
- `end_time_sec`
- `raw_text`
- `clean_text`
- `source_segment_id` as `NULL`

#### `utterance_transitions`
- inserts placeholder adjacency rows

Written columns:
- `meeting_id`
- `left_utterance_id`
- `right_utterance_id`
- `transition_index`
- `gold_boundary_label`
- `pred_boundary_prob`
- `pred_boundary_label`

### `db_task_worker.py`

Role:
- dispatcher only
- scans Postgres for missing downstream work
- upserts `workflow_tasks`
- requeues stale `running` tasks whose heartbeat expired
- does not build payloads
- does not call the HTTP inference endpoint

DB reads:

#### Stage 1 build dispatch
- `meetings.meeting_id`
- `meetings.started_at`
- `meetings.ended_at`
- `utterances.utterance_id`
- `utterances.meeting_id`
- `meeting_artifacts.artifact_id`
- `meeting_artifacts.artifact_type`
- `meeting_artifacts.artifact_version`

Purpose:
- find meetings with utterances but missing one or more Stage 1 request artifacts

#### Stage 1 forward dispatch
- `meeting_artifacts.meeting_id`
- `meeting_artifacts.artifact_id`
- `meeting_artifacts.artifact_type`
- `meeting_artifacts.artifact_version`
- `meeting_artifacts.created_at`

Purpose:
- find meetings that have `stage1_requests_jsonl` but do not yet have `stage1_responses_json`

#### Stage 2 build dispatch
- `meeting_artifacts.meeting_id`
- `meeting_artifacts.artifact_id`
- `meeting_artifacts.artifact_type`
- `meeting_artifacts.artifact_version`
- `meeting_artifacts.created_at`

Purpose:
- find meetings that have `stage1_responses_jsonl` but are missing one or more Stage 2 segmentation artifacts

#### Stage 2 forward dispatch
- `meeting_artifacts.meeting_id`
- `meeting_artifacts.artifact_id`
- `meeting_artifacts.artifact_type`
- `meeting_artifacts.artifact_version`
- `meeting_artifacts.created_at`

Purpose:
- find meetings that have `stage2_inputs_json` but do not yet have `stage2_responses_json`

DB writes:

#### `workflow_tasks`
- upserts `stage1_build`
- upserts `stage1_forward`
- upserts `stage2_build`
- upserts `stage2_forward`
- may move stale tasks from `running` to `retry_scheduled` or `failed_permanent`

Written or updated columns:
- `task_type`
- `meeting_id`
- `artifact_version`
- `status`
- `payload_json`
- `max_attempts`
- `next_attempt_at`
- `locked_by`
- `locked_at`
- `heartbeat_at`
- `last_error`
- `updated_at`

#### `workflow_task_attempts`
- closes stale in-flight attempts during stale-task sweeps

### `stage1_payload_service.py`

Role:
- claims `stage1_build` rows from `workflow_tasks`
- reconciles whether Stage 1 request artifacts are missing in DB or on local disk
- runs `build_online_inference_payloads.py` for one meeting at a time
- marks task success or schedules DB-backed retries
- enqueues downstream `stage1_forward` work on successful completion

DB reads:
- `workflow_tasks`
- `meeting_artifacts`

DB writes:

#### `workflow_tasks`
- claims one `stage1_build` task by moving it to `running`
- updates `attempt_count`, `locked_by`, `locked_at`, `heartbeat_at`
- marks it `succeeded`, `retry_scheduled`, or `failed_permanent`
- upserts downstream `stage1_forward` tasks on success

#### `workflow_task_attempts`
- inserts one attempt row per claimed task
- fills `finished_at`, `outcome`, `error_summary`, `stderr_tail`, `duration_ms`

Artifact DB writes:
- none directly here
- actual artifact upserts happen inside `build_online_inference_payloads.py`

### `build_online_inference_payloads.py`

Role:
- one-meeting Stage 1 builder
- reads utterances from DB
- derives model utterances
- creates Stage 1 sliding-window requests
- saves artifacts locally
- optionally uploads them to object storage via `rclone`
- upserts Stage 1 request artifact references in Postgres

Important notes:
- does not use `utterance_transitions` to build Stage 1 requests
- uses `utterances` plus `meeting_speakers`
- is idempotent for the same `meeting_id` and `artifact_version`

Live Jitsi gating rules enforced here:
- zero cleaned utterances: skip inference
- all cleaned utterances below `min_utterance_chars` (default `20`): skip inference
- fewer than `2` cleaned eligible source utterances: skip inference
- `2` to `6` cleaned eligible source utterances: allow inference and record `short_meeting_low_confidence`
- `7+` cleaned eligible source utterances: normal inference

DB reads:

#### `utterances`
- `meeting_id`
- `utterance_id`
- `utterance_index`
- `start_time_sec`
- `end_time_sec`
- `clean_text`
- `meeting_speaker_id`

#### `meeting_speakers`
- `meeting_speaker_id`
- `speaker_label`

DB writes:

#### `meeting_artifacts`
- upserts `stage1_requests_jsonl`
- upserts `stage1_requests_json`
- upserts `stage1_model_utterances_json`
- upserts `stage1_manifest_json`
- may also upsert Stage 2 input artifacts if Stage 2 build is explicitly requested

Updated columns:
- `object_key`
- `content_type`
- `created_at`

Local artifacts written:
- `/mnt/block/user-behaviour/inference_requests/stage1/<meeting_id>/v<version>/stage1_requests.jsonl`
- `/mnt/block/user-behaviour/inference_requests/stage1/<meeting_id>/v<version>/stage1_requests.json`
- `/mnt/block/user-behaviour/inference_requests/stage1/<meeting_id>/v<version>/model_utterances.json`
- `/mnt/block/user-behaviour/inference_requests/stage1/<meeting_id>/v<version>/manifest.json`

### `stage1_forward_service.py`

Role:
- claims `stage1_forward` rows from `workflow_tasks`
- loads Stage 1 request artifacts from local disk
- posts them to the configured HTTP endpoint
- saves response artifacts locally
- optionally uploads response artifacts to object storage
- upserts response artifact references in Postgres

DB reads:
- `workflow_tasks`
- `meeting_artifacts`

DB writes:

#### `workflow_tasks`
- claims one `stage1_forward` task by moving it to `running`
- updates `attempt_count`, `locked_by`, `locked_at`, `heartbeat_at`
- marks it `succeeded`, `retry_scheduled`, or `failed_permanent`

#### `workflow_task_attempts`
- inserts one attempt row per claimed task
- fills `finished_at`, `outcome`, `error_summary`, `stderr_tail`, `duration_ms`

#### `meeting_artifacts`
- upserts `stage1_responses_json`
- upserts `stage1_responses_jsonl` when the endpoint response can be represented as row-wise JSONL

Updated columns:
- `object_key`
- `content_type`
- `created_at`

Important behavior:
- if the Stage 1 request file is empty, the service does not call the endpoint
- it writes a skipped local response and records `stage1_responses_json`
- if upload is enabled, it may temporarily store `stage1_responses_json` as `local://...` until object-storage upload succeeds
- it does not currently write predictions back into `utterance_transitions`
- when row-wise response JSONL exists, it enqueues downstream `stage2_build`

### `stage2_input_service.py`

Role:
- claims `stage2_build` rows from `workflow_tasks`
- loads Stage 1 model utterances and Stage 1 response JSONL from local disk
- reconstructs predicted topic segments from Stage 1 boundary decisions
- writes Stage 2 input artifacts plus `reconstructed_segments.json`
- optionally uploads those artifacts to object storage
- upserts Stage 2 artifact references in Postgres
- enqueues downstream `stage2_forward`

DB reads:
- `workflow_tasks`
- `meeting_artifacts`

DB writes:

#### `workflow_tasks`
- claims one `stage2_build` task by moving it to `running`
- updates `attempt_count`, `locked_by`, `locked_at`, `heartbeat_at`
- marks it `succeeded`, `retry_scheduled`, or `failed_permanent`

#### `workflow_task_attempts`
- inserts one attempt row per claimed task
- fills `finished_at`, `outcome`, `error_summary`, `stderr_tail`, `duration_ms`

#### `meeting_artifacts`
- upserts `stage2_inputs_jsonl`
- upserts `stage2_inputs_json`
- upserts `reconstructed_segments_json`

### `stage2_forward_service.py`

Role:
- claims `stage2_forward` rows from `workflow_tasks`
- loads `stage2_inputs.json` from local disk
- by default posts each segment object individually to the configured Stage 2 endpoint, typically `POST /summarize`
- uses its own `STAGE2_FORWARD_URL`; it does not derive the endpoint from Stage 1 settings
- saves response artifacts locally
- optionally uploads response artifacts to object storage
- upserts Stage 2 response artifact references in Postgres

DB reads:
- `workflow_tasks`
- `meeting_artifacts`

DB writes:

#### `workflow_tasks`
- claims one `stage2_forward` task by moving it to `running`
- updates `attempt_count`, `locked_by`, `locked_at`, `heartbeat_at`
- marks it `succeeded`, `retry_scheduled`, or `failed_permanent`

#### `workflow_task_attempts`
- inserts one attempt row per claimed task
- fills `finished_at`, `outcome`, `error_summary`, `stderr_tail`, `duration_ms`

#### `meeting_artifacts`
- upserts `stage2_responses_json`
- upserts `stage2_responses_jsonl` when the endpoint response can be represented as row-wise JSONL

## Workflow task lifecycle

Typical `stage1_build` lifecycle:
1. `db_task_worker.py` upserts `workflow_tasks(task_type='stage1_build', status='pending')`
2. `stage1_payload_service.py` claims it and moves it to `running`
3. heartbeat updates `heartbeat_at` while the builder subprocess runs
4. on success, task becomes `succeeded` and a downstream `stage1_forward` task is upserted
5. on failure, task becomes `retry_scheduled` or `failed_permanent`

Typical `stage1_forward` lifecycle:
1. dispatcher or builder-success path upserts `workflow_tasks(task_type='stage1_forward', status='pending')`
2. `stage1_forward_service.py` claims it and moves it to `running`
3. heartbeat updates `heartbeat_at` while HTTP post and artifact upload run
4. on success, task becomes `succeeded`
5. on failure, task becomes `retry_scheduled` or `failed_permanent`

Retry model:
- retries are stored in Postgres, not in memory
- `attempt_count` increments on claim
- `next_attempt_at` controls delayed retries
- backoff is exponential with jitter
- a stale-task sweep moves abandoned `running` tasks back to retryable state or permanent failure

## Artifact locations

### Local

- received transcripts:
  - `/mnt/block/user-behaviour/received_transcripts`
- parsed transcript outputs:
  - `/mnt/block/user-behaviour/parsed_transcripts`
- Stage 1 request artifacts:
  - `/mnt/block/user-behaviour/inference_requests/stage1/<meeting_id>/v<version>/`
- Stage 1 response artifacts:
  - `/mnt/block/user-behaviour/inference_responses/stage1/<meeting_id>/v<version>/`

### Object storage

Default prefixes:
- Stage 1 requests:
  - `production/inference_requests/stage1/<meeting_id>/v<version>/`
- Stage 1 responses:
  - `production/inference_responses/stage1/<meeting_id>/v<version>/`

## What to inspect in the DB

If you want a quick state check, the most useful tables are:

- `meetings`
  - confirms the meeting exists
- `utterances`
  - confirms the transcript was parsed and inserted
- `meeting_artifacts`
  - confirms transcript, Stage 1 request, and Stage 1 response artifacts
- `workflow_tasks`
  - confirms whether build and forward work is pending, running, retrying, or done
- `workflow_task_attempts`
  - shows attempt history, recent failures, and stderr tails
- `utterance_transitions`
  - confirms placeholder source-adjacency rows exist

Typical `meeting_artifacts` progression:
1. `raw_transcript`
2. `parsed_transcript`
3. `stage1_requests_jsonl`
4. `stage1_requests_json`
5. `stage1_model_utterances_json`
6. `stage1_manifest_json`
7. `stage1_responses_json`
8. `stage1_responses_jsonl` when the endpoint returns row-like results

Typical `workflow_tasks` progression:
1. `stage1_build`
2. `stage1_forward`

## Failure model

- `jitsi_transcript_receiver.py` is independent of Stage 1 build and Stage 1 forwarding
- `db_task_worker.py` is independent of the actual build and post work
- `stage1_payload_service.py` and `stage1_forward_service.py` are independent workers
- missing Stage 1 request artifacts in DB or on local disk cause the build path to reconcile
- missing Stage 1 response artifacts cause the forward path to reconcile
- task retries and stale-task recovery are DB-backed
