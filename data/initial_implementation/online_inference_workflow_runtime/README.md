# Online Inference Workflow Runtime

This folder now lives under `data/initial_implementation/` because it preserves the original standalone runtime from April 6, 2026. The final integrated layout now uses `../../proj07-db/` for schema/bootstrap assets and `../../proj07-runtime/` for the running stack.

This folder is a self-contained runtime bundle for the mocked online inference workflow:

- ingest Jitsi-style transcripts from block storage
- derive the internal `meeting_id` strictly from the transcript filename
- store parsed transcript data in Postgres and object storage
- build Stage 1 online inference payloads
- replay mock Stage 1 and Stage 2 endpoints
- store reconstructed segments and generated recap artifacts
- generate emulated user feedback events
- materialize corrected recap state and `user_corrected` segments
- register uploaded runtime artifacts in `meeting_artifacts`
- register generated recap output in `summaries`, `topic_segments`, and `segment_summaries`

## Expected transcript filename format

Input files must follow the real Jitsi-style naming pattern:

`transcript_YYYY-MM-DDTHH:MM:SS(.fraction)Z_<uuid>.txt`

Example:

`transcript_2026-04-01T20:14:08.123456789Z_9f8732d1-0a2e-4a5f-8d6d-3f2b1c8e7411.txt`

The internal meeting id is derived automatically as:

`jitsi_20260401T201408Z_9f8732d1`

Manual meeting-id entry is intentionally not supported.

## Files

- `ingest_jitsi_transcript.py`: parses and ingests a Jitsi transcript
- `build_online_inference_payloads.py`: builds Stage 1 online inference artifacts
- `replay_to_hypothetical_endpoints.py`: replays mock Stage 1 and Stage 2 service calls
- `generate_feedback_events.py`: creates emulated user feedback events from generated recap output
- `materialize_corrected_recap.py`: applies feedback events and writes corrected recap state
- `feedback_common.py`: shared helpers used by the runtime scripts
- `flowise_stage2_prompt.txt`: prompt template used by the replay script
- `run_online_inference_workflow_batch.sh`: batch runner for one or many transcripts
- `online_inference_workflow.env.example`: environment template
- `requirements.txt`: minimal Python dependency list

## VM setup

1. Copy this folder to the VM.
2. Copy `online_inference_workflow.env.example` to `online_inference_workflow.env`.
3. Update values if your database, bucket, or paths differ.
4. Ensure transcripts are present under:
   `/mnt/block/user-behaviour/Transcripts`
5. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

6. Make sure the database has the updated `meeting_artifacts` constraint from:

- `../../proj07-db/init_sql/001_init_postgres_schema.sql`
- `../../proj07-db/init_sql/002_feedback_loop_schema.sql`

If you already have an existing database, re-apply the updated SQL before running this bundle.

## Run all transcripts in the transcript root

```bash
bash run_online_inference_workflow_batch.sh
```

## Run only selected transcripts

```bash
bash run_online_inference_workflow_batch.sh \
  transcript_2026-04-01T20:14:08.123456789Z_9f8732d1-0a2e-4a5f-8d6d-3f2b1c8e7411.txt \
  transcript_2026-04-03T18:09:44.223456789Z_61459c2e-3b08-41f5-b5a0-5d0bcdf4f812.txt
```

## Outputs

Local outputs are written under `/mnt/block/user-behaviour` and `/mnt/block/staging/feedback_loop` by default:

- `/mnt/block/user-behaviour/parsed_transcripts/`
- `/mnt/block/user-behaviour/online_inference/stage1/`
- `/mnt/block/user-behaviour/inference_responses/stage1/`
- `/mnt/block/user-behaviour/online_inference/stage2/`
- `/mnt/block/user-behaviour/inference_responses/stage2/`
- `/mnt/block/user-behaviour/reconstructed_segments/`
- `/mnt/block/user-behaviour/recaps/generated/`
- `/mnt/block/staging/feedback_loop/feedback_events/`
- `/mnt/block/staging/feedback_loop/edited_recaps/`
- `/mnt/block/user-behaviour/logs/online_inference_workflow/`

Object-store outputs are written under `production/` by default:

- `production/jitsi/raw_transcripts/{meeting_id}/`
- `production/jitsi/parsed_transcripts/{meeting_id}/`
- `production/inference_requests/stage1/{meeting_id}/`
- `production/inference_responses/stage1/{meeting_id}/`
- `production/inference_requests/stage2/{meeting_id}/`
- `production/inference_responses/stage2/{meeting_id}/`
- `production/reconstructed_segments/{meeting_id}/`
- `production/recaps/generated/{meeting_id}/`
- `production/feedback_events/{meeting_id}/`
- `production/recaps/edited/{meeting_id}/`
