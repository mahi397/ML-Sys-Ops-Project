# Initial Implementation

This folder preserves the original standalone data runtime bundle that landed on April 6, 2026.

The final integrated layout now uses `../proj07-db/` for schema/bootstrap assets and `../proj07-runtime/` for the running stack. Keep the folders here for lineage tracing, side-by-side comparison, or one-off batch reruns that still depend on the original scripts.

## What is here

- `external_data_training_runtime/`
  - AMI ingest plus the first Stage 1 dataset bootstrap flow
- `endpoint_replay_runtime/`
  - standalone hypothetical Stage 1 / Stage 2 endpoint replay flow
- `online_inference_workflow_runtime/`
  - standalone Jitsi ingest plus online-inference batch flow
- `retraining_dataset_runtime/`
  - rolling feedback-pool and retraining-snapshot batch builder
- `mock_jitsi_meet/`
  - sample Jitsi-style transcripts used by the original standalone workflow bundle
- `setup.sh`
  - archived-only bootstrap for the legacy batch runtimes

The archived batch scripts now default to the shared `../.env` file, while still allowing their old per-folder env files as a fallback or override.

## Relationship to the final integration

- `online_inference_workflow_runtime/`
  - source lineage for `../proj07-runtime/proj07_services/api/jitsi_transcript_receiver.py`, `../proj07-runtime/proj07_services/pipeline/ingest_saved_jitsi_transcript.py`, `../proj07-runtime/proj07_services/pipeline/build_online_inference_payloads.py`, and `../proj07-runtime/proj07_services/workers/user_summary_materialize_service.py`
- `endpoint_replay_runtime/`
  - source lineage for `../proj07-runtime/proj07_services/workers/stage1_forward_service.py` and `../proj07-runtime/proj07_services/workers/stage2_forward_service.py`
- `external_data_training_runtime/`
  - source lineage for the base schema and offline dataset bootstrap assumptions that now live under `../proj07-db/init_sql/`
- `retraining_dataset_runtime/`
  - preserved as a standalone batch workflow; later integration work shares its feedback-oriented schema expectations but does not replace the whole runtime with a single always-on service
