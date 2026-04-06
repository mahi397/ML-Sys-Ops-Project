# Endpoint Replay Runtime

This folder is a self-contained runtime bundle for replaying hypothetical endpoint traffic:

- start from existing Stage 1 online inference request artifacts
- hit hypothetical Stage 1 and Stage 2 endpoints using mock or HTTP/Flowise modes
- store request/response artifacts in object storage
- store reconstructed segments and generated recap artifacts
- register the resulting runtime state in Postgres

## Expected inputs

This runtime assumes the Stage 1 request artifacts already exist, for example from the online inference workflow or any other synthetic/request-building step:

- `/mnt/block/user-behaviour/online_inference/stage1/{meeting_id}/v{version}/stage1_requests.jsonl`
- `/mnt/block/user-behaviour/online_inference/stage1/{meeting_id}/v{version}/model_utterances.json`

## Files

- `replay_to_hypothetical_endpoints.py`: replays Stage 1 and Stage 2 hypothetical endpoint traffic
- `discover_endpoint_replay_meetings.py`: finds meetings that already have Stage 1 request artifacts
- `feedback_common.py`: shared DB/object-store helpers used by the replay script
- `flowise_stage2_prompt.txt`: prompt template for Flowise Stage 2 mode
- `run_endpoint_replay_batch.sh`: batch runner for one or many meetings
- `endpoint_replay.env.example`: environment template
- `requirements.txt`: minimal Python dependency list

## VM setup

1. Copy this folder to the VM.
2. Copy `endpoint_replay.env.example` to `endpoint_replay.env`.
3. Update values if your DB, bucket, or local roots differ.
4. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Run all discoverable meetings

```bash
bash run_endpoint_replay_batch.sh
```

## Run only selected meetings

```bash
bash run_endpoint_replay_batch.sh \
  jitsi_20260401T201408Z_9f8732d1 \
  jitsi_20260403T180944Z_61459c2e
```

## Local outputs

- `/mnt/block/user-behaviour/inference_responses/stage1/`
- `/mnt/block/user-behaviour/online_inference/stage2/`
- `/mnt/block/user-behaviour/inference_responses/stage2/`
- `/mnt/block/user-behaviour/reconstructed_segments/`
- `/mnt/block/user-behaviour/recaps/generated/`
- `/mnt/block/user-behaviour/logs/endpoint_replay/`

## Object-store outputs

- `production/inference_responses/stage1/{meeting_id}/`
- `production/inference_requests/stage2/{meeting_id}/`
- `production/inference_responses/stage2/{meeting_id}/`
- `production/reconstructed_segments/{meeting_id}/`
- `production/recaps/generated/{meeting_id}/`
