#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENDPOINT_REPLAY_ENV_FILE:-${SCRIPT_DIR}/endpoint_replay.env}"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATABASE_URL="${DATABASE_URL:-postgresql://proj07_user:proj07@127.0.0.1:5432/proj07_sql_db}"
RCLONE_REMOTE="${RCLONE_REMOTE:-rclone_s3}"
BUCKET="${BUCKET:-objstore-proj07}"

ONLINE_STAGE1_ROOT="${ONLINE_STAGE1_ROOT:-/mnt/block/user-behaviour/online_inference/stage1}"
STAGE1_RESPONSE_ROOT="${STAGE1_RESPONSE_ROOT:-/mnt/block/user-behaviour/inference_responses/stage1}"
ONLINE_STAGE2_ROOT="${ONLINE_STAGE2_ROOT:-/mnt/block/user-behaviour/online_inference/stage2}"
STAGE2_RESPONSE_ROOT="${STAGE2_RESPONSE_ROOT:-/mnt/block/user-behaviour/inference_responses/stage2}"
SEGMENTS_ROOT="${SEGMENTS_ROOT:-/mnt/block/user-behaviour/reconstructed_segments}"
RECAP_ROOT="${RECAP_ROOT:-/mnt/block/user-behaviour/recaps/generated}"
LOG_ROOT="${LOG_ROOT:-/mnt/block/user-behaviour/logs/endpoint_replay}"

INPUT_MODE="${INPUT_MODE:-synthetic}"
STAGE1_MODE="${STAGE1_MODE:-mock}"
STAGE2_MODE="${STAGE2_MODE:-mock}"
VERSION="${VERSION:-1}"
MODEL_VERSION="${MODEL_VERSION:-flowise-stage2-v1}"
SYNTHETIC_MEETING_COUNT="${SYNTHETIC_MEETING_COUNT:-3}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-42}"
STAGE1_URL="${STAGE1_URL:-}"
STAGE2_URL="${STAGE2_URL:-}"
FLOWISE_BASE_URL="${FLOWISE_BASE_URL:-}"
FLOWISE_FLOW_ID="${FLOWISE_FLOW_ID:-}"
FLOWISE_API_KEY="${FLOWISE_API_KEY:-}"

export DATABASE_URL
export RCLONE_REMOTE
export BUCKET
export FLOWISE_BASE_URL
export FLOWISE_FLOW_ID
export FLOWISE_API_KEY

mkdir -p \
  "${ONLINE_STAGE1_ROOT}" \
  "${STAGE1_RESPONSE_ROOT}" \
  "${ONLINE_STAGE2_ROOT}" \
  "${STAGE2_RESPONSE_ROOT}" \
  "${SEGMENTS_ROOT}" \
  "${RECAP_ROOT}" \
  "${LOG_ROOT}"

declare -a meeting_ids=()
if [[ $# -gt 0 ]]; then
  for arg in "$@"; do
    meeting_ids+=("${arg}")
  done
fi

if [[ "${INPUT_MODE}" == "synthetic" ]]; then
  declare -a synthetic_args=(
    --output-root "${ONLINE_STAGE1_ROOT}"
    --version "${VERSION}"
    --seed "${SYNTHETIC_SEED}"
    --upload-artifacts
    --rclone-remote "${RCLONE_REMOTE}"
    --bucket "${BUCKET}"
    --log-file "${LOG_ROOT}/synthetic_input_generation.log"
  )
  if [[ ${#meeting_ids[@]} -gt 0 ]]; then
    for meeting_id in "${meeting_ids[@]}"; do
      synthetic_args+=(--meeting-id "${meeting_id}")
    done
  else
    synthetic_args+=(--meeting-count "${SYNTHETIC_MEETING_COUNT}")
  fi

  meeting_ids=()
  while IFS= read -r line; do
    if [[ -n "${line}" ]]; then
      meeting_ids+=("${line}")
    fi
  done < <("${PYTHON_BIN}" "${SCRIPT_DIR}/generate_synthetic_endpoint_inputs.py" "${synthetic_args[@]}")
elif [[ ${#meeting_ids[@]} -eq 0 ]]; then
  while IFS= read -r line; do
    if [[ -n "${line}" ]]; then
      meeting_ids+=("${line}")
    fi
  done < <("${PYTHON_BIN}" "${SCRIPT_DIR}/discover_endpoint_replay_meetings.py" --stage1-root "${ONLINE_STAGE1_ROOT}" --version "${VERSION}")
fi

if [[ ${#meeting_ids[@]} -eq 0 ]]; then
  if [[ "${INPUT_MODE}" == "synthetic" ]]; then
    echo "No synthetic endpoint replay meetings were generated." >&2
  else
    echo "No endpoint replay meetings found. Build Stage 1 request artifacts first or set INPUT_MODE=synthetic." >&2
  fi
  exit 1
fi

declare -a optional_args=()
if [[ -n "${STAGE1_URL}" ]]; then
  optional_args+=(--stage1-url "${STAGE1_URL}")
fi
if [[ -n "${STAGE2_URL}" ]]; then
  optional_args+=(--stage2-url "${STAGE2_URL}")
fi
if [[ -n "${FLOWISE_BASE_URL}" ]]; then
  optional_args+=(--flowise-base-url "${FLOWISE_BASE_URL}")
fi
if [[ -n "${FLOWISE_FLOW_ID}" ]]; then
  optional_args+=(--flowise-flow-id "${FLOWISE_FLOW_ID}")
fi
if [[ -n "${FLOWISE_API_KEY}" ]]; then
  optional_args+=(--flowise-api-key "${FLOWISE_API_KEY}")
fi

for meeting_id in "${meeting_ids[@]}"; do
  echo
  echo "=== Endpoint replay meeting ==="
  echo "Meeting ID: ${meeting_id}"

  declare -a replay_args=(
    --meeting-id "${meeting_id}"
    --version "${VERSION}"
    --stage1-requests-jsonl "${ONLINE_STAGE1_ROOT}/${meeting_id}/v${VERSION}/stage1_requests.jsonl"
    --model-utterances-json "${ONLINE_STAGE1_ROOT}/${meeting_id}/v${VERSION}/model_utterances.json"
    --stage1-mode "${STAGE1_MODE}"
    --stage2-mode "${STAGE2_MODE}"
    --model-version "${MODEL_VERSION}"
    --upload-artifacts
    --stage1-response-root "${STAGE1_RESPONSE_ROOT}"
    --stage2-input-root "${ONLINE_STAGE2_ROOT}"
    --stage2-response-root "${STAGE2_RESPONSE_ROOT}"
    --segments-root "${SEGMENTS_ROOT}"
    --recap-root "${RECAP_ROOT}"
    --rclone-remote "${RCLONE_REMOTE}"
    --bucket "${BUCKET}"
    --log-file "${LOG_ROOT}/${meeting_id}_endpoint_replay.log"
  )
  if [[ "${INPUT_MODE}" == "synthetic" ]]; then
    replay_args+=(--skip-db-registration)
  fi

  "${PYTHON_BIN}" "${SCRIPT_DIR}/replay_to_hypothetical_endpoints.py" \
    "${replay_args[@]}" \
    "${optional_args[@]}"
done

echo
echo "Endpoint replay runtime completed for ${#meeting_ids[@]} meeting(s)."
