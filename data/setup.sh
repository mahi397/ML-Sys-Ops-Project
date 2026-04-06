#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${DATA_DIR}/.venv}"
BLOCK_ROOT="${BLOCK_ROOT:-/mnt/block}"
TRANSCRIPT_ROOT="${TRANSCRIPT_ROOT:-${BLOCK_ROOT}/user-behaviour/Transcripts}"

function banner() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

function require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

function copy_if_missing() {
  local src="$1"
  local dst="$2"
  if [[ -f "${src}" && ! -f "${dst}" ]]; then
    cp "${src}" "${dst}"
  fi
}

require_cmd python3
require_cmd docker
require_cmd rclone

if docker compose version >/dev/null 2>&1; then
  :
else
  echo "Missing required command: docker compose" >&2
  exit 1
fi

banner "Creating Python virtual environment at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

banner "Installing Python dependencies"
python -m pip install --upgrade pip
python -m pip install -r "${DATA_DIR}/requirements.txt"

banner "Preparing environment files"
copy_if_missing "${DATA_DIR}/proj07-db/.env.example" "${DATA_DIR}/proj07-db/.env"
copy_if_missing "${DATA_DIR}/external_data_training_runtime/external_data_training.env.example" "${DATA_DIR}/external_data_training_runtime/external_data_training.env"
copy_if_missing "${DATA_DIR}/endpoint_replay_runtime/endpoint_replay.env.example" "${DATA_DIR}/endpoint_replay_runtime/endpoint_replay.env"
copy_if_missing "${DATA_DIR}/online_inference_workflow_runtime/online_inference_workflow.env.example" "${DATA_DIR}/online_inference_workflow_runtime/online_inference_workflow.env"
copy_if_missing "${DATA_DIR}/retraining_dataset_runtime/retraining_dataset.env.example" "${DATA_DIR}/retraining_dataset_runtime/retraining_dataset.env"

banner "Marking batch scripts executable"
chmod +x "${DATA_DIR}/setup.sh"
chmod +x "${DATA_DIR}/external_data_training_runtime/run_external_data_training_batch.sh"
chmod +x "${DATA_DIR}/endpoint_replay_runtime/run_endpoint_replay_batch.sh"
chmod +x "${DATA_DIR}/online_inference_workflow_runtime/run_online_inference_workflow_batch.sh"
chmod +x "${DATA_DIR}/retraining_dataset_runtime/run_retraining_dataset_batch.sh"

banner "Preparing block-storage layout"
mkdir -p \
  "${BLOCK_ROOT}/postgres-data" \
  "${BLOCK_ROOT}/ingest_logs" \
  "${BLOCK_ROOT}/staging/current_job/raw" \
  "${BLOCK_ROOT}/staging/current_job/processed" \
  "${BLOCK_ROOT}/staging/feedback_loop" \
  "${BLOCK_ROOT}/roberta_stage1" \
  "${BLOCK_ROOT}/user-behaviour/logs" \
  "${TRANSCRIPT_ROOT}" \
  "${BLOCK_ROOT}/user-behaviour/parsed_transcripts" \
  "${BLOCK_ROOT}/user-behaviour/online_inference/stage1" \
  "${BLOCK_ROOT}/user-behaviour/online_inference/stage2" \
  "${BLOCK_ROOT}/user-behaviour/inference_responses/stage1" \
  "${BLOCK_ROOT}/user-behaviour/inference_responses/stage2" \
  "${BLOCK_ROOT}/user-behaviour/reconstructed_segments" \
  "${BLOCK_ROOT}/user-behaviour/recaps/generated"

banner "Seeding mock Jitsi transcripts into ${TRANSCRIPT_ROOT}"
for transcript in "${DATA_DIR}"/mock_jitsi_meet/transcript_*.txt; do
  if [[ -f "${transcript}" ]]; then
    cp -n "${transcript}" "${TRANSCRIPT_ROOT}/"
  fi
done

cat <<EOF

Setup complete.

Next steps:
1. Activate the virtual environment:
   source "${VENV_DIR}/bin/activate"
2. Start Postgres and Adminer:
   cd "${DATA_DIR}/proj07-db" && docker compose up -d
3. Run the runtimes in order:
   bash "${DATA_DIR}/external_data_training_runtime/run_external_data_training_batch.sh"
   bash "${DATA_DIR}/endpoint_replay_runtime/run_endpoint_replay_batch.sh"
   bash "${DATA_DIR}/online_inference_workflow_runtime/run_online_inference_workflow_batch.sh"
   bash "${DATA_DIR}/retraining_dataset_runtime/run_retraining_dataset_batch.sh"
EOF
