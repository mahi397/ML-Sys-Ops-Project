#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${DATA_DIR}/.venv}"
BLOCK_ROOT="${BLOCK_ROOT:-/mnt/block}"
TRANSCRIPT_ROOT="${TRANSCRIPT_ROOT:-${BLOCK_ROOT}/user-behaviour/Transcripts}"
RCLONE_REMOTE="${RCLONE_REMOTE:-rclone_s3}"
OBJECT_BUCKET="${OBJECT_BUCKET:-objstore-proj07}"
AMI_OBJECT_PREFIX="${AMI_OBJECT_PREFIX:-ami_public_manual_1.6.2}"
AMI_ARCHIVE_URL="${AMI_ARCHIVE_URL:-https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip}"
AMI_STAGE_ROOT="${AMI_STAGE_ROOT:-${BLOCK_ROOT}/ami_upload_staging}"
AMI_LOCAL_ROOT="${AMI_LOCAL_ROOT:-${AMI_STAGE_ROOT}/${AMI_OBJECT_PREFIX}}"
AMI_ARCHIVE_NAME="${AMI_ARCHIVE_NAME:-ami_public_manual_1.6.2.zip}"
AMI_ARCHIVE_PATH="${AMI_ARCHIVE_PATH:-${AMI_STAGE_ROOT}/${AMI_ARCHIVE_NAME}}"

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

function remote_has_file() {
  local remote_dir="$1"
  local filename="$2"
  local listing

  if ! listing="$(rclone lsf --files-only "${remote_dir}" 2>/dev/null)"; then
    return 1
  fi

  printf '%s\n' "${listing}" | grep -qx "${filename}"
}

function remote_dir_has_files() {
  local remote_dir="$1"
  local listing

  if ! listing="$(rclone lsf --files-only "${remote_dir}" 2>/dev/null)"; then
    return 1
  fi

  [[ -n "${listing}" ]]
}

function ami_corpus_uploaded() {
  remote_has_file "${RCLONE_REMOTE}:${OBJECT_BUCKET}/${AMI_OBJECT_PREFIX}/corpusResources" "meetings.xml" &&
    remote_has_file "${RCLONE_REMOTE}:${OBJECT_BUCKET}/${AMI_OBJECT_PREFIX}/corpusResources" "participants.xml" &&
    remote_has_file "${RCLONE_REMOTE}:${OBJECT_BUCKET}/${AMI_OBJECT_PREFIX}/ontologies" "default-topics.xml" &&
    remote_dir_has_files "${RCLONE_REMOTE}:${OBJECT_BUCKET}/${AMI_OBJECT_PREFIX}/words" &&
    remote_dir_has_files "${RCLONE_REMOTE}:${OBJECT_BUCKET}/${AMI_OBJECT_PREFIX}/segments" &&
    remote_dir_has_files "${RCLONE_REMOTE}:${OBJECT_BUCKET}/${AMI_OBJECT_PREFIX}/topics" &&
    remote_dir_has_files "${RCLONE_REMOTE}:${OBJECT_BUCKET}/${AMI_OBJECT_PREFIX}/abstractive"
}

function cleanup_ami_stage_root() {
  if [[ ! -e "${AMI_STAGE_ROOT}" ]]; then
    return
  fi

  if [[ -z "${AMI_STAGE_ROOT}" || "${AMI_STAGE_ROOT}" == "/" || "${AMI_STAGE_ROOT}" == "${BLOCK_ROOT}" ]]; then
    echo "Refusing to remove unsafe AMI staging path: ${AMI_STAGE_ROOT}" >&2
    exit 1
  fi

  banner "Removing local AMI staging from ${AMI_STAGE_ROOT}"
  rm -rf "${AMI_STAGE_ROOT}"
}

function stage_ami_corpus_to_object_store() {
  if ami_corpus_uploaded; then
    banner "AMI raw corpus already present in object storage at ${RCLONE_REMOTE}:${OBJECT_BUCKET}/${AMI_OBJECT_PREFIX}"
    cleanup_ami_stage_root
    return
  fi

  banner "Downloading AMI raw corpus to ${AMI_STAGE_ROOT}"
  mkdir -p "${AMI_STAGE_ROOT}" "${AMI_LOCAL_ROOT}"
  curl -L --fail --retry 3 "${AMI_ARCHIVE_URL}" --output "${AMI_ARCHIVE_PATH}"

  banner "Extracting AMI raw corpus into ${AMI_LOCAL_ROOT}"
  unzip -oq "${AMI_ARCHIVE_PATH}" -d "${AMI_LOCAL_ROOT}"

  banner "Uploading AMI raw corpus to ${RCLONE_REMOTE}:${OBJECT_BUCKET}/${AMI_OBJECT_PREFIX}"
  rclone copy "${AMI_LOCAL_ROOT}" "${RCLONE_REMOTE}:${OBJECT_BUCKET}/${AMI_OBJECT_PREFIX}" -P

  cleanup_ami_stage_root
}

require_cmd python3
require_cmd docker
require_cmd rclone
require_cmd curl
require_cmd unzip

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

stage_ami_corpus_to_object_store

banner "Seeding mock Jitsi transcripts into ${TRANSCRIPT_ROOT}"
for transcript in "${DATA_DIR}"/mock_jitsi_meet/transcript_*.txt; do
  if [[ -f "${transcript}" ]]; then
    cp -n "${transcript}" "${TRANSCRIPT_ROOT}/"
  fi
done

cat <<EOF

Setup complete.

AMI raw corpus uploaded to:
  ${RCLONE_REMOTE}:${OBJECT_BUCKET}/${AMI_OBJECT_PREFIX}/

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
