#!/usr/bin/env bash
set -euo pipefail

INITIAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$(cd "${INITIAL_ROOT}/.." && pwd)"
COMMON_ENV_EXAMPLE="${DATA_ROOT}/.env.example"
COMMON_ENV_FILE="${DATA_ROOT}/.env"
EXTERNAL_DATA_TRAINING_DIR="${INITIAL_ROOT}/external_data_training_runtime"
ENDPOINT_REPLAY_DIR="${INITIAL_ROOT}/endpoint_replay_runtime"
ONLINE_INFERENCE_DIR="${INITIAL_ROOT}/online_inference_workflow_runtime"
RETRAINING_DATASET_DIR="${INITIAL_ROOT}/retraining_dataset_runtime"
MOCK_JITSI_DIR="${INITIAL_ROOT}/mock_jitsi_meet"
VENV_DIR="${VENV_DIR:-${INITIAL_ROOT}/.venv}"
BLOCK_ROOT="${BLOCK_ROOT:-/mnt/block}"
LEGACY_TRANSCRIPT_ROOT="${LEGACY_TRANSCRIPT_ROOT:-${BLOCK_ROOT}/user-behaviour/Transcripts}"
LEGACY_STAGE1_ROOT="${LEGACY_STAGE1_ROOT:-${BLOCK_ROOT}/user-behaviour/online_inference/stage1}"
LEGACY_STAGE2_ROOT="${LEGACY_STAGE2_ROOT:-${BLOCK_ROOT}/user-behaviour/online_inference/stage2}"
LEGACY_RECAP_ROOT="${LEGACY_RECAP_ROOT:-${BLOCK_ROOT}/user-behaviour/recaps/generated}"
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

function has_cmd() {
  command -v "$1" >/dev/null 2>&1
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
  if has_cmd unzip; then
    unzip -oq "${AMI_ARCHIVE_PATH}" -d "${AMI_LOCAL_ROOT}"
  else
    python3 -m zipfile -e "${AMI_ARCHIVE_PATH}" "${AMI_LOCAL_ROOT}"
  fi

  banner "Uploading AMI raw corpus to ${RCLONE_REMOTE}:${OBJECT_BUCKET}/${AMI_OBJECT_PREFIX}"
  rclone copy "${AMI_LOCAL_ROOT}" "${RCLONE_REMOTE}:${OBJECT_BUCKET}/${AMI_OBJECT_PREFIX}" -P

  cleanup_ami_stage_root
}

require_cmd python3
require_cmd rclone
require_cmd curl

banner "Creating Python virtual environment at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

banner "Installing Python dependencies for the archived runtimes"
python -m pip install --upgrade pip
python -m pip install -r "${DATA_ROOT}/requirements.txt"

banner "Preparing archived runtime environment files"
copy_if_missing "${COMMON_ENV_EXAMPLE}" "${COMMON_ENV_FILE}"
copy_if_missing "${EXTERNAL_DATA_TRAINING_DIR}/external_data_training.env.example" "${EXTERNAL_DATA_TRAINING_DIR}/external_data_training.env"
copy_if_missing "${ENDPOINT_REPLAY_DIR}/endpoint_replay.env.example" "${ENDPOINT_REPLAY_DIR}/endpoint_replay.env"
copy_if_missing "${ONLINE_INFERENCE_DIR}/online_inference_workflow.env.example" "${ONLINE_INFERENCE_DIR}/online_inference_workflow.env"
copy_if_missing "${RETRAINING_DATASET_DIR}/retraining_dataset.env.example" "${RETRAINING_DATASET_DIR}/retraining_dataset.env"

banner "Marking archived batch scripts executable"
chmod +x "${INITIAL_ROOT}/setup.sh"
chmod +x "${EXTERNAL_DATA_TRAINING_DIR}/run_external_data_training_batch.sh"
chmod +x "${ENDPOINT_REPLAY_DIR}/run_endpoint_replay_batch.sh"
chmod +x "${ONLINE_INFERENCE_DIR}/run_online_inference_workflow_batch.sh"
chmod +x "${RETRAINING_DATASET_DIR}/run_retraining_dataset_batch.sh"

banner "Preparing archived block-storage layout"
mkdir -p \
  "${BLOCK_ROOT}/postgres-data" \
  "${BLOCK_ROOT}/ingest_logs" \
  "${BLOCK_ROOT}/staging/current_job/raw" \
  "${BLOCK_ROOT}/staging/current_job/processed" \
  "${BLOCK_ROOT}/staging/feedback_loop" \
  "${BLOCK_ROOT}/roberta_stage1" \
  "${BLOCK_ROOT}/user-behaviour/logs" \
  "${LEGACY_TRANSCRIPT_ROOT}" \
  "${BLOCK_ROOT}/user-behaviour/parsed_transcripts" \
  "${LEGACY_STAGE1_ROOT}" \
  "${LEGACY_STAGE2_ROOT}" \
  "${BLOCK_ROOT}/user-behaviour/inference_responses/stage1" \
  "${BLOCK_ROOT}/user-behaviour/inference_responses/stage2" \
  "${BLOCK_ROOT}/user-behaviour/reconstructed_segments" \
  "${LEGACY_RECAP_ROOT}"

stage_ami_corpus_to_object_store

banner "Seeding mock Jitsi transcripts into ${LEGACY_TRANSCRIPT_ROOT}"
for transcript in "${MOCK_JITSI_DIR}"/transcript_*.txt; do
  if [[ -f "${transcript}" ]]; then
    cp -n "${transcript}" "${LEGACY_TRANSCRIPT_ROOT}/"
  fi
done

cat <<EOF

Initial implementation setup complete.

AMI raw corpus uploaded to:
  ${RCLONE_REMOTE}:${OBJECT_BUCKET}/${AMI_OBJECT_PREFIX}/

Next steps:
1. Activate the archived-runtime virtual environment:
   source "${VENV_DIR}/bin/activate"
2. Shared environment file:
   ${COMMON_ENV_FILE}
3. Start the final integrated stack if you need Postgres plus the current services:
   cd "${DATA_ROOT}/proj07-runtime" && docker compose up -d
4. Run the archived batch bundles as needed:
   bash "${EXTERNAL_DATA_TRAINING_DIR}/run_external_data_training_batch.sh"
   bash "${ENDPOINT_REPLAY_DIR}/run_endpoint_replay_batch.sh"
   bash "${ONLINE_INFERENCE_DIR}/run_online_inference_workflow_batch.sh"
   bash "${RETRAINING_DATASET_DIR}/run_retraining_dataset_batch.sh"
EOF
