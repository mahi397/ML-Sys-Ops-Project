#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMMON_ENV_FILE="${DATA_ROOT}/.env"
LEGACY_ENV_FILE="${SCRIPT_DIR}/retraining_dataset.env"
ENV_FILE="${RETRAINING_DATASET_ENV_FILE:-${COMMON_ENV_FILE}}"

if [[ "${ENV_FILE}" == "${COMMON_ENV_FILE}" && ! -f "${ENV_FILE}" && -f "${LEGACY_ENV_FILE}" ]]; then
  ENV_FILE="${LEGACY_ENV_FILE}"
fi

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATABASE_URL="${DATABASE_URL:-postgresql://${POSTGRES_USER:-proj07_user}:${POSTGRES_PASSWORD:-proj07}@127.0.0.1:${POSTGRES_PORT:-5432}/${POSTGRES_DB:-proj07_sql_db}}"
RCLONE_REMOTE="${RCLONE_REMOTE:-rclone_s3}"
BUCKET="${BUCKET:-${OBJECT_BUCKET:-objstore-proj07}}"

LOCAL_TMP_ROOT="${LOCAL_TMP_ROOT:-/mnt/block/staging/feedback_loop}"
LOG_ROOT="${LOG_ROOT:-/mnt/block/user-behaviour/logs/retraining_dataset}"

WINDOW_SIZE="${WINDOW_SIZE:-7}"
TRANSITION_INDEX="${TRANSITION_INDEX:-3}"
MIN_UTTERANCE_CHARS="${MIN_UTTERANCE_CHARS:-1}"
MAX_WORDS_PER_UTTERANCE="${MAX_WORDS_PER_UTTERANCE:-50}"

FEEDBACK_POOL_VERSION="${FEEDBACK_POOL_VERSION:-auto}"
STAGE1_FEEDBACK_POOL_PREFIX="${STAGE1_FEEDBACK_POOL_PREFIX:-datasets/roberta_stage1_feedback_pool}"

DATASET_ROOT="${DATASET_ROOT:-/mnt/block/roberta_stage1}"
BASE_VERSION="${BASE_VERSION:-auto}"
SNAPSHOT_VERSION="${SNAPSHOT_VERSION:-auto}"
FINAL_DATASET_OBJECT_PREFIX="${FINAL_DATASET_OBJECT_PREFIX:-datasets/roberta_stage1}"

export DATABASE_URL
export RCLONE_REMOTE
export BUCKET
export LOCAL_TMP_ROOT
export LOG_ROOT
export WINDOW_SIZE
export TRANSITION_INDEX
export MIN_UTTERANCE_CHARS
export MAX_WORDS_PER_UTTERANCE
export STAGE1_FEEDBACK_POOL_PREFIX

mkdir -p "${LOCAL_TMP_ROOT}" "${LOG_ROOT}"

declare -a meeting_ids=()
if [[ $# -gt 0 ]]; then
  for arg in "$@"; do
    meeting_ids+=("${arg}")
  done
else
  while IFS= read -r line; do
    if [[ -n "${line}" ]]; then
      meeting_ids+=("${line}")
    fi
  done < <("${PYTHON_BIN}" "${SCRIPT_DIR}/discover_retraining_dataset_meetings.py")
fi

if [[ ${#meeting_ids[@]} -eq 0 ]]; then
  echo "No new retraining-dataset meetings found. Nothing to compile."
  exit 0
fi

declare -a meeting_flag_args=()
for meeting_id in "${meeting_ids[@]}"; do
  meeting_flag_args+=(--meeting-id "${meeting_id}")
done

echo
echo "Retraining dataset runtime will compile dataset artifacts from ${#meeting_ids[@]} newly eligible meeting(s)."

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_feedback_pool.py" \
  --version "${FEEDBACK_POOL_VERSION}" \
  "${meeting_flag_args[@]}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_retraining_snapshot.py" \
  --dataset-root "${DATASET_ROOT}" \
  --base-version "${BASE_VERSION}" \
  --feedback-pool-root "${LOCAL_TMP_ROOT}/datasets/roberta_stage1_feedback_pool" \
  --feedback-pool-version "${FEEDBACK_POOL_VERSION}" \
  --snapshot-version "${SNAPSHOT_VERSION}" \
  --object-prefix "${FINAL_DATASET_OBJECT_PREFIX}" \
  --upload-artifacts

echo
echo "Retraining dataset runtime completed for ${#meeting_ids[@]} meeting(s)."
