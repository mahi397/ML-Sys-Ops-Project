#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${EXTERNAL_DATA_TRAINING_ENV_FILE:-${SCRIPT_DIR}/external_data_training.env}"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
RCLONE_REMOTE="${RCLONE_REMOTE:-rclone_s3}"
OBJECT_BUCKET="${OBJECT_BUCKET:-objstore-proj07}"
AMI_OBJECT_PREFIX="${AMI_OBJECT_PREFIX:-ami_public_manual_1.6.2}"

DB_CONTAINER="${DB_CONTAINER:-postgres}"
DB_USER="${DB_USER:-proj07_user}"
DB_NAME="${DB_NAME:-proj07_sql_db}"

BLOCK_ROOT="${BLOCK_ROOT:-/mnt/block}"
RAW_ROOT="${RAW_ROOT:-${BLOCK_ROOT}/staging/current_job/raw}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${BLOCK_ROOT}/staging/current_job/processed}"
INGEST_LOG_ROOT="${INGEST_LOG_ROOT:-${BLOCK_ROOT}/ingest_logs}"

STAGE1_ROOT="${STAGE1_ROOT:-${BLOCK_ROOT}/roberta_stage1}"
STAGE1_V1_ROOT="${STAGE1_V1_ROOT:-${STAGE1_ROOT}/v1}"
STAGE1_V2_ROOT="${STAGE1_V2_ROOT:-${STAGE1_ROOT}/v2}"

DATASET_VERSION="${DATASET_VERSION:-1}"
BASE_SCHEMA_SQL="${BASE_SCHEMA_SQL:-${SCRIPT_DIR}/001_init_postgres_schema.sql}"
DEFAULT_AMI_MEETINGS="${DEFAULT_AMI_MEETINGS:-ES2002a ES2002b ES2003a ES2004a ES2005a ES2006a ES2007a ES2008a}"

if [[ "${BASE_SCHEMA_SQL}" != /* ]]; then
  BASE_SCHEMA_SQL="${SCRIPT_DIR}/${BASE_SCHEMA_SQL#./}"
fi

function require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

function ensure_dir() {
  mkdir -p "$1"
}

function meeting_csv() {
  local first=1
  for meeting in "$@"; do
    if [[ ${first} -eq 1 ]]; then
      printf '%s' "${meeting}"
      first=0
    else
      printf ',%s' "${meeting}"
    fi
  done
}

function psql_scalar() {
  local sql="$1"
  docker exec "${DB_CONTAINER}" \
    psql -U "${DB_USER}" -d "${DB_NAME}" -tAc "${sql}" | tr -d '[:space:]'
}

function ensure_layout() {
  ensure_dir "${RAW_ROOT}"
  ensure_dir "${PROCESSED_ROOT}"
  ensure_dir "${INGEST_LOG_ROOT}"
  ensure_dir "${STAGE1_V1_ROOT}"
  ensure_dir "${STAGE1_V2_ROOT}"
}

function ensure_base_schema() {
  if [[ ! -f "${BASE_SCHEMA_SQL}" ]]; then
    echo "Base schema SQL not found: ${BASE_SCHEMA_SQL}" >&2
    exit 1
  fi

  local meetings_exists
  meetings_exists="$(psql_scalar "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'meetings');" || true)"
  if [[ "${meetings_exists}" != "t" ]]; then
    echo "Applying base schema from ${BASE_SCHEMA_SQL}"
    docker exec -i "${DB_CONTAINER}" \
      psql -U "${DB_USER}" -d "${DB_NAME}" -v ON_ERROR_STOP=1 -f - \
      < "${BASE_SCHEMA_SQL}"
  fi

  local version_col
  version_col="$(psql_scalar "SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'meetings' AND column_name = 'dataset_version');" || true)"
  local split_col
  split_col="$(psql_scalar "SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'meetings' AND column_name = 'dataset_split');" || true)"
  if [[ "${version_col}" != "t" || "${split_col}" != "t" ]]; then
    echo "The current meetings table is missing dataset_version/dataset_split. Use the updated 001_init_postgres_schema.sql on a fresh database before running this runtime." >&2
    exit 1
  fi
}

function ensure_ami_shared_cache() {
  ensure_dir "${RAW_ROOT}/corpusResources"
  ensure_dir "${RAW_ROOT}/ontologies"

  if [[ ! -f "${RAW_ROOT}/corpusResources/meetings.xml" ]]; then
    rclone copyto \
      "${RCLONE_REMOTE}:${OBJECT_BUCKET}/${AMI_OBJECT_PREFIX}/corpusResources/meetings.xml" \
      "${RAW_ROOT}/corpusResources/meetings.xml" -P
  fi

  if [[ ! -f "${RAW_ROOT}/corpusResources/participants.xml" ]]; then
    rclone copyto \
      "${RCLONE_REMOTE}:${OBJECT_BUCKET}/${AMI_OBJECT_PREFIX}/corpusResources/participants.xml" \
      "${RAW_ROOT}/corpusResources/participants.xml" -P
  fi

  if [[ ! -f "${RAW_ROOT}/ontologies/default-topics.xml" ]]; then
    rclone copyto \
      "${RCLONE_REMOTE}:${OBJECT_BUCKET}/${AMI_OBJECT_PREFIX}/ontologies/default-topics.xml" \
      "${RAW_ROOT}/ontologies/default-topics.xml" -P
  fi
}

require_cmd "${PYTHON_BIN}"
require_cmd docker
require_cmd rclone

ensure_layout
ensure_base_schema
ensure_ami_shared_cache

declare -a meetings
if [[ $# -gt 0 ]]; then
  for arg in "$@"; do
    meetings+=("${arg}")
  done
else
  # shellcheck disable=SC2206
  meetings=(${DEFAULT_AMI_MEETINGS})
fi

meeting_list="$(meeting_csv "${meetings[@]}")"

echo
echo "External data training runtime: AMI data -> object storage -> canonical Postgres -> Stage 1 datasets"
echo "Meetings: ${meetings[*]}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_ingest.py" \
  --meeting "${meetings[@]}" \
  --rclone-remote "${RCLONE_REMOTE}" \
  --bucket "${OBJECT_BUCKET}" \
  --prefix "${AMI_OBJECT_PREFIX}" \
  --raw-root "${RAW_ROOT}" \
  --processed-root "${PROCESSED_ROOT}" \
  --scripts-dir "${SCRIPT_DIR}" \
  --pg-container "${DB_CONTAINER}" \
  --db-user "${DB_USER}" \
  --db-name "${DB_NAME}" \
  --log-file "${INGEST_LOG_ROOT}/external_data_training_run_ingest.log"

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_st1_db.py" \
  --meeting-ids "${meeting_list}" \
  --pg-container "${DB_CONTAINER}" \
  --db-user "${DB_USER}" \
  --db-name "${DB_NAME}" \
  --dataset-root "${STAGE1_V1_ROOT}" \
  --dataset-version "${DATASET_VERSION}" \
  --log-file "${INGEST_LOG_ROOT}/external_data_training_build_stage1_dataset.log" \
  --rclone-remote "${RCLONE_REMOTE}" \
  --object-bucket "${OBJECT_BUCKET}" \
  --object-prefix "datasets/roberta_stage1/v1"

"${PYTHON_BIN}" "${SCRIPT_DIR}/augment_stage1_train.py" \
  --skip-download-input \
  --input-root "${STAGE1_V1_ROOT}" \
  --output-root "${STAGE1_V2_ROOT}" \
  --log-file "${INGEST_LOG_ROOT}/external_data_training_augment_stage1.log" \
  --rclone-remote "${RCLONE_REMOTE}" \
  --object-bucket "${OBJECT_BUCKET}" \
  --input-object-prefix "datasets/roberta_stage1/v1" \
  --object-prefix "datasets/roberta_stage1/v2"

echo
echo "External data training runtime complete."
echo "Local outputs:"
echo "  processed AMI artifacts: ${PROCESSED_ROOT}"
echo "  Stage 1 v1 dataset: ${STAGE1_V1_ROOT}"
echo "  Stage 1 v2 augmented dataset: ${STAGE1_V2_ROOT}"
echo "Object-store outputs:"
echo "  ${RCLONE_REMOTE}:${OBJECT_BUCKET}/processed/ami/v1/"
echo "  ${RCLONE_REMOTE}:${OBJECT_BUCKET}/datasets/roberta_stage1/v1/"
echo "  ${RCLONE_REMOTE}:${OBJECT_BUCKET}/datasets/roberta_stage1/v2/"
