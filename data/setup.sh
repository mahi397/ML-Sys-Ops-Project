#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_ENV_EXAMPLE="${DATA_DIR}/.env.example"
COMMON_ENV_FILE="${DATA_DIR}/.env"
PROJ07_DB_DIR="${DATA_DIR}/proj07-db"
PROJ07_RUNTIME_DIR="${DATA_DIR}/proj07-runtime"
RUNTIME_ENV_FILE="${PROJ07_RUNTIME_DIR}/.env"
RUNTIME_COMPOSE_PROJECT="${RUNTIME_COMPOSE_PROJECT:-$(basename "${PROJ07_RUNTIME_DIR}")}"
INITIAL_IMPLEMENTATION_DIR="${DATA_DIR}/initial_implementation"
EXTERNAL_DATA_TRAINING_DIR="${INITIAL_IMPLEMENTATION_DIR}/external_data_training_runtime"
ENDPOINT_REPLAY_DIR="${INITIAL_IMPLEMENTATION_DIR}/endpoint_replay_runtime"
ONLINE_INFERENCE_DIR="${INITIAL_IMPLEMENTATION_DIR}/online_inference_workflow_runtime"
RETRAINING_DATASET_DIR="${INITIAL_IMPLEMENTATION_DIR}/retraining_dataset_runtime"
MOCK_JITSI_DIR="${INITIAL_IMPLEMENTATION_DIR}/mock_jitsi_meet"
INITIAL_SETUP_SCRIPT="${INITIAL_IMPLEMENTATION_DIR}/setup.sh"
VENV_DIR="${VENV_DIR:-${DATA_DIR}/.venv}"
BLOCK_ROOT="${BLOCK_ROOT:-/mnt/block}"
LEGACY_TRANSCRIPT_ROOT="${LEGACY_TRANSCRIPT_ROOT:-${BLOCK_ROOT}/user-behaviour/Transcripts}"
FINAL_RECEIVED_TRANSCRIPT_ROOT="${FINAL_RECEIVED_TRANSCRIPT_ROOT:-${BLOCK_ROOT}/user-behaviour/received_transcripts}"
LEGACY_STAGE1_ROOT="${LEGACY_STAGE1_ROOT:-${BLOCK_ROOT}/user-behaviour/online_inference/stage1}"
LEGACY_STAGE2_ROOT="${LEGACY_STAGE2_ROOT:-${BLOCK_ROOT}/user-behaviour/online_inference/stage2}"
LEGACY_RECAP_ROOT="${LEGACY_RECAP_ROOT:-${BLOCK_ROOT}/user-behaviour/recaps/generated}"
FINAL_STAGE1_ROOT="${FINAL_STAGE1_ROOT:-${BLOCK_ROOT}/user-behaviour/inference_requests/stage1}"
FINAL_STAGE2_ROOT="${FINAL_STAGE2_ROOT:-${BLOCK_ROOT}/user-behaviour/inference_requests/stage2}"
FINAL_USER_SUMMARY_ROOT="${FINAL_USER_SUMMARY_ROOT:-${BLOCK_ROOT}/user-behaviour/user_summary_edits}"
RCLONE_REMOTE="${RCLONE_REMOTE:-rclone_s3}"
OBJECT_BUCKET="${OBJECT_BUCKET:-${BUCKET:-objstore-proj07}}"
AMI_OBJECT_PREFIX="${AMI_OBJECT_PREFIX:-ami_public_manual_1.6.2}"
AMI_ARCHIVE_URL="${AMI_ARCHIVE_URL:-https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip}"
AMI_STAGE_ROOT="${AMI_STAGE_ROOT:-${BLOCK_ROOT}/ami_upload_staging}"
AMI_LOCAL_ROOT="${AMI_LOCAL_ROOT:-${AMI_STAGE_ROOT}/${AMI_OBJECT_PREFIX}}"
AMI_ARCHIVE_NAME="${AMI_ARCHIVE_NAME:-ami_public_manual_1.6.2.zip}"
AMI_ARCHIVE_PATH="${AMI_ARCHIVE_PATH:-${AMI_STAGE_ROOT}/${AMI_ARCHIVE_NAME}}"
BOOTSTRAP_SYNTHETIC_STAGE1_ENABLED="${BOOTSTRAP_SYNTHETIC_STAGE1_ENABLED:-true}"
BOOTSTRAP_SYNTHETIC_STAGE1_VERSION="${BOOTSTRAP_SYNTHETIC_STAGE1_VERSION:-1}"
BOOTSTRAP_SYNTHETIC_STAGE1_MEETING_COUNT="${BOOTSTRAP_SYNTHETIC_STAGE1_MEETING_COUNT:-3}"
BOOTSTRAP_SYNTHETIC_STAGE1_SEED="${BOOTSTRAP_SYNTHETIC_STAGE1_SEED:-42}"
RUNTIME_CONTAINER_NAMES=(
  postgres
  adminer
  jitsi_transcript_receiver
  db_task_worker
  stage1_payload_service
  stage1_forward_service
  stage2_input_service
  stage2_forward_service
  user_summary_materialize_service
  retraining_dataset_service
)

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

function ensure_symlink() {
  local target="$1"
  local link_path="$2"

  if [[ -L "${link_path}" ]]; then
    if [[ "$(readlink "${link_path}")" == "${target}" ]]; then
      return
    fi
    rm -f "${link_path}"
  elif [[ -e "${link_path}" ]]; then
    banner "Leaving existing file in place at ${link_path}; it is not a symlink to ${target}"
    return
  fi

  ln -s "${target}" "${link_path}"
}

function is_truthy() {
  local raw_value="${1:-}"
  case "${raw_value,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

function join_by() {
  local delimiter="$1"
  shift || true

  local item
  local first=1
  for item in "$@"; do
    if [[ ${first} -eq 1 ]]; then
      printf '%s' "${item}"
      first=0
    else
      printf '%s%s' "${delimiter}" "${item}"
    fi
  done
}

function array_contains() {
  local needle="$1"
  shift || true

  local item
  for item in "$@"; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
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

function load_runtime_env() {
  if [[ -f "${COMMON_ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${COMMON_ENV_FILE}"
    set +a
  fi

  POSTGRES_DB="${POSTGRES_DB:-proj07_sql_db}"
  POSTGRES_USER="${POSTGRES_USER:-proj07_user}"
  POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-proj07}"
  POSTGRES_PORT="${POSTGRES_PORT:-5432}"
  POSTGRES_DATA_DIR="${POSTGRES_DATA_DIR:-${BLOCK_ROOT}/postgres-data}"
  ADMINER_PORT="${ADMINER_PORT:-5050}"
  RCLONE_REMOTE="${RCLONE_REMOTE:-rclone_s3}"
  OBJECT_BUCKET="${OBJECT_BUCKET:-${BUCKET:-objstore-proj07}}"
  BUCKET="${BUCKET:-${OBJECT_BUCKET}}"
  BOOTSTRAP_SYNTHETIC_STAGE1_ENABLED="${BOOTSTRAP_SYNTHETIC_STAGE1_ENABLED:-true}"
  BOOTSTRAP_SYNTHETIC_STAGE1_VERSION="${BOOTSTRAP_SYNTHETIC_STAGE1_VERSION:-1}"
  BOOTSTRAP_SYNTHETIC_STAGE1_MEETING_COUNT="${BOOTSTRAP_SYNTHETIC_STAGE1_MEETING_COUNT:-3}"
  BOOTSTRAP_SYNTHETIC_STAGE1_SEED="${BOOTSTRAP_SYNTHETIC_STAGE1_SEED:-42}"
}

function postgres_data_initialized() {
  if [[ -f "${POSTGRES_DATA_DIR}/PG_VERSION" ]]; then
    return 0
  fi

  if [[ -d "${POSTGRES_DATA_DIR}/base" || -d "${POSTGRES_DATA_DIR}/global" ]]; then
    return 0
  fi

  if [[ -d "${POSTGRES_DATA_DIR}" ]] && find "${POSTGRES_DATA_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    return 0
  fi

  return 1
}

function runtime_defined_services() {
  (
    cd "${PROJ07_RUNTIME_DIR}"
    docker compose config --services
  )
}

function runtime_running_services() {
  (
    cd "${PROJ07_RUNTIME_DIR}"
    docker compose ps --status running --services 2>/dev/null || true
  )
}

function runtime_missing_services() {
  local running_services=()
  local service=""

  while IFS= read -r service; do
    if [[ -n "${service}" ]]; then
      running_services+=("${service}")
    fi
  done < <(runtime_running_services)

  while IFS= read -r service; do
    if [[ -z "${service}" ]]; then
      continue
    fi
    if ! array_contains "${service}" "${running_services[@]}"; then
      printf '%s\n' "${service}"
    fi
  done < <(runtime_defined_services)
}

function ensure_no_runtime_name_conflicts() {
  local container_name=""
  local project_label=""
  local status=""

  for container_name in "${RUNTIME_CONTAINER_NAMES[@]}"; do
    if ! docker inspect "${container_name}" >/dev/null 2>&1; then
      continue
    fi

    project_label="$(
      docker inspect -f '{{ index .Config.Labels "com.docker.compose.project" }}' "${container_name}" 2>/dev/null || true
    )"
    if [[ "${project_label}" == "${RUNTIME_COMPOSE_PROJECT}" ]]; then
      continue
    fi

    status="$(
      docker inspect -f '{{ .State.Status }}' "${container_name}" 2>/dev/null || true
    )"
    echo "Container name conflict: ${container_name} already exists (status=${status:-unknown}, compose_project=${project_label:-none}). Remove or rename it before running data/setup.sh." >&2
    exit 1
  done
}

function bootstrap_synthetic_stage1_inputs() {
  local manifest_path="${FINAL_STAGE1_ROOT}/_synthetic_generation/v${BOOTSTRAP_SYNTHETIC_STAGE1_VERSION}.json"

  if ! is_truthy "${BOOTSTRAP_SYNTHETIC_STAGE1_ENABLED}"; then
    banner "Skipping synthetic Stage 1 bootstrap because BOOTSTRAP_SYNTHETIC_STAGE1_ENABLED=${BOOTSTRAP_SYNTHETIC_STAGE1_ENABLED}"
    return
  fi

  if [[ -f "${manifest_path}" ]]; then
    banner "Synthetic Stage 1 bootstrap already present at ${manifest_path}"
    return
  fi

  banner "Generating synthetic Stage 1 bootstrap artifacts under ${FINAL_STAGE1_ROOT}"
  python "${ENDPOINT_REPLAY_DIR}/generate_synthetic_endpoint_inputs.py" \
    --output-root "${FINAL_STAGE1_ROOT}" \
    --version "${BOOTSTRAP_SYNTHETIC_STAGE1_VERSION}" \
    --meeting-count "${BOOTSTRAP_SYNTHETIC_STAGE1_MEETING_COUNT}" \
    --seed "${BOOTSTRAP_SYNTHETIC_STAGE1_SEED}" \
    --upload-artifacts \
    --rclone-remote "${RCLONE_REMOTE}" \
    --bucket "${OBJECT_BUCKET}" \
    --stage1-object-prefix "${STAGE1_OBJECT_PREFIX:-production/inference_requests/stage1}" \
    --log-file "${BLOCK_ROOT}/ingest_logs/synthetic_stage1_bootstrap.log"
}

function wait_for_postgres() {
  local attempt=1
  local max_attempts=30

  banner "Waiting for Postgres to become ready"
  until docker exec postgres pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" >/dev/null 2>&1; do
    if [[ ${attempt} -ge ${max_attempts} ]]; then
      echo "Postgres did not become ready after ${max_attempts} attempts" >&2
      exit 1
    fi
    sleep 2
    attempt=$((attempt + 1))
  done
}

function apply_sql_file() {
  local sql_path="$1"
  banner "Applying SQL migration $(basename "${sql_path}")"
  docker exec -i postgres \
    psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -v ON_ERROR_STOP=1 -f - \
    < "${sql_path}"
}

function apply_runtime_schema_updates() {
  local meetings_exists
  meetings_exists="$(
    docker exec postgres \
      psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -tAc \
      "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'meetings');" \
      | tr -d '[:space:]'
  )"

  if [[ "${meetings_exists}" != "t" ]]; then
    echo "The meetings table does not exist after stack startup. Check Postgres initialization logs." >&2
    exit 1
  fi

  apply_sql_file "${PROJ07_DB_DIR}/init_sql/002_add_user_auth_columns.sql"
  apply_sql_file "${PROJ07_DB_DIR}/init_sql/002_feedback_loop_schema.sql"
  apply_sql_file "${PROJ07_DB_DIR}/init_sql/003_workflow_tasks.sql"
  apply_sql_file "${PROJ07_DB_DIR}/init_sql/004_retrain_audit_logs.sql"
  apply_sql_file "${PROJ07_DB_DIR}/init_sql/005_meeting_validity.sql"
}

function start_runtime_stack() {
  local running_services=()
  local missing_services=()
  local service=""

  ensure_no_runtime_name_conflicts

  if postgres_data_initialized; then
    banner "Detected existing Postgres data at ${POSTGRES_DATA_DIR}; setup will reuse it and apply idempotent schema updates after startup"
  else
    banner "Postgres data directory is empty at ${POSTGRES_DATA_DIR}; first-start schema bootstrap will run when Postgres comes up"
  fi

  while IFS= read -r service; do
    if [[ -n "${service}" ]]; then
      running_services+=("${service}")
    fi
  done < <(runtime_running_services)

  while IFS= read -r service; do
    if [[ -n "${service}" ]]; then
      missing_services+=("${service}")
    fi
  done < <(runtime_missing_services)

  if [[ ${#running_services[@]} -gt 0 ]]; then
    banner "Runtime services already running: $(join_by ', ' "${running_services[@]}")"
  else
    banner "No runtime services are currently running for project ${RUNTIME_COMPOSE_PROJECT}"
  fi

  if [[ ${#missing_services[@]} -eq 0 ]]; then
    banner "All runtime services are already up; skipping docker compose up -d"
  else
    banner "Ensuring the integrated runtime stack is up from ${PROJ07_RUNTIME_DIR}"
    (
      cd "${PROJ07_RUNTIME_DIR}"
      docker compose up -d
    )
  fi

  wait_for_postgres
  apply_runtime_schema_updates
}

require_cmd python3
require_cmd docker
require_cmd rclone
require_cmd curl

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
copy_if_missing "${COMMON_ENV_EXAMPLE}" "${COMMON_ENV_FILE}"
ensure_symlink "../.env" "${RUNTIME_ENV_FILE}"
copy_if_missing "${EXTERNAL_DATA_TRAINING_DIR}/external_data_training.env.example" "${EXTERNAL_DATA_TRAINING_DIR}/external_data_training.env"
copy_if_missing "${ENDPOINT_REPLAY_DIR}/endpoint_replay.env.example" "${ENDPOINT_REPLAY_DIR}/endpoint_replay.env"
copy_if_missing "${ONLINE_INFERENCE_DIR}/online_inference_workflow.env.example" "${ONLINE_INFERENCE_DIR}/online_inference_workflow.env"
copy_if_missing "${RETRAINING_DATASET_DIR}/retraining_dataset.env.example" "${RETRAINING_DATASET_DIR}/retraining_dataset.env"

load_runtime_env

banner "Marking batch scripts executable"
chmod +x "${DATA_DIR}/setup.sh"
chmod +x "${INITIAL_SETUP_SCRIPT}"
chmod +x "${EXTERNAL_DATA_TRAINING_DIR}/run_external_data_training_batch.sh"
chmod +x "${ENDPOINT_REPLAY_DIR}/run_endpoint_replay_batch.sh"
chmod +x "${ONLINE_INFERENCE_DIR}/run_online_inference_workflow_batch.sh"
chmod +x "${RETRAINING_DATASET_DIR}/run_retraining_dataset_batch.sh"

banner "Preparing block-storage layout"
mkdir -p \
  "${BLOCK_ROOT}/postgres-data" \
  "${BLOCK_ROOT}/ingest_logs" \
  "${BLOCK_ROOT}/ingest_logs/retraining_dataset_service" \
  "${BLOCK_ROOT}/staging/current_job/raw" \
  "${BLOCK_ROOT}/staging/current_job/processed" \
  "${BLOCK_ROOT}/staging/feedback_loop" \
  "${BLOCK_ROOT}/roberta_stage1" \
  "${BLOCK_ROOT}/user-behaviour/logs" \
  "${LEGACY_TRANSCRIPT_ROOT}" \
  "${FINAL_RECEIVED_TRANSCRIPT_ROOT}" \
  "${BLOCK_ROOT}/user-behaviour/parsed_transcripts" \
  "${LEGACY_STAGE1_ROOT}" \
  "${LEGACY_STAGE2_ROOT}" \
  "${FINAL_STAGE1_ROOT}" \
  "${FINAL_STAGE2_ROOT}" \
  "${BLOCK_ROOT}/user-behaviour/inference_responses/stage1" \
  "${BLOCK_ROOT}/user-behaviour/inference_responses/stage2" \
  "${BLOCK_ROOT}/user-behaviour/reconstructed_segments" \
  "${LEGACY_RECAP_ROOT}" \
  "${FINAL_USER_SUMMARY_ROOT}"

stage_ami_corpus_to_object_store
bootstrap_synthetic_stage1_inputs
start_runtime_stack

banner "Seeding mock Jitsi transcripts into ${LEGACY_TRANSCRIPT_ROOT} and ${FINAL_RECEIVED_TRANSCRIPT_ROOT}"
for transcript in "${MOCK_JITSI_DIR}"/transcript_*.txt; do
  if [[ -f "${transcript}" ]]; then
    cp -n "${transcript}" "${LEGACY_TRANSCRIPT_ROOT}/"
    cp -n "${transcript}" "${FINAL_RECEIVED_TRANSCRIPT_ROOT}/"
  fi
done

cat <<EOF

Setup complete.

AMI raw corpus uploaded to:
  ${RCLONE_REMOTE}:${OBJECT_BUCKET}/${AMI_OBJECT_PREFIX}/

Next steps:
1. Activate the virtual environment:
   source "${VENV_DIR}/bin/activate"
2. Shared environment file:
   ${COMMON_ENV_FILE}
3. The integrated runtime stack is now running from:
   ${PROJ07_RUNTIME_DIR}
4. Read the integrated runtime guide:
   ${PROJ07_RUNTIME_DIR}/README.md
5. If you want the archived standalone bundle only:
   bash "${INITIAL_SETUP_SCRIPT}"
6. Legacy batch runtimes from the initial implementation remain available under:
   ${INITIAL_IMPLEMENTATION_DIR}
   bash "${EXTERNAL_DATA_TRAINING_DIR}/run_external_data_training_batch.sh"
   bash "${ENDPOINT_REPLAY_DIR}/run_endpoint_replay_batch.sh"
   bash "${ONLINE_INFERENCE_DIR}/run_online_inference_workflow_batch.sh"
   bash "${RETRAINING_DATASET_DIR}/run_retraining_dataset_batch.sh"
EOF
