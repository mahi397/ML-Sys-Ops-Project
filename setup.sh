#!/bin/bash
# setup.sh — Unified setup for the full NeuralOps / Jitsi ML system
# Run once on a fresh RTX node (CHI@UC) before docker compose up
# Assumes: repo cloned at ~/ML-Sys-Ops-Project, root .env filled in
#
# Usage:
#   bash setup.sh              # full setup (skips Jitsi by default)
#   DEPLOY_JITSI=true bash setup.sh   # also deploys Jitsi stack
#   SETUP_MODE=data-jitsi DEPLOY_JITSI=true bash setup.sh
#       # data + Jitsi only; use STAGE1_FORWARD_URL/STAGE2_FORWARD_URL for remote serving

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
ENV_SETUP_MODE="${SETUP_MODE-}"
ENV_DOWNLOAD_ML_MODELS="${DOWNLOAD_ML_MODELS-}"
ENV_START_MLFLOW_SERVICES="${START_MLFLOW_SERVICES-}"
ENV_START_DATA_SERVICES="${START_DATA_SERVICES-}"
ENV_START_SERVING_SERVICES="${START_SERVING_SERVICES-}"
ENV_START_TRAINING_SERVICES="${START_TRAINING_SERVICES-}"
ENV_START_MONITORING_SERVICES="${START_MONITORING_SERVICES-}"
ENV_DEPLOY_JITSI="${DEPLOY_JITSI-}"
ENV_POSTGRES_DATA_DIR="${POSTGRES_DATA_DIR-}"

REPO_DIR="${REPO_DIR:-${HOME}/ML-Sys-Ops-Project}"
BLOCK_ROOT="${BLOCK_ROOT:-/mnt/block}"
RCLONE_REMOTE="${RCLONE_REMOTE:-rclone_s3}"
OBJECT_BUCKET="${OBJECT_BUCKET:-${BUCKET:-${OBJSTORE_BUCKET:-objstore-proj07}}}"
DATASET_VERSION="${DATASET_VERSION:-v2}"
FEEDBACK_VERSION="${FEEDBACK_VERSION:-v1}"
DEPLOY_JITSI="${DEPLOY_JITSI:-false}"
SETUP_CONTINUE_ON_ERROR="${SETUP_CONTINUE_ON_ERROR:-true}"
SETUP_FAILED_STEPS=()
RCLONE_READY=0

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "${GREEN}  $*${NC}"; }
info() { echo -e "${YELLOW}  $*${NC}"; }
err()  { echo -e "${RED}  $*${NC}"; }

is_truthy() {
    local value="${1:-}"
    case "${value,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

default_var() {
    local name="$1" value="$2"
    if [[ -z "${!name+x}" ]]; then
        printf -v "${name}" '%s' "${value}"
    fi
}

record_step_failure() {
    local label="$1" status="${2:-1}"
    SETUP_FAILED_STEPS+=("${label} (exit ${status})")
    err "Step failed but setup will continue: ${label} (exit ${status})"
    if ! is_truthy "${SETUP_CONTINUE_ON_ERROR}"; then
        err "Stopping because SETUP_CONTINUE_ON_ERROR=${SETUP_CONTINUE_ON_ERROR}"
        exit "${status}"
    fi
}

env_value_needs_fill() {
    local value="${1:-}"
    case "${value}" in
        ""|replace_with_*|"<"*">")
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

set_env_kv() {
    local key="$1" value="$2" file="$3" tmp
    tmp="$(mktemp)"
    awk -v key="$key" -v value="$value" '
        BEGIN { done = 0 }
        $0 ~ ("^[[:space:]]*#?[[:space:]]*" key "=") {
            if (!done) {
                print key "=" value
                done = 1
            }
            next
        }
        { print }
        END {
            if (!done) {
                print key "=" value
            }
        }
    ' "$file" > "$tmp"
    mv "$tmp" "$file"
}

read_rclone_config_value() {
    local remote="$1" key="$2" config_file="$3"
    awk -v section="[$remote]" -v key="$key" '
        $0 == section {
            in_section = 1
            next
        }
        /^\[/ {
            if (in_section) {
                exit
            }
            next
        }
        in_section && $0 ~ ("^[[:space:]]*" key "[[:space:]]*=") {
            value = $0
            sub(/^[^=]*=/, "", value)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            print value
            exit
        }
    ' "$config_file"
}

populate_aws_credentials_from_rclone() {
    local config_file="${RCLONE_CONFIG:-${HOME}/.config/rclone/rclone.conf}"
    local access_key secret_key endpoint
    local wrote_any=0

    if [[ ! -f "${config_file}" ]]; then
        return 0
    fi

    access_key="$(read_rclone_config_value "${RCLONE_REMOTE}" "access_key_id" "${config_file}")"
    secret_key="$(read_rclone_config_value "${RCLONE_REMOTE}" "secret_access_key" "${config_file}")"
    endpoint="$(read_rclone_config_value "${RCLONE_REMOTE}" "endpoint" "${config_file}")"

    if env_value_needs_fill "${AWS_ACCESS_KEY_ID:-}" && [[ -n "${access_key}" ]]; then
        export AWS_ACCESS_KEY_ID="${access_key}"
        set_env_kv "AWS_ACCESS_KEY_ID" "${access_key}" "${REPO_DIR}/.env"
        wrote_any=1
    fi

    if env_value_needs_fill "${AWS_SECRET_ACCESS_KEY:-}" && [[ -n "${secret_key}" ]]; then
        export AWS_SECRET_ACCESS_KEY="${secret_key}"
        set_env_kv "AWS_SECRET_ACCESS_KEY" "${secret_key}" "${REPO_DIR}/.env"
        wrote_any=1
    fi

    if env_value_needs_fill "${MLFLOW_S3_ENDPOINT_URL:-}" && [[ -n "${endpoint}" ]]; then
        export MLFLOW_S3_ENDPOINT_URL="${endpoint}"
        set_env_kv "MLFLOW_S3_ENDPOINT_URL" "${endpoint}" "${REPO_DIR}/.env"
        wrote_any=1
    fi

    if [[ "${wrote_any}" -eq 1 ]]; then
        ok "Filled MLflow/S3 SDK credentials from ${RCLONE_REMOTE} in rclone.conf"
    elif env_value_needs_fill "${AWS_ACCESS_KEY_ID:-}" || env_value_needs_fill "${AWS_SECRET_ACCESS_KEY:-}"; then
        info "AWS SDK credentials were not found in rclone remote ${RCLONE_REMOTE}; MLflow may need AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY in root .env"
    fi
}

postgres_psql() {
    local database="$1"
    shift
    docker compose exec -T postgres psql -U "${POSTGRES_USER:-proj07_user}" -d "${database}" "$@"
}

postgres_query_scalar() {
    local database="$1" query="$2"
    postgres_psql "${database}" -qAt -c "${query}" | tr -d '[:space:]'
}

postgres_entrypoint_finished() {
    docker compose exec -T postgres sh -c '[ "$(cat /proc/1/comm 2>/dev/null)" = "postgres" ]' >/dev/null 2>&1
}

wait_for_postgres_sql() {
    local database="${1:-postgres}" max_attempts="${2:-90}" attempt=1

    until postgres_entrypoint_finished && postgres_psql "${database}" -qAt -c "SELECT 1" >/dev/null 2>&1; do
        if [[ "${attempt}" -ge "${max_attempts}" ]]; then
            return 1
        fi
        sleep 2
        attempt=$((attempt + 1))
    done
}

apply_sql_file_with_retry() {
    local sql_file="$1" database="${2:-${POSTGRES_DB:-proj07_sql_db}}" max_attempts="${3:-30}" attempt=1

    until postgres_psql "${database}" -v ON_ERROR_STOP=1 -f - < "${sql_file}"; do
        if [[ "${attempt}" -ge "${max_attempts}" ]]; then
            return 1
        fi
        info "  retrying $(basename "${sql_file}") after Postgres finishes startup..."
        sleep 2
        attempt=$((attempt + 1))
    done
}

echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  NeuralOps Full System — Setup Script${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"

# ── Load .env early so vars are available ────────────────────────────────────
if [[ -f "${REPO_DIR}/.env" ]]; then
    set -a; source "${REPO_DIR}/.env"; set +a
    ok ".env loaded"
else
    err ".env not found at ${REPO_DIR}/.env — copy .env.example and fill it in"
    exit 1
fi

RCLONE_REMOTE="${RCLONE_REMOTE:-rclone_s3}"
OBJECT_BUCKET="${OBJECT_BUCKET:-${BUCKET:-${OBJSTORE_BUCKET:-objstore-proj07}}}"
BUCKET="${BUCKET:-${OBJECT_BUCKET}}"
OBJSTORE_BUCKET="${OBJSTORE_BUCKET:-${OBJECT_BUCKET}}"
export OBJECT_BUCKET BUCKET OBJSTORE_BUCKET

[[ -n "${ENV_SETUP_MODE}" ]] && SETUP_MODE="${ENV_SETUP_MODE}"
[[ -n "${ENV_DOWNLOAD_ML_MODELS}" ]] && DOWNLOAD_ML_MODELS="${ENV_DOWNLOAD_ML_MODELS}"
[[ -n "${ENV_START_MLFLOW_SERVICES}" ]] && START_MLFLOW_SERVICES="${ENV_START_MLFLOW_SERVICES}"
[[ -n "${ENV_START_DATA_SERVICES}" ]] && START_DATA_SERVICES="${ENV_START_DATA_SERVICES}"
[[ -n "${ENV_START_SERVING_SERVICES}" ]] && START_SERVING_SERVICES="${ENV_START_SERVING_SERVICES}"
[[ -n "${ENV_START_TRAINING_SERVICES}" ]] && START_TRAINING_SERVICES="${ENV_START_TRAINING_SERVICES}"
[[ -n "${ENV_START_MONITORING_SERVICES}" ]] && START_MONITORING_SERVICES="${ENV_START_MONITORING_SERVICES}"
[[ -n "${ENV_DEPLOY_JITSI}" ]] && DEPLOY_JITSI="${ENV_DEPLOY_JITSI}"
[[ -n "${ENV_POSTGRES_DATA_DIR}" ]] && POSTGRES_DATA_DIR="${ENV_POSTGRES_DATA_DIR}"

if [[ -z "${POSTGRES_DATA_DIR:-}" ]]; then
    if [[ -e "${BLOCK_ROOT}/postgres-data" ]]; then
        POSTGRES_DATA_DIR="${BLOCK_ROOT}/postgres-data"
    elif [[ -e "${BLOCK_ROOT}/postgres_data" ]]; then
        POSTGRES_DATA_DIR="${BLOCK_ROOT}/postgres_data"
    else
        POSTGRES_DATA_DIR="${BLOCK_ROOT}/postgres-data"
    fi
fi
export POSTGRES_DATA_DIR

SETUP_MODE="${SETUP_MODE:-full}"
case "${SETUP_MODE,,}" in
    full)
        default_var DOWNLOAD_ML_MODELS true
        default_var START_MLFLOW_SERVICES true
        default_var START_DATA_SERVICES true
        default_var START_SERVING_SERVICES true
        default_var START_TRAINING_SERVICES true
        default_var START_MONITORING_SERVICES true
        ;;
    data-jitsi|data_jitsi|data-only|data_only|jitsi-data|jitsi_data)
        default_var DOWNLOAD_ML_MODELS false
        default_var START_MLFLOW_SERVICES false
        default_var START_DATA_SERVICES true
        default_var START_SERVING_SERVICES false
        default_var START_TRAINING_SERVICES false
        default_var START_MONITORING_SERVICES false
        ;;
    *)
        err "Unsupported SETUP_MODE=${SETUP_MODE}. Use full or data-jitsi."
        exit 1
        ;;
esac

if is_truthy "${START_SERVING_SERVICES}" && ! is_truthy "${START_MLFLOW_SERVICES}"; then
    info "START_SERVING_SERVICES=true requires MLflow; enabling START_MLFLOW_SERVICES"
    START_MLFLOW_SERVICES=true
fi

if is_truthy "${START_TRAINING_SERVICES}" && ! is_truthy "${START_SERVING_SERVICES}"; then
    info "START_TRAINING_SERVICES=true requires local serving-api; disabling training services"
    START_TRAINING_SERVICES=false
fi

if is_truthy "${START_MONITORING_SERVICES}" && ! is_truthy "${START_SERVING_SERVICES}"; then
    info "START_MONITORING_SERVICES=true requires local serving-api; disabling monitoring services"
    START_MONITORING_SERVICES=false
fi

ok "Setup mode: ${SETUP_MODE}"
info "Service groups: data=${START_DATA_SERVICES} serving=${START_SERVING_SERVICES} training=${START_TRAINING_SERVICES} monitoring=${START_MONITORING_SERVICES} mlflow=${START_MLFLOW_SERVICES}"
info "Postgres data dir: ${POSTGRES_DATA_DIR}"

if is_truthy "${START_DATA_SERVICES}" && ! is_truthy "${START_SERVING_SERVICES}"; then
    if [[ "${STAGE1_FORWARD_URL:-http://serving-api:8000/segment}" == *"serving-api"* || "${STAGE2_FORWARD_URL:-http://serving-api:8000/summarize}" == *"serving-api"* ]]; then
        info "Local serving is disabled. Set STAGE1_FORWARD_URL and STAGE2_FORWARD_URL to the remote serving VM before forwarding meetings."
    fi
fi

# ── 1. Docker ─────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[1/10] Docker...${NC}"
if ! command -v docker &>/dev/null; then
    info "Docker not found — installing..."
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "${USER}"
    ok "Docker installed"
    info "Applying docker group to current session..."
    exec sg docker "$0 $@"
else
    ok "Docker already installed: $(docker --version)"
fi

if ! docker compose version >/dev/null 2>&1; then
    info "Installing docker compose plugin..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker-compose-plugin
fi
ok "Docker Compose: $(docker compose version)"

# ── 2. NVIDIA Container Toolkit ───────────────────────────────────────────────
echo -e "\n${YELLOW}[2/10] NVIDIA Container Toolkit...${NC}"
if is_truthy "${START_SERVING_SERVICES}" || is_truthy "${START_TRAINING_SERVICES}"; then
    if ! command -v nvidia-ctk &>/dev/null && ! dpkg -s nvidia-container-toolkit &>/dev/null; then
        info "Installing nvidia-container-toolkit..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
            sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
        sudo apt-get update -qq
        sudo apt-get install -y -qq nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
        ok "NVIDIA toolkit installed"
    else
        ok "NVIDIA toolkit already installed"
    fi

    docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1 && \
        ok "GPU accessible in Docker" || \
        err "WARNING: GPU not accessible in Docker — serving will run CPU-only"
else
    info "Skipping NVIDIA toolkit check because local serving/training services are disabled"
fi

# ── 3. rclone ─────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[3/10] rclone...${NC}"
if ! command -v rclone &>/dev/null; then
    info "rclone not found — installing..."
    curl -fsSL https://rclone.org/install.sh | sudo bash
    ok "rclone installed: $(rclone --version | head -1)"
else
    ok "rclone already installed: $(rclone --version | head -1)"
fi

if [[ ! -f "${HOME}/.config/rclone/rclone.conf" ]]; then
    err "rclone config not found at ~/.config/rclone/rclone.conf"
    echo "Configure the rclone_s3 remote before continuing:"
    echo "  rclone config"
    echo "  name: rclone_s3  type: s3  provider: Ceph"
    echo "  endpoint: https://chi.tacc.chameleoncloud.org:7480"
    echo "  access_key_id + secret_access_key from CHI@TACC openrc"
    record_step_failure "rclone config missing" 1
else
    populate_aws_credentials_from_rclone
    info "Verifying ${RCLONE_REMOTE}:${OBJECT_BUCKET}/ ..."
    if rclone lsd "${RCLONE_REMOTE}:${OBJECT_BUCKET}/" >/dev/null 2>&1; then
        RCLONE_READY=1
        ok "rclone remote OK"
    else
        err "Cannot access ${RCLONE_REMOTE}:${OBJECT_BUCKET}/ — setup will skip object-store staging"
        info "Available rclone remotes:"
        rclone listremotes 2>/dev/null || true
        info "Check RCLONE_REMOTE, OBJECT_BUCKET, and ~/.config/rclone/rclone.conf"
        record_step_failure "rclone object-store access check" 1
    fi
fi

# ── 4. Block storage layout ───────────────────────────────────────────────────
echo -e "\n${YELLOW}[4/10] Block storage layout...${NC}"
sudo mkdir -p \
    "${POSTGRES_DATA_DIR}" \
    "${BLOCK_ROOT}/minio_data" \
    "${BLOCK_ROOT}/ray-checkpoints" \
    "${BLOCK_ROOT}/jitsi" \
    "${BLOCK_ROOT}/roberta_stage1/${DATASET_VERSION}" \
    "${BLOCK_ROOT}/roberta_stage1_feedback_pool/${FEEDBACK_VERSION}" \
    "${BLOCK_ROOT}/ingest_logs" \
    "${BLOCK_ROOT}/staging/current_job/raw" \
    "${BLOCK_ROOT}/staging/current_job/processed" \
    "${BLOCK_ROOT}/staging/feedback_loop" \
    "${BLOCK_ROOT}/user-behaviour/received_transcripts" \
    "${BLOCK_ROOT}/user-behaviour/parsed_transcripts" \
    "${BLOCK_ROOT}/user-behaviour/inference_requests/stage1" \
    "${BLOCK_ROOT}/user-behaviour/inference_requests/stage2" \
    "${BLOCK_ROOT}/user-behaviour/inference_responses/stage1" \
    "${BLOCK_ROOT}/user-behaviour/inference_responses/stage2" \
    "${BLOCK_ROOT}/user-behaviour/reconstructed_segments" \
    "${BLOCK_ROOT}/user-behaviour/user_summary_edits"
sudo chown -R "${USER}:${USER}" "${BLOCK_ROOT}"
# Postgres data dir ownership is fixed in step 7 (after stop, before start)
ok "Block storage ready at ${BLOCK_ROOT}"

# ── 5. Stage training datasets ────────────────────────────────────────────────
echo -e "\n${YELLOW}[5/10] Staging training data from object storage...${NC}"

DATASET_LOCAL="${BLOCK_ROOT}/roberta_stage1/${DATASET_VERSION}"
FEEDBACK_LOCAL="${BLOCK_ROOT}/roberta_stage1_feedback_pool/${FEEDBACK_VERSION}"

if [[ "${RCLONE_READY}" -ne 1 ]]; then
    info "Skipping training dataset download because rclone object-store access is unavailable"
else
    if ls "${DATASET_LOCAL}"/*.jsonl >/dev/null 2>&1; then
        ok "Training data already staged at ${DATASET_LOCAL}"
    else
        info "Downloading roberta_stage1/${DATASET_VERSION} ..."
        if ! rclone copy \
            "${RCLONE_REMOTE}:${OBJECT_BUCKET}/datasets/roberta_stage1/${DATASET_VERSION}/" \
            "${DATASET_LOCAL}/" --progress; then
            record_step_failure "download roberta_stage1/${DATASET_VERSION}" 1
        fi
    fi

    if ls "${FEEDBACK_LOCAL}"/*.jsonl >/dev/null 2>&1; then
        ok "Feedback pool already staged at ${FEEDBACK_LOCAL}"
    else
        info "Downloading roberta_stage1_feedback_pool/${FEEDBACK_VERSION} ..."
        if ! rclone copy \
            "${RCLONE_REMOTE}:${OBJECT_BUCKET}/datasets/roberta_stage1_feedback_pool/${FEEDBACK_VERSION}/" \
            "${FEEDBACK_LOCAL}/" --progress; then
            record_step_failure "download roberta_stage1_feedback_pool/${FEEDBACK_VERSION}" 1
        fi
    fi
    ok "Training dataset staging checked"
fi

# ── 6. Download ML models ─────────────────────────────────────────────────────
echo -e "\n${YELLOW}[6/10] ML models (RoBERTa + Mistral)...${NC}"
if is_truthy "${DOWNLOAD_ML_MODELS}"; then
    MODELS_DIR="${REPO_DIR}/serving/models"
    mkdir -p "${MODELS_DIR}"

    python3 -m pip --version &>/dev/null || sudo apt-get install -y -qq python3-pip
    _pip() {
        local sub="$1"; shift
        python3 -m pip "$sub" "$@" 2>/dev/null || \
        python3 -m pip "$sub" --break-system-packages "$@"
    }

    if [[ ! -d "${MODELS_DIR}/roberta-seg" || ! -f "${MODELS_DIR}/roberta-seg/config.json" ]]; then
        info "Downloading RoBERTa base weights (~500MB)..."
        _pip install --quiet transformers torch
        python3 -c "
from transformers import RobertaForSequenceClassification, RobertaTokenizer
m = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
t = RobertaTokenizer.from_pretrained('roberta-base')
m.save_pretrained('${MODELS_DIR}/roberta-seg')
t.save_pretrained('${MODELS_DIR}/roberta-seg')
print('RoBERTa saved')
"
        ok "RoBERTa saved to ${MODELS_DIR}/roberta-seg"
    else
        ok "RoBERTa already present"
    fi

    GGUF="${MODELS_DIR}/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    if [[ ! -f "${GGUF}" ]]; then
        info "Downloading Mistral-7B Q4_K_M (~4.4GB) — this takes a while..."
        _pip install --quiet huggingface-hub
        python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
    filename='mistral-7b-instruct-v0.2.Q4_K_M.gguf',
    local_dir='${MODELS_DIR}',
    local_dir_use_symlinks=False
)
print('Mistral downloaded')
"
        ok "Mistral saved to ${GGUF}"
    else
        ok "Mistral already present"
    fi

    echo "Models:"
    ls -lh "${MODELS_DIR}/"
else
    info "Skipping ML model download because DOWNLOAD_ML_MODELS=${DOWNLOAD_ML_MODELS}"
    info "Use SETUP_MODE=full or DOWNLOAD_ML_MODELS=true on a serving VM."
fi

# ── 7. Postgres init + schema migrations ──────────────────────────────────────
echo -e "\n${YELLOW}[7/10] Postgres + schema migrations...${NC}"

cd "${REPO_DIR}"
# Stop postgres first so it isn't running when we fix data dir ownership
docker compose stop postgres 2>/dev/null || true
sudo chown -R 999:999 "${POSTGRES_DATA_DIR}"
if docker compose up -d --remove-orphans postgres; then
    info "Waiting for postgres SQL service to be ready..."
    if wait_for_postgres_sql "postgres" 90; then
        ok "Postgres SQL ready"
    else
        record_step_failure "postgres SQL readiness" 1
    fi

    if wait_for_postgres_sql "${POSTGRES_DB:-proj07_sql_db}" 90; then
        if [[ "$(postgres_query_scalar "postgres" "SELECT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'mlflowdb');" 2>/dev/null)" == "t" ]]; then
            ok "mlflowdb already exists"
        elif postgres_psql "postgres" -c "CREATE DATABASE mlflowdb;"; then
            ok "mlflowdb created"
        else
            record_step_failure "create mlflowdb" 1
        fi

        # Run schema migrations in order. 001 is bootstrap-only and is also
        # mounted into docker-entrypoint-initdb.d for fresh Postgres volumes.
        INIT_SQL_DIR="${REPO_DIR}/data/proj07-db/init_sql"
        if [[ -d "${INIT_SQL_DIR}" ]]; then
            mapfile -t SQL_FILES < <(find "${INIT_SQL_DIR}" -maxdepth 1 -name '*.sql' | sort)
            if [[ "${#SQL_FILES[@]}" -eq 0 ]]; then
                info "No SQL migrations found in ${INIT_SQL_DIR}"
            else
                BASE_SCHEMA_EXISTS="$(postgres_query_scalar "${POSTGRES_DB:-proj07_sql_db}" "SELECT to_regclass('public.meetings') IS NOT NULL;" 2>/dev/null || echo "f")"
                info "Running schema migrations..."
                for f in "${SQL_FILES[@]}"; do
                    if [[ "${BASE_SCHEMA_EXISTS}" == "t" && "$(basename "$f")" == "001_init_postgres_schema.sql" ]]; then
                        info "  $(basename "$f") already applied by Postgres bootstrap; skipping"
                        continue
                    fi

                    info "  $(basename "$f")..."
                    if ! apply_sql_file_with_retry "$f" "${POSTGRES_DB:-proj07_sql_db}" 30; then
                        record_step_failure "apply SQL migration $(basename "$f")" 1
                    fi
                done
            fi

            if TABLE_COUNT=$(postgres_query_scalar "${POSTGRES_DB:-proj07_sql_db}" "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';" 2>/dev/null); then
                ok "Schema initialized: ${TABLE_COUNT} tables"
            else
                record_step_failure "schema table count check" 1
            fi
        else
            err "SQL init dir not found: ${INIT_SQL_DIR}"
            record_step_failure "SQL init dir missing" 1
        fi
    else
        record_step_failure "application database readiness" 1
    fi
else
    record_step_failure "start postgres container" 1
fi

# ── 8. MinIO + MLflow + full stack ────────────────────────────────────────────
echo -e "\n${YELLOW}[8/10] Starting selected Docker services...${NC}"

cd "${REPO_DIR}"
if is_truthy "${START_MLFLOW_SERVICES}"; then
    if docker compose up -d --remove-orphans minio minio-create-buckets mlflow; then
        info "Waiting for MLflow to be ready..."
        MLFLOW_READY=0
        for i in {1..30}; do
            if curl -sf http://localhost:5000/health >/dev/null 2>&1; then
                MLFLOW_READY=1
                ok "MLflow ready"
                break
            fi
            echo "  Waiting... ($((i*5))s)"
            sleep 5
        done
        if [[ "${MLFLOW_READY}" -ne 1 ]]; then
            record_step_failure "MLflow readiness" 1
        fi
    else
        record_step_failure "start MinIO/MLflow services" 1
    fi
else
    info "Skipping MinIO/MLflow services because START_MLFLOW_SERVICES=${START_MLFLOW_SERVICES}"
fi

# Seed dataset_versions
if postgres_psql "${POSTGRES_DB:-proj07_sql_db}" -c "
INSERT INTO dataset_versions (dataset_name, stage, source_type, object_key)
SELECT 'roberta_stage1', 'stage1', 'ami',
       'datasets/roberta_stage1/${DATASET_VERSION}/'
WHERE NOT EXISTS (
    SELECT 1 FROM dataset_versions WHERE dataset_name = 'roberta_stage1'
);
INSERT INTO dataset_versions (dataset_name, stage, source_type, object_key)
SELECT 'roberta_stage1_feedback_pool', 'stage1', 'production_feedback',
       'datasets/roberta_stage1_feedback_pool/${FEEDBACK_VERSION}/'
WHERE NOT EXISTS (
    SELECT 1 FROM dataset_versions WHERE dataset_name = 'roberta_stage1_feedback_pool'
);
"; then
    ok "dataset_versions seeded"
else
    record_step_failure "seed dataset_versions" 1
fi

# Restore MLflow model registry
# MLFLOW_CONTAINER=$(docker ps --format '{{.Names}}' | grep mlflow | grep -v minio | head -1)
# if [[ -f "${HOME}/restore_mlflow.py" && -n "${MLFLOW_CONTAINER}" ]]; then
#     docker cp "${HOME}/restore_mlflow.py" "${MLFLOW_CONTAINER}:/restore_mlflow.py"
#     docker exec "${MLFLOW_CONTAINER}" python /restore_mlflow.py

#     docker exec "${MLFLOW_CONTAINER}" python -c "
# import mlflow
# mlflow.set_tracking_uri('http://localhost:5000')
# client = mlflow.tracking.MlflowClient()
# try:
#     client.create_registered_model('jitsi-topic-segmenter',
#         description='RoBERTa-base full fine-tune, best sweep params, test_pk=0.213')
# except: pass

# try:
#     mv1 = client.create_model_version('jitsi-topic-segmenter',
#         source='s3://proj07-mlflow-artifacts/1/fdc4b6d0966b4aa9bbc6f95c01b5fcda/artifacts/model',
#         description='Optuna trial #10, test_pk=0.213, test_f1=0.232, production model')
#     client.set_registered_model_alias('jitsi-topic-segmenter', 'production', mv1.version)
#     print(f'production -> v{mv1.version}')
# except Exception as e:
#     print(f'production: {e}')

# try:
#     mv2 = client.create_model_version('jitsi-topic-segmenter',
#         source='s3://proj07-mlflow-artifacts/1/dbd0cb5d052c42f5bae2e898684be6cc/artifacts/model',
#         description='distilroberta-base full fine-tune, test_pk=0.286, fallback model')
#     client.set_registered_model_alias('jitsi-topic-segmenter', 'fallback', mv2.version)
#     print(f'fallback -> v{mv2.version}')
# except Exception as e:
#     print(f'fallback: {e}')
# print('Registry restore complete')
# " && ok "Model registry restored"
# else
#     info "restore_mlflow.py not found at ${HOME}/ — skipping registry restore"
#     echo "  Copy it there and run manually after stack is up"
# fi

DATA_SERVICES=(
    adminer
    jitsi_transcript_receiver
    db_task_worker
    stage1_payload_service
    stage1_forward_service
    stage2_input_service
    stage2_forward_service
    user_summary_materialize_service
    retraining_dataset_service
    production_drift_monitor
)
SERVING_SERVICES=(serving-api)
TRAINING_SERVICES=(retrain-watcher online-eval)
MONITORING_SERVICES=(prometheus grafana alertmanager node-exporter)

SELECTED_SERVICES=()
if is_truthy "${START_DATA_SERVICES}"; then
    SELECTED_SERVICES+=("${DATA_SERVICES[@]}")
fi
if is_truthy "${START_SERVING_SERVICES}"; then
    SELECTED_SERVICES+=("${SERVING_SERVICES[@]}")
fi
if is_truthy "${START_TRAINING_SERVICES}"; then
    SELECTED_SERVICES+=("${TRAINING_SERVICES[@]}")
fi
if is_truthy "${START_MONITORING_SERVICES}"; then
    SELECTED_SERVICES+=("${MONITORING_SERVICES[@]}")
fi

# Bring up selected services. The traffic generator remains profile-gated.
if [[ "${#SELECTED_SERVICES[@]}" -gt 0 ]]; then
    info "Bringing up selected services: ${SELECTED_SERVICES[*]}"
elif is_truthy "${START_MLFLOW_SERVICES}"; then
    info "Only MinIO/MLflow services requested; skipping application service startup"
else
    info "No application service groups selected"
fi

if [[ "${#SELECTED_SERVICES[@]}" -gt 0 ]] && docker compose up -d --remove-orphans "${SELECTED_SERVICES[@]}"; then
    ok "Selected services started"
elif [[ "${#SELECTED_SERVICES[@]}" -gt 0 ]]; then
    record_step_failure "start selected docker compose services" 1
else
    ok "Docker service startup step complete"
fi

# ── 9. Monitoring infra dirs ───────────────────────────────────────────────────
echo -e "\n${YELLOW}[9/10] Verifying monitoring config files...${NC}"
MONITORING_OK=1
for f in \
    "${REPO_DIR}/serving/monitoring/prometheus.yml" \
    "${REPO_DIR}/serving/monitoring/alerts.yml" \
    "${REPO_DIR}/serving/monitoring/alertmanager.yml" \
    "${REPO_DIR}/serving/monitoring/grafana/provisioning/datasources/prometheus.yml" \
    "${REPO_DIR}/serving/monitoring/grafana/provisioning/dashboards/dashboards.yml" \
    "${REPO_DIR}/serving/monitoring/grafana/provisioning/dashboards/jitsi-serving.json"; do
    if [[ ! -f "$f" ]]; then
        err "Missing monitoring file: $f"
        MONITORING_OK=0
    fi
done
[[ ${MONITORING_OK} -eq 1 ]] && ok "Monitoring config OK" || \
    info "Some monitoring files missing — Grafana/Prometheus may not load dashboards"

# ── 10. Jitsi deployment (optional) ──────────────────────────────────────────
echo -e "\n${YELLOW}[10/10] Jitsi deployment...${NC}"
if [[ "${DEPLOY_JITSI}" == "true" ]]; then
    JITSI_DIR="${REPO_DIR}/jitsi-deployment"
    GLOBAL_ENV="${REPO_DIR}/.env"

    # Helper: set or append KEY=VALUE in file
    _set_kv() {
        local key="$1" value="$2" file="$3"
        if ! set_env_kv "${key}" "${value}" "${file}"; then
            record_step_failure "write ${key} to ${file}" 1
        fi
    }

    _IP="${FLOATING_IP:-$(hostname -I | awk '{print $1}')}"
    _HP="${HTTPS_PORT:-${JITSI_PORT:-8443}}"
    _RP="${INGEST_PORT:-9099}"

    # Populate the single global env file used by both root compose and Jitsi.
    _set_kv PUBLIC_URL                              "https://${_IP}:${_HP}"                                          "${GLOBAL_ENV}"
    _set_kv TZ                                      "${TZ:-UTC}"                                                     "${GLOBAL_ENV}"
    _set_kv HTTPS_PORT                              "${_HP}"                                                          "${GLOBAL_ENV}"
    _set_kv HTTP_PORT                               "${HTTP_PORT:-8088}"                                              "${GLOBAL_ENV}"
    _set_kv ENABLE_HTTP_REDIRECT                    "${ENABLE_HTTP_REDIRECT:-1}"                                      "${GLOBAL_ENV}"
    _set_kv JVB_ADVERTISE_IPS                       "${_IP}"                                                          "${GLOBAL_ENV}"
    _set_kv MEETING_PORTAL_DATABASE_URL             "${MEETING_PORTAL_DATABASE_URL:-postgresql://${POSTGRES_USER:-proj07_user}:${POSTGRES_PASSWORD}@${_IP}:5432/${POSTGRES_DB:-proj07_sql_db}}" "${GLOBAL_ENV}"
    _set_kv JITSI_TRANSCRIPT_INGEST_URL             "http://${_IP}:${_RP}/ingest/jitsi-transcript"                   "${GLOBAL_ENV}"
    _set_kv INGEST_TOKEN                            "${INGEST_TOKEN}"                                                 "${GLOBAL_ENV}"
    _set_kv JITSI_TRANSCRIPT_POLL_SECONDS           "${JITSI_TRANSCRIPT_POLL_SECONDS:-5}"                             "${GLOBAL_ENV}"
    _set_kv JITSI_TRANSCRIPT_SETTLE_SECONDS         "${JITSI_TRANSCRIPT_SETTLE_SECONDS:-3}"                           "${GLOBAL_ENV}"
    _set_kv JITSI_TRANSCRIPT_UPLOAD_TIMEOUT         "${JITSI_TRANSCRIPT_UPLOAD_TIMEOUT:-120}"                         "${GLOBAL_ENV}"
    _set_kv MEETING_PORTAL_HTTPS_ONLY               "${MEETING_PORTAL_HTTPS_ONLY:-true}"                              "${GLOBAL_ENV}"
    _set_kv MEETING_PORTAL_TOKEN_TTL_SECONDS        "${MEETING_PORTAL_TOKEN_TTL_SECONDS:-3600}"                       "${GLOBAL_ENV}"
    _set_kv MEETING_PORTAL_RCLONE_REMOTE            "${MEETING_PORTAL_RCLONE_REMOTE:-${RCLONE_REMOTE:-rclone_s3}}"   "${GLOBAL_ENV}"
    _set_kv MEETING_PORTAL_RCLONE_BUCKET            "${MEETING_PORTAL_RCLONE_BUCKET:-${OBJECT_BUCKET:-objstore-proj07}}" "${GLOBAL_ENV}"
    _set_kv MEETING_PORTAL_RCLONE_TIMEOUT_SECONDS   "${MEETING_PORTAL_RCLONE_TIMEOUT_SECONDS:-10}"                    "${GLOBAL_ENV}"
    _set_kv MEETING_PORTAL_STAGE1_RCLONE_FALLBACK_ENABLED "${MEETING_PORTAL_STAGE1_RCLONE_FALLBACK_ENABLED:-true}"   "${GLOBAL_ENV}"
    _set_kv JIGASI_DISABLE_SIP                      "${JIGASI_DISABLE_SIP:-1}"                                        "${GLOBAL_ENV}"

    # Propagate secrets only if already set in root .env (installer generates them otherwise)
    [[ -n "${JWT_APP_SECRET:-}" ]]                && _set_kv JWT_APP_SECRET                "${JWT_APP_SECRET}"                "${GLOBAL_ENV}"
    [[ -n "${MEETING_PORTAL_SESSION_SECRET:-}" ]] && _set_kv MEETING_PORTAL_SESSION_SECRET "${MEETING_PORTAL_SESSION_SECRET}" "${GLOBAL_ENV}"
    [[ -n "${JITSI_HOST_EXTERNAL_KEY:-}" ]]       && _set_kv JITSI_HOST_EXTERNAL_KEY       "${JITSI_HOST_EXTERNAL_KEY}"       "${GLOBAL_ENV}"
    # SIP creds only forwarded if SIP gateway is enabled
    if [[ "${JIGASI_DISABLE_SIP:-1}" == "0" ]]; then
        [[ -n "${JIGASI_SIP_URI:-}" ]]            && _set_kv JIGASI_SIP_URI                "${JIGASI_SIP_URI}"                "${GLOBAL_ENV}"
        [[ -n "${JIGASI_SIP_PASSWORD:-}" ]]       && _set_kv JIGASI_SIP_PASSWORD           "${JIGASI_SIP_PASSWORD}"           "${GLOBAL_ENV}"
        [[ -n "${JIGASI_SIP_SERVER:-}" ]]         && _set_kv JIGASI_SIP_SERVER             "${JIGASI_SIP_SERVER}"             "${GLOBAL_ENV}"
    fi

    ok "Global .env populated for Jitsi"
    info "Running Jitsi installer (downloads upstream images + Vosk model ~1GB)..."
    if sudo bash "${JITSI_DIR}/install-jitsi-vm.sh" --env-file "${GLOBAL_ENV}"; then
        ok "Jitsi deployment complete"
        echo ""
        echo "  Jitsi web:    https://${_IP}:${_HP}"
        echo "  Generated Jitsi secrets are written back to the root .env when possible:"
        echo "    grep -E 'JWT_APP_SECRET|MEETING_PORTAL_SESSION_SECRET|JITSI_HOST_EXTERNAL_KEY' ${GLOBAL_ENV}"
    else
        status=$?
        record_step_failure "Jitsi deployment" "${status}"
    fi
else
    info "Skipping Jitsi deployment (set DEPLOY_JITSI=true to include)"
    echo "  Root .env can stay minimal. setup.sh + the installer will derive the"
    echo "  Jitsi URLs, fill project defaults, and generate secrets as needed."
    echo "  When ready, run: DEPLOY_JITSI=true bash setup.sh"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
IP="${FLOATING_IP:-$(hostname -I | awk '{print $1}')}"

echo -e "\n${GREEN}══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Setup mode:      ${SETUP_MODE}"
echo "  Adminer:         http://${IP}:${ADMINER_PORT:-5050}"
echo "  Ingest endpoint: http://${IP}:${INGEST_PORT:-9099}/ingest/jitsi-transcript"
if [[ "${DEPLOY_JITSI}" == "true" ]]; then
    echo "  Jitsi web:       https://${IP}:${HTTPS_PORT:-${JITSI_PORT:-8443}}"
fi
if is_truthy "${START_SERVING_SERVICES}"; then
    echo "  Ray Serve API:   http://${IP}:8000/health"
    echo "  Ray Dashboard:   http://${IP}:8265  (serving only — training uses standalone Ray)"
else
    echo "  Stage 1 forward: ${STAGE1_FORWARD_URL:-http://serving-api:8000/segment}"
    echo "  Stage 2 forward: ${STAGE2_FORWARD_URL:-http://serving-api:8000/summarize}"
fi
if is_truthy "${START_MLFLOW_SERVICES}"; then
    echo "  MLflow:          http://${IP}:5000"
    echo "  MinIO:           http://${IP}:9001"
fi
if is_truthy "${START_MONITORING_SERVICES}"; then
    echo "  Grafana:         http://${IP}:3000  (admin / ${GRAFANA_PASSWORD:-admin})"
    echo "  Prometheus:      http://${IP}:9090"
    echo "  AlertManager:    http://${IP}:9093"
fi
echo ""
if is_truthy "${START_MLFLOW_SERVICES}"; then
    echo "  Registered models:"
    echo "    @production -> Optuna best (test_pk=0.213)"
    echo "    @fallback   -> distilroberta full finetune (test_pk=0.286)"
    echo ""
fi
if is_truthy "${START_DATA_SERVICES}"; then
    echo "  Training datasets in use:"
    echo "    roberta_stage1/${DATASET_VERSION}"
    echo "    roberta_stage1_feedback_pool/${FEEDBACK_VERSION}"
    echo ""
fi
if is_truthy "${START_TRAINING_SERVICES}"; then
    echo "  To trigger a retrain manually:"
    echo "    docker compose exec retrain-watcher python /app/retrain.py"
    echo ""
fi
if [[ "${DEPLOY_JITSI}" != "true" ]]; then
    echo "  To deploy Jitsi:"
    echo "    Fill in root .env, then: DEPLOY_JITSI=true bash setup.sh"
    echo ""
fi
if [[ ${#SETUP_FAILED_STEPS[@]} -gt 0 ]]; then
    echo "  Setup completed with skipped/failed step(s):"
    for failed_step in "${SETUP_FAILED_STEPS[@]}"; do
        echo "    - ${failed_step}"
    done
    echo ""
fi
echo "  Selected services:"
SUMMARY_SERVICES=(postgres)
if is_truthy "${START_MLFLOW_SERVICES}"; then
    SUMMARY_SERVICES+=(minio minio-create-buckets mlflow)
fi
if [[ "${#SELECTED_SERVICES[@]}" -gt 0 ]]; then
    SUMMARY_SERVICES+=("${SELECTED_SERVICES[@]}")
fi
docker compose ps "${SUMMARY_SERVICES[@]}" --format "table {{.Name}}\t{{.Status}}" 2>/dev/null || docker compose ps "${SUMMARY_SERVICES[@]}"
