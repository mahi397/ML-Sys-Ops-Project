#!/bin/bash
# setup.sh — Unified setup for the full NeuralOps / Jitsi ML system
# Run once on a fresh RTX node (CHI@UC) before docker compose up
# Assumes: repo cloned at ~/ML-Sys-Ops-Project, root .env filled in
#
# Usage:
#   bash setup.sh              # full setup (skips Jitsi by default)
#   DEPLOY_JITSI=true bash setup.sh   # also deploys Jitsi stack

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
REPO_DIR="${REPO_DIR:-${HOME}/ML-Sys-Ops-Project}"
BLOCK_ROOT="${BLOCK_ROOT:-/mnt/block}"
RCLONE_REMOTE="${RCLONE_REMOTE:-rclone_s3}"
OBJSTORE_BUCKET="${OBJSTORE_BUCKET:-objstore-proj07}"
DATASET_VERSION="${DATASET_VERSION:-v1}"
FEEDBACK_VERSION="${FEEDBACK_VERSION:-v1}"
DEPLOY_JITSI="${DEPLOY_JITSI:-true}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "${GREEN}  $*${NC}"; }
info() { echo -e "${YELLOW}  $*${NC}"; }
err()  { echo -e "${RED}  $*${NC}"; }

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
    echo "Configure the chi_tacc remote before continuing:"
    echo "  rclone config"
    echo "  name: chi_tacc  type: s3  provider: Ceph"
    echo "  endpoint: https://chi.tacc.chameleoncloud.org:7480"
    echo "  access_key_id + secret_access_key from CHI@TACC openrc"
    exit 1
fi

info "Verifying ${RCLONE_REMOTE}:${OBJSTORE_BUCKET}/ ..."
rclone lsd "${RCLONE_REMOTE}:${OBJSTORE_BUCKET}/" >/dev/null 2>&1 || {
    err "Cannot access ${RCLONE_REMOTE}:${OBJSTORE_BUCKET}/ — check rclone config"
    exit 1
}
ok "rclone remote OK"

# ── 4. Block storage layout ───────────────────────────────────────────────────
echo -e "\n${YELLOW}[4/10] Block storage layout...${NC}"
sudo mkdir -p \
    "${BLOCK_ROOT}/postgres_data" \
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
# postgres_data ownership is fixed in step 7 (after stop, before start)
ok "Block storage ready at ${BLOCK_ROOT}"

# ── 5. Stage training datasets ────────────────────────────────────────────────
echo -e "\n${YELLOW}[5/10] Staging training data from object storage...${NC}"

DATASET_LOCAL="${BLOCK_ROOT}/roberta_stage1/${DATASET_VERSION}"
FEEDBACK_LOCAL="${BLOCK_ROOT}/roberta_stage1_feedback_pool/${FEEDBACK_VERSION}"

if ls "${DATASET_LOCAL}"/*.jsonl >/dev/null 2>&1; then
    ok "Training data already staged at ${DATASET_LOCAL}"
else
    info "Downloading roberta_stage1/${DATASET_VERSION} ..."
    rclone copy \
        "${RCLONE_REMOTE}:${OBJSTORE_BUCKET}/datasets/roberta_stage1/${DATASET_VERSION}/" \
        "${DATASET_LOCAL}/" --progress
fi

if ls "${FEEDBACK_LOCAL}"/*.jsonl >/dev/null 2>&1; then
    ok "Feedback pool already staged at ${FEEDBACK_LOCAL}"
else
    info "Downloading roberta_stage1_feedback_pool/${FEEDBACK_VERSION} ..."
    rclone copy \
        "${RCLONE_REMOTE}:${OBJSTORE_BUCKET}/datasets/roberta_stage1_feedback_pool/${FEEDBACK_VERSION}/" \
        "${FEEDBACK_LOCAL}/" --progress
fi
ok "Training datasets staged"

# ── 6. Download ML models ─────────────────────────────────────────────────────
echo -e "\n${YELLOW}[6/10] ML models (RoBERTa + Mistral)...${NC}"
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

# ── 7. Postgres init + schema migrations ──────────────────────────────────────
echo -e "\n${YELLOW}[7/10] Postgres + schema migrations...${NC}"

cd "${REPO_DIR}"
# Stop postgres first so it isn't running when we fix data dir ownership
docker compose stop postgres 2>/dev/null || true
sudo chown -R 999:999 "${BLOCK_ROOT}/postgres_data"
docker compose up -d --remove-orphans postgres
info "Waiting for postgres to be healthy..."
until docker compose exec postgres pg_isready -U "${POSTGRES_USER:-proj07_user}" -d "${POSTGRES_DB:-proj07_sql_db}" >/dev/null 2>&1; do
    sleep 2
done
ok "Postgres healthy"

# Create mlflowdb
docker compose exec postgres psql -U "${POSTGRES_USER:-proj07_user}" -d postgres \
    -c "CREATE DATABASE mlflowdb;" 2>/dev/null && \
    ok "mlflowdb created" || ok "mlflowdb already exists"

# Run schema migrations in order
INIT_SQL_DIR="${REPO_DIR}/data/proj07-db/init_sql"
if [[ -d "${INIT_SQL_DIR}" ]]; then
    info "Running schema migrations..."
    for f in $(ls "${INIT_SQL_DIR}"/*.sql | sort); do
        info "  $(basename "$f")..."
        docker compose exec -T postgres psql -U "${POSTGRES_USER:-proj07_user}" \
            -d "${POSTGRES_DB:-proj07_sql_db}" < "$f"
    done
    TABLE_COUNT=$(docker compose exec postgres psql -U "${POSTGRES_USER:-proj07_user}" \
        -d "${POSTGRES_DB:-proj07_sql_db}" \
        -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';" \
        | tr -d ' \n')
    ok "Schema initialized: ${TABLE_COUNT} tables"
else
    err "SQL init dir not found: ${INIT_SQL_DIR}"
fi

# ── 8. MinIO + MLflow + full stack ────────────────────────────────────────────
echo -e "\n${YELLOW}[8/10] Starting MinIO, MLflow, and full stack...${NC}"

cd "${REPO_DIR}"
docker compose up -d --remove-orphans minio minio-create-buckets mlflow

info "Waiting for MLflow to be ready..."
for i in {1..30}; do
    if curl -sf http://localhost:5000/health >/dev/null 2>&1; then
        ok "MLflow ready"
        break
    fi
    echo "  Waiting... ($((i*5))s)"
    sleep 5
done

# Seed dataset_versions
docker compose exec postgres psql -U "${POSTGRES_USER:-proj07_user}" \
    -d "${POSTGRES_DB:-proj07_sql_db}" -c "
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
" && ok "dataset_versions seeded"

# Restore MLflow model registry
MLFLOW_CONTAINER=$(docker ps --format '{{.Names}}' | grep mlflow | grep -v minio | head -1)
# if [[ -f "${HOME}/restore_mlflow.py" && -n "${MLFLOW_CONTAINER}" ]]; then
#     docker cp "${HOME}/restore_mlflow.py" "${MLFLOW_CONTAINER}:/restore_mlflow.py"
#     docker exec "${MLFLOW_CONTAINER}" python /restore_mlflow.py

docker exec "${MLFLOW_CONTAINER}" python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
client = mlflow.tracking.MlflowClient()
try:
    client.create_registered_model('jitsi-topic-segmenter',
        description='RoBERTa-base full fine-tune, best sweep params, test_pk=0.213')
except: pass

try:
    mv1 = client.create_model_version('jitsi-topic-segmenter',
        source='s3://proj07-mlflow-artifacts/1/fdc4b6d0966b4aa9bbc6f95c01b5fcda/artifacts/model',
        description='Optuna trial #10, test_pk=0.213, test_f1=0.232, production model')
    client.set_registered_model_alias('jitsi-topic-segmenter', 'production', mv1.version)
    print(f'production -> v{mv1.version}')
except Exception as e:
    print(f'production: {e}')

try:
    mv2 = client.create_model_version('jitsi-topic-segmenter',
        source='s3://proj07-mlflow-artifacts/1/dbd0cb5d052c42f5bae2e898684be6cc/artifacts/model',
        description='distilroberta-base full fine-tune, test_pk=0.286, fallback model')
    client.set_registered_model_alias('jitsi-topic-segmenter', 'fallback', mv2.version)
    print(f'fallback -> v{mv2.version}')
except Exception as e:
    print(f'fallback: {e}')
print('Registry restore complete')
" && ok "Model registry restored"
# else
#     info "restore_mlflow.py not found at ${HOME}/ — skipping registry restore"
#     echo "  Copy it there and run manually after stack is up"
# fi

# Bring up remaining services (including monitoring profile for online-eval)
info "Bringing up full stack..."
docker compose --profile monitoring up -d --remove-orphans
ok "Full stack started"

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
    JITSI_ENV="${REPO_DIR}/.jitsi.env"
    JITSI_CONFIG_ROOT="${BLOCK_ROOT}/jitsi/config"
    _IP="${FLOATING_IP:-$(hostname -I | awk '{print $1}')}"
    _HP="${HTTPS_PORT:-${JITSI_PORT:-8443}}"
    _RP="${INGEST_PORT:-9099}"

    # Helper: set or append KEY=VALUE in file
    _set_kv() {
        local key="$1" value="$2" file="$3"
        if grep -q "^${key}=" "${file}" 2>/dev/null; then
            sed -i "s|^${key}=.*|${key}=${value}|" "${file}"
        else
            echo "${key}=${value}" >> "${file}"
        fi
    }

    if [[ ! -f "${JITSI_ENV}" ]]; then
        info "First-time Jitsi setup — generating secrets and downloading Vosk model..."

        # Bootstrap .jitsi.env from example
        cp "${REPO_DIR}/.jitsi.env.example" "${JITSI_ENV}"

        # Run installer once for secret generation and env setup.
        # We pass --skip-vosk-download so we can handle the model ourselves below.
        STACK_ENV="${JITSI_DIR}/stack.env"
        [[ ! -f "${STACK_ENV}" ]] && cp "${JITSI_DIR}/stack.env.example" "${STACK_ENV}"

        _set_kv PUBLIC_URL              "https://${_IP}:${_HP}"                                                   "${STACK_ENV}"
        _set_kv HTTPS_PORT              "${_HP}"                                                                   "${STACK_ENV}"
        _set_kv HTTP_PORT               "${HTTP_PORT:-8088}"                                                       "${STACK_ENV}"
        _set_kv ENABLE_HTTP_REDIRECT    "${ENABLE_HTTP_REDIRECT:-1}"                                               "${STACK_ENV}"
        _set_kv JVB_ADVERTISE_IPS       "${_IP}"                                                                   "${STACK_ENV}"
        _set_kv MEETING_PORTAL_DATABASE_URL \
            "postgresql://${POSTGRES_USER:-proj07_user}:${POSTGRES_PASSWORD}@${_IP}:5432/${POSTGRES_DB:-proj07_sql_db}" \
            "${STACK_ENV}"
        _set_kv JITSI_TRANSCRIPT_INGEST_URL "http://${_IP}:${_RP}/ingest/jitsi-transcript" "${STACK_ENV}"
        _set_kv MEETING_PORTAL_RCLONE_REMOTE "${RCLONE_REMOTE:-chi_tacc}"                  "${STACK_ENV}"
        _set_kv MEETING_PORTAL_RCLONE_BUCKET "${OBJSTORE_BUCKET:-objstore-proj07}"          "${STACK_ENV}"
        _set_kv JIGASI_DISABLE_SIP           "1"                                            "${STACK_ENV}"

        sudo bash "${JITSI_DIR}/install-jitsi-vm.sh" \
            --env-file "${STACK_ENV}" \
            --skip-docker-install \
            --skip-vosk-download
        ok "Installer ran — secrets generated"

        # Tear down the installer-managed stack so root compose can own the ports
        INSTALLER_SOURCE_ROOT="${BLOCK_ROOT}/jitsi/jitsi-docker-jitsi-meet"
        if [[ -d "${INSTALLER_SOURCE_ROOT}" ]]; then
            info "Stopping installer-managed Jitsi stack (handing ports to root compose)..."
            (
                cd "${INSTALLER_SOURCE_ROOT}"
                docker compose --project-name jitsi-vm \
                    -f docker-compose.yml -f jigasi.yml -f transcriber.yml \
                    -f jitsi-deployment/compose/vm-services.yml \
                    down 2>/dev/null
            ) || true
            ok "Installer stack stopped"
        fi

        # Extract the generated secrets from installer output into .jitsi.env
        INSTALLER_ENV="${BLOCK_ROOT}/jitsi/jitsi-docker-jitsi-meet/.env"
        if [[ -f "${INSTALLER_ENV}" ]]; then
            for key in JWT_APP_SECRET MEETING_PORTAL_SESSION_SECRET INGEST_TOKEN \
                       JITSI_HOST_EXTERNAL_KEY JICOFO_AUTH_PASSWORD JVB_AUTH_PASSWORD \
                       JIGASI_XMPP_PASSWORD JIGASI_TRANSCRIBER_PASSWORD \
                       JIBRI_RECORDER_PASSWORD JIBRI_XMPP_PASSWORD; do
                val="$(grep "^${key}=" "${INSTALLER_ENV}" | head -1 | cut -d= -f2-)"
                [[ -n "${val}" ]] && _set_kv "${key}" "${val}" "${JITSI_ENV}"
            done
            ok "Generated secrets written to ${JITSI_ENV}"
        fi

        # Populate the rest of .jitsi.env from root .env values
        _set_kv PUBLIC_URL              "https://${_IP}:${_HP}"                                                   "${JITSI_ENV}"
        _set_kv JVB_ADVERTISE_IPS       "${_IP}"                                                                   "${JITSI_ENV}"
        _set_kv HTTPS_PORT              "${_HP}"                                                                   "${JITSI_ENV}"
        _set_kv HTTP_PORT               "${HTTP_PORT:-8088}"                                                       "${JITSI_ENV}"
        _set_kv MEETING_PORTAL_DATABASE_URL \
            "postgresql://${POSTGRES_USER:-proj07_user}:${POSTGRES_PASSWORD}@${_IP}:5432/${POSTGRES_DB:-proj07_sql_db}" \
            "${JITSI_ENV}"
        _set_kv JITSI_TRANSCRIPT_INGEST_URL "http://${_IP}:${_RP}/ingest/jitsi-transcript" "${JITSI_ENV}"
        _set_kv MEETING_PORTAL_RCLONE_REMOTE "${RCLONE_REMOTE:-chi_tacc}"                  "${JITSI_ENV}"
        _set_kv MEETING_PORTAL_RCLONE_BUCKET "${OBJSTORE_BUCKET:-objstore-proj07}"          "${JITSI_ENV}"

        ok ".jitsi.env ready — subsequent restarts use: docker compose --profile jitsi up -d"
    else
        ok ".jitsi.env already exists — skipping secret generation"
        info "To regenerate, delete ${JITSI_ENV} and re-run setup.sh"
    fi

    # Download Vosk model if not already present
    VOSK_MODEL_PATH_VAL="$(grep "^VOSK_MODEL_PATH=" "${JITSI_ENV}" | cut -d= -f2-)"
    VOSK_MODEL_PATH_VAL="${VOSK_MODEL_PATH_VAL:-${JITSI_CONFIG_ROOT}/models/vosk-model-en-us-0.22-lgraph}"
    if [[ ! -d "${VOSK_MODEL_PATH_VAL}" ]] || [[ -z "$(ls -A "${VOSK_MODEL_PATH_VAL}" 2>/dev/null)" ]]; then
        info "Downloading Vosk model (~1GB) to ${VOSK_MODEL_PATH_VAL}..."
        VOSK_URL="https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip"
        VOSK_TMPDIR="$(mktemp -d)"
        curl -fL "${VOSK_URL}" -o "${VOSK_TMPDIR}/model.zip"
        unzip -q "${VOSK_TMPDIR}/model.zip" -d "${VOSK_TMPDIR}"
        EXTRACTED="$(find "${VOSK_TMPDIR}" -mindepth 1 -maxdepth 1 -type d | head -1)"
        sudo mkdir -p "$(dirname "${VOSK_MODEL_PATH_VAL}")"
        sudo mv "${EXTRACTED}" "${VOSK_MODEL_PATH_VAL}"
        rm -rf "${VOSK_TMPDIR}"
        ok "Vosk model saved to ${VOSK_MODEL_PATH_VAL}"
    else
        ok "Vosk model already present"
    fi
    _set_kv VOSK_MODEL_PATH "${VOSK_MODEL_PATH_VAL}" "${JITSI_ENV}"

    # Copy rclone config so the meeting-portal container can reach chi.tacc
    mkdir -p "${JITSI_CONFIG_ROOT}/rclone"
    [[ -f "${HOME}/.config/rclone/rclone.conf" ]] && \
        install -m 600 "${HOME}/.config/rclone/rclone.conf" "${JITSI_CONFIG_ROOT}/rclone/rclone.conf" && \
        ok "rclone config copied to Jitsi config dir"

    # Create required Jitsi config directories (idempotent)
    sudo mkdir -p \
        "${JITSI_CONFIG_ROOT}/web/crontabs" \
        "${JITSI_CONFIG_ROOT}/prosody/config" \
        "${JITSI_CONFIG_ROOT}/prosody/prosody-plugins-custom" \
        "${JITSI_CONFIG_ROOT}/jicofo" \
        "${JITSI_CONFIG_ROOT}/jvb" \
        "${JITSI_CONFIG_ROOT}/jigasi" \
        "${JITSI_CONFIG_ROOT}/transcripts" \
        "${JITSI_CONFIG_ROOT}/meeting-portal-app/room-contexts" \
        "${JITSI_CONFIG_ROOT}/transcript-uploader" \
        "${JITSI_CONFIG_ROOT}/models"
    sudo chown -R "${USER}:${USER}" "${BLOCK_ROOT}/jitsi"

    # Start/restart the full Jitsi stack via root compose
    cd "${REPO_DIR}"
    info "Starting Jitsi stack via root compose (--profile jitsi)..."
    docker compose --profile jitsi up -d --build --remove-orphans
    ok "Jitsi stack started"

    echo ""
    echo "  Jitsi web:  https://${_IP}:${_HP}"
    echo "  To manage:  docker compose --profile jitsi [ps|logs -f|restart|down]"
else
    info "Skipping Jitsi deployment (set DEPLOY_JITSI=true to include)"
    echo "  When ready:  cp .jitsi.env.example .jitsi.env  # fill in values, then:"
    echo "               DEPLOY_JITSI=true bash setup.sh"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
IP="${FLOATING_IP:-$(hostname -I | awk '{print $1}')}"

echo -e "\n${GREEN}══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Ray Serve API:   http://${IP}:8000/health"
echo "  Ray Dashboard:   http://${IP}:8265  (serving only — training uses standalone Ray)"
echo "  MLflow:          http://${IP}:5000"
echo "  MinIO:           http://${IP}:9001"
echo "  Grafana:         http://${IP}:3000  (admin / ${GRAFANA_PASSWORD:-admin})"
echo "  Prometheus:      http://${IP}:9090"
echo "  AlertManager:    http://${IP}:9093"
echo "  Adminer:         http://${IP}:${ADMINER_PORT:-5050}"
echo "  Ingest endpoint: http://${IP}:${INGEST_PORT:-9099}/ingest/jitsi-transcript"
echo ""
echo "  Registered models:"
echo "    @production -> Optuna best (test_pk=0.213)"
echo "    @fallback   -> distilroberta full finetune (test_pk=0.286)"
echo ""
echo "  Training datasets in use:"
echo "    roberta_stage1/${DATASET_VERSION}"
echo "    roberta_stage1_feedback_pool/${FEEDBACK_VERSION}"
echo ""
echo "  To trigger a retrain manually:"
echo "    docker compose --profile retrain run retrain-job"
echo ""
echo "  To deploy Jitsi:"
echo "    Fill in jitsi-deployment/stack.env, then:"
echo "    DEPLOY_JITSI=true bash setup.sh"
echo ""
echo "  Running services:"
docker compose ps --format "table {{.Name}}\t{{.Status}}" 2>/dev/null || docker compose ps
