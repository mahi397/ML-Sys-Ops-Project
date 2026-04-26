#!/bin/bash
# setup.sh — Training subsystem setup for RTX node (CHI@UC)
# Run once before docker compose up
# Assumes: repo cloned at ~/ML-Sys-Ops-Project, .env filled in

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

REPO_DIR="${REPO_DIR:-${HOME}/ML-Sys-Ops-Project}"
BLOCK_ROOT="${BLOCK_ROOT:-/mnt/block}"
RCLONE_REMOTE="${RCLONE_REMOTE:-chi_tacc}"
OBJSTORE_BUCKET="${OBJSTORE_BUCKET:-objstore-proj07}"
DATASET_VERSION="${DATASET_VERSION:-v2}"
FEEDBACK_VERSION="${FEEDBACK_VERSION:-v1}"

echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  NeuralOps Training — Setup Script${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════${NC}"

# ── 1. Install Docker if not present ────────────────────────────
echo -e "\n${YELLOW}[1/8] Docker...${NC}"
if ! command -v docker &>/dev/null; then
    echo "Docker not found — installing..."
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker ${USER}
    echo -e "${GREEN}Docker installed${NC}"
    echo -e "${YELLOW}Applying docker group to current session...${NC}"
    exec sg docker "$0 $@"
else
    echo "Docker already installed: $(docker --version)"
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "Installing docker compose plugin..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker-compose-plugin
fi
echo -e "${GREEN}Docker Compose OK: $(docker compose version)${NC}"

# ── 2. Install rclone if not present ────────────────────────────
echo -e "\n${YELLOW}[2/8] rclone...${NC}"
if ! command -v rclone &>/dev/null; then
    echo "rclone not found — installing..."
    curl -fsSL https://rclone.org/install.sh | sudo bash
    echo -e "${GREEN}rclone installed: $(rclone --version | head -1)${NC}"
else
    echo "rclone already installed: $(rclone --version | head -1)"
fi

# Check rclone config exists
if [ ! -f "${HOME}/.config/rclone/rclone.conf" ]; then
    echo -e "${RED}rclone config not found at ~/.config/rclone/rclone.conf${NC}"
    echo -e "${YELLOW}Configure the chi_tacc remote before continuing:${NC}"
    echo "  rclone config"
    echo "  name:     chi_tacc"
    echo "  type:     s3"
    echo "  provider: Ceph"
    echo "  endpoint: https://chi.tacc.chameleoncloud.org:7480"
    echo "  access_key_id + secret_access_key from CHI@TACC openrc"
    exit 1
fi

# Verify remote access
echo "Verifying ${RCLONE_REMOTE}:${OBJSTORE_BUCKET}/ access..."
rclone lsd ${RCLONE_REMOTE}:${OBJSTORE_BUCKET}/ >/dev/null 2>&1 || {
    echo -e "${RED}Cannot access ${RCLONE_REMOTE}:${OBJSTORE_BUCKET}/${NC}"
    echo "Check rclone config and credentials"
    exit 1
}
echo -e "${GREEN}rclone remote OK${NC}"

# ── 3. NVIDIA Container Toolkit ──────────────────────────────────
echo -e "\n${YELLOW}[3/8] NVIDIA Container Toolkit...${NC}"
if ! dpkg -l 2>/dev/null | grep -q nvidia-container-toolkit; then
    echo "Installing nvidia-container-toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo -e "${GREEN}NVIDIA toolkit installed${NC}"
else
    echo "NVIDIA toolkit already installed"
fi

docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1 && \
    echo -e "${GREEN}GPU accessible in Docker${NC}" || \
    echo -e "${RED}WARNING: GPU not accessible in Docker${NC}"

# ── 4. Block storage layout ──────────────────────────────────────
echo -e "\n${YELLOW}[4/8] Creating block storage layout...${NC}"
sudo mkdir -p \
    ${BLOCK_ROOT}/postgres_data \
    ${BLOCK_ROOT}/minio_data \
    ${BLOCK_ROOT}/ray-checkpoints \
    ${BLOCK_ROOT}/roberta_stage1/${DATASET_VERSION} \
    ${BLOCK_ROOT}/roberta_stage1_feedback_pool/${FEEDBACK_VERSION}
sudo chown -R ${USER}:${USER} ${BLOCK_ROOT}
echo -e "${GREEN}Block storage ready at ${BLOCK_ROOT}${NC}"

# ── 5. Stage training data from object storage ───────────────────
echo -e "\n${YELLOW}[5/8] Staging training data from object storage...${NC}"

DATASET_LOCAL="${BLOCK_ROOT}/roberta_stage1/${DATASET_VERSION}"
FEEDBACK_LOCAL="${BLOCK_ROOT}/roberta_stage1_feedback_pool/${FEEDBACK_VERSION}"

if ls ${DATASET_LOCAL}/*.jsonl >/dev/null 2>&1; then
    echo "Training data already staged at ${DATASET_LOCAL}"
else
    echo "Downloading roberta_stage1/${DATASET_VERSION}..."
    rclone copy \
        ${RCLONE_REMOTE}:${OBJSTORE_BUCKET}/datasets/roberta_stage1/${DATASET_VERSION}/ \
        ${DATASET_LOCAL}/ \
        --progress
fi

if ls ${FEEDBACK_LOCAL}/*.jsonl >/dev/null 2>&1; then
    echo "Feedback pool already staged at ${FEEDBACK_LOCAL}"
else
    echo "Downloading roberta_stage1_feedback_pool/${FEEDBACK_VERSION}..."
    rclone copy \
        ${RCLONE_REMOTE}:${OBJSTORE_BUCKET}/datasets/roberta_stage1_feedback_pool/${FEEDBACK_VERSION}/ \
        ${FEEDBACK_LOCAL}/ \
        --progress
fi

echo "Staged files:"
ls ${DATASET_LOCAL}/
ls ${FEEDBACK_LOCAL}/

# ── 6. Start postgres and initialize databases ───────────────────
echo -e "\n${YELLOW}[6/8] Starting Postgres and initializing databases...${NC}"

cd ${REPO_DIR}

docker compose up -d postgres
echo "Waiting for postgres to be healthy..."
until docker compose exec postgres pg_isready -U proj07_user >/dev/null 2>&1; do
    sleep 2
done
echo -e "${GREEN}Postgres healthy${NC}"

# Create mlflowdb (MLflow manages its own schema via Alembic)
docker compose exec postgres psql -U proj07_user -d postgres \
    -c "CREATE DATABASE mlflowdb;" 2>/dev/null && \
    echo "mlflowdb created" || echo "mlflowdb already exists"

# Initialize app schema
echo "Running schema migrations..."
for f in $(ls ${REPO_DIR}/data/proj07-db/init_sql/*.sql | sort); do
    echo "  Running $(basename $f)..."
    docker compose exec -T postgres psql -U proj07_user -d proj07_sql_db < $f
done

TABLE_COUNT=$(docker compose exec postgres psql -U proj07_user -d proj07_sql_db \
    -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';" \
    | tr -d ' \n')
echo -e "${GREEN}Schema initialized: ${TABLE_COUNT} tables${NC}"

# ── 7. Seed dataset_versions ─────────────────────────────────────
echo -e "\n${YELLOW}[7/8] Seeding dataset_versions table...${NC}"

docker compose exec postgres psql -U proj07_user -d proj07_sql_db -c "
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
"
echo -e "${GREEN}dataset_versions seeded${NC}"

# ── 8. Bring up training services ────────────────────────────────
echo -e "\n${YELLOW}[8/8] Bringing up training services...${NC}"

docker compose up -d minio minio-create-buckets mlflow ray-head

echo "Waiting for MLflow to be ready..."
for i in {1..30}; do
    if curl -sf http://localhost:5000/health >/dev/null 2>&1; then
        echo -e "${GREEN}MLflow ready${NC}"
        break
    fi
    echo "  Waiting... ($((i*5))s)"
    sleep 5
done

# Restore MLflow model registry (skip if already restored)
echo "Checking if MLflow registry already restored..."
MLFLOW_CONTAINER=$(docker ps --format '{{.Names}}' | grep mlflow | grep -v minio | head -1)

_mlflow_already_restored() {
    # Returns 0 (true) if jitsi-topic-segmenter exists AND has a 'production' alias
    curl -sf "http://localhost:5000/api/2.0/mlflow/registered-models/alias?name=jitsi-topic-segmenter&alias=production" \
        >/dev/null 2>&1
}

if _mlflow_already_restored; then
    echo -e "${GREEN}MLflow registry already has jitsi-topic-segmenter@production — skipping restore${NC}"
else
    echo "Restoring MLflow model registry..."
    if [ -f "${HOME}/restore_mlflow.py" ]; then
        docker cp ${HOME}/restore_mlflow.py ${MLFLOW_CONTAINER}:/restore_mlflow.py
        docker exec ${MLFLOW_CONTAINER} python /restore_mlflow.py

        docker exec ${MLFLOW_CONTAINER} python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
client = mlflow.tracking.MlflowClient()
try:
    client.create_registered_model('jitsi-topic-segmenter',
        description='RoBERTa-base full fine-tune, best sweep params, test_pk=0.213')
except:
    pass

# production — Optuna best (v1 AMI, test_pk=0.213)
try:
    mv1 = client.create_model_version('jitsi-topic-segmenter',
        source='s3://proj07-mlflow-artifacts/1/fdc4b6d0966b4aa9bbc6f95c01b5fcda/artifacts/model',
        description='Optuna trial #10, test_pk=0.213, test_f1=0.232, production model, trained on v1 AMI')
    client.set_registered_model_alias('jitsi-topic-segmenter', 'production', mv1.version)
    print(f'production -> v{mv1.version}')
except Exception as e:
    print(f'production: {e}')

# fallback — roberta-base full finetune (v1 AMI, test_pk=0.228)
try:
    mv2 = client.create_model_version('jitsi-topic-segmenter',
        source='s3://proj07-mlflow-artifacts/1/dbd0cb5d052c42f5bae2e898684be6cc/artifacts/model',
        description='distilroberta-base full fine-tune, test_pk=0.286, best recall=0.444, fallback model, trained on v1 AMI')
    client.set_registered_model_alias('jitsi-topic-segmenter', 'fallback', mv2.version)
    print(f'fallback -> v{mv2.version}')
except Exception as e:
    print(f'fallback: {e}')

print('Registry restore complete')
" && echo -e "${GREEN}Model registry restored — production + fallback aliases set${NC}"
    else
        echo -e "${YELLOW}restore_mlflow.py not found at ${HOME}/${NC}"
        echo "Copy it there and run manually:"
        echo "  docker cp ~/restore_mlflow.py ${MLFLOW_CONTAINER}:/restore_mlflow.py"
        echo "  docker exec ${MLFLOW_CONTAINER} python /restore_mlflow.py"
    fi
fi

# Start retrain-watcher last
docker compose up -d retrain-watcher

# ── Summary ──────────────────────────────────────────────────────
# IP=$(curl -sf http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || \
#      hostname -I | awk '{print $1}')
if [ -f "${REPO_DIR}/.env" ]; then
    IP=$(grep "^FLOATING_IP=" "${REPO_DIR}/.env" | cut -d'=' -f2 | tr -d ' ')
fi
IP="${IP:-$(hostname -I | awk '{print $1}')}"

echo -e "\n${GREEN}══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Training subsystem setup complete!${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
echo ""
echo "  MLflow:        http://${IP}:5000"
echo "  Ray Dashboard: http://${IP}:8265"
echo "  MinIO:         http://${IP}:9001"
echo ""
echo "  Registered models:"
echo "    production -> Optuna best (test_pk=0.213)"
echo "    fallback   -> roberta-base full finetune (test_pk=0.228)"
echo ""
echo "  Dataset in use: roberta_stage1/${DATASET_VERSION} (AMI + synthetic)"
echo "  Feedback pool:  roberta_stage1_feedback_pool/${FEEDBACK_VERSION}"
echo ""
echo "  Note: When Aneesh's pipeline produces a new dataset_versions row,"
echo "  retrain.py will automatically pick up the latest version."
echo ""
echo "  Running services:"
docker compose ps --format "table {{.Name}}\t{{.Status}}" | \
    grep -E "postgres|minio|mlflow|ray|retrain" || docker compose ps
echo ""
echo "  To test retraining trigger:"
echo "    1. Insert a test meeting + 5 user feedback events"
echo "    2. docker compose restart retrain-watcher"
echo "    3. docker compose logs retrain-watcher -f"
echo ""
echo "  To bring up full stack (after Aneesh + Shruti integrate):"
echo "    docker compose up -d"
