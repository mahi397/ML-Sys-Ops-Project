#!/bin/bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Jitsi Serving + Monitoring — Full Setup${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════${NC}"

# ── 1. Docker ────────────────────────────────────────────────────
echo -e "\n${YELLOW}[1/6] Docker...${NC}"
if ! command -v docker &> /dev/null; then
    curl -sSL https://get.docker.com/ | sudo sh
    sudo usermod -aG docker $USER
    echo -e "${GREEN}Docker installed. Run: newgrp docker${NC}"
    echo -e "${YELLOW}Then re-run this script.${NC}"
    exit 0
else
    echo "Docker OK"
fi

# ── 2. NVIDIA Container Toolkit ──────────────────────────────────
echo -e "\n${YELLOW}[2/6] NVIDIA Container Toolkit...${NC}"
if ! dpkg -l 2>/dev/null | grep -q nvidia-container-toolkit; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo -e "${GREEN}NVIDIA toolkit installed${NC}"
else
    echo "NVIDIA toolkit OK"
fi

# Verify GPU
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1 && \
    echo -e "${GREEN}GPU in Docker ${NC}" || \
    echo -e "${RED}GPU NOT accessible ${NC}"

# ── 3. Clone repo ────────────────────────────────────────────────
echo -e "\n${YELLOW}[3/6] Repository...${NC}"
cd ~
if [ ! -d "ML-Sys-Ops-Project" ]; then
    git clone https://github.com/mahi397/ML-Sys-Ops-Project.git
fi
cd ML-Sys-Ops-Project/serving
echo "In $(pwd)"

# ── 4. Ensure monitoring/ exists ─────────────────────────────────
echo -e "\n${YELLOW}[4/6] Checking project structure...${NC}"
mkdir -p models
mkdir -p monitoring/grafana/provisioning/datasources
mkdir -p monitoring/grafana/provisioning/dashboards
mkdir -p monitoring/grafana/dashboards

# Check required files exist
 #monitoring/grafana/dashboards/jitsi-serving.json \
for f in ray_serve/serve.py ray_serve/Dockerfile.ray ray_serve/requirements_ray.txt \
         monitoring/prometheus.yml monitoring/alerts.yml monitoring/alertmanager.yml \
         monitoring/grafana/provisioning/datasources/prometheus.yml \
         monitoring/grafana/provisioning/dashboards/dashboards.yml \
         monitoring/grafana/provisioning/dashboards/jitsi-serving.json \
         docker-compose.yml; do
    if [ ! -f "$f" ]; then
        echo -e "${RED}Missing: $f${NC}"
        echo "Make sure you've pushed all monitoring files to the repo"
        exit 1
    fi
done
echo -e "${GREEN}All files present ${NC}"

# ── 5. Download models ───────────────────────────────────────────
echo -e "\n${YELLOW}[5/6] Models...${NC}"

if [ ! -d "models/roberta-seg" ]; then
    echo "Downloading RoBERTa..."
    pip install transformers torch --quiet 2>/dev/null || \
        pip install --break-system-packages transformers torch --quiet 2>/dev/null
    python3 -c "
from transformers import RobertaForSequenceClassification, RobertaTokenizer
m = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
t = RobertaTokenizer.from_pretrained('roberta-base')
m.save_pretrained('models/roberta-seg')
t.save_pretrained('models/roberta-seg')
print('RoBERTa saved')
"
else
    echo "RoBERTa OK"
fi

GGUF="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
if [ ! -f "$GGUF" ]; then
    echo "Downloading Mistral-7B Q4 (~4.4GB)..."
    pip install huggingface-hub --quiet 2>/dev/null || \
        pip install --break-system-packages huggingface-hub --quiet 2>/dev/null
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
    filename='mistral-7b-instruct-v0.2.Q4_K_M.gguf',
    local_dir='models', local_dir_use_symlinks=False
)
print('Mistral downloaded')
"
else
    echo "Mistral OK"
fi

echo "Models:"
ls -lh models/

# ── 6. Build & start ─────────────────────────────────────────────
echo -e "\n${YELLOW}[6/6] Building and starting (first build ~10-15 min)...${NC}"

docker compose build ray-serve
docker compose up -d

echo -e "\n${YELLOW}Waiting for Ray Serve to load models...${NC}"
for i in {1..40}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}Ray Serve healthy ${NC}"
        curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || \
            curl -s http://localhost:8000/health
        break
    fi
    echo "  Waiting... ($((i*5))s)"
    sleep 5
done

# Verify all services
echo -e "\n${YELLOW}Service status:${NC}"
for svc in "Ray Serve:8000/health" "Metrics:8000/metrics" "Ray Dashboard:8265" \
           "Prometheus:9090/-/ready" "Grafana:3000/api/health" \
           "AlertManager:9093/-/ready" "Node Exporter:9100/metrics"; do
    name="${svc%%:*}"
    url="http://localhost:${svc#*:}"
    if curl -sf "$url" > /dev/null 2>&1; then
        echo -e "  ${GREEN} $name${NC}"
    else
        echo -e "  ${RED} $name${NC}"
    fi
done

IP=$(curl -sf http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || hostname -I | awk '{print $1}')

echo -e "\n${GREEN}══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Done!${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
echo ""
echo "  API:          http://${IP}:8000/health"
echo "  Grafana:      http://${IP}:3000  (admin / jitsi2026)"
echo "  Prometheus:   http://${IP}:9090"
echo "  Ray Dashboard: http://${IP}:8265"
echo "  AlertManager: http://${IP}:9093"
echo ""
echo "  Next: run benchmark to populate dashboards:"
echo "    pip install requests numpy"
echo "    python3 benchmark_ray.py --url http://localhost:8000/segment --n 200"
echo ""
echo "  Logs:    docker compose logs -f ray-serve"
echo "  Stop:    docker compose down"
echo "  Restart: docker compose up -d"