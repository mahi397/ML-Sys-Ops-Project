#!/bin/bash
# ray.sh — Demonstrates Ray Train resuming from checkpoint after simulated worker failure.

# Key evidence in output:
#   Run 1: "Checkpoint saved after epoch 1" then killed
#   Run 2: "RESTORING from previous run" + "Resumed from checkpoint at epoch 1
#           — continuing from epoch 2"

set -e

REPO=/home/cc/ML-Sys-Ops-Project/train
DATA_DIR=/home/cc/ami_processed
STORAGE=/home/cc/artifacts/ray_checkpoints
MLFLOW_URI=http://129.114.25.90:8000
IMAGE=jitsi-train:latest
LOG1=/tmp/ray_run1.log
LOG2=/tmp/ray_run2.log

DOCKER_BASE="docker run --rm --gpus all \
  --shm-size=10.24gb \
  -v ${REPO}:/app \
  -v ${DATA_DIR}:/data/ami_processed \
  -v ${STORAGE}:/ray_checkpoints \
  -e MLFLOW_TRACKING_URI=${MLFLOW_URI} \
  -e TOKENIZERS_PARALLELISM=false \
  -e GIT_SHA=$(git -C ${REPO} rev-parse HEAD) \
  --entrypoint python \
  ${IMAGE}"

RAY_ARGS="/app/train_ray.py \
  --config /app/configs/roberta_base_full.yaml \
  --data_dir /data/ami_processed \
  --storage_path /ray_checkpoints \
  --num_workers 1"

echo "================================================"
echo " Ray Train Fault Tolerance Demo"
echo "================================================"
echo ""

# Step 1: Clean previous checkpoints
echo "STEP 1: Clearing previous Ray checkpoints..."
sudo rm -rf ${STORAGE}/jitsi-roberta-fault-tolerant
mkdir -p ${STORAGE}
echo "Done."
echo ""

# Step 2: RUN 1 — start training, kill after epoch 1 checkpoint
echo "STEP 2: Starting RUN 1 (will be killed after epoch 1 checkpoint saves)..."
echo ""

${DOCKER_BASE} ${RAY_ARGS} 2>&1 | tee ${LOG1} &
DOCKER_PID=$!
echo "Docker PID: ${DOCKER_PID}"

# Poll for checkpoint file
echo "Waiting for epoch 1 checkpoint..."
for i in $(seq 1 120); do
    sleep 15
    if find ${STORAGE} -name "state.pt" 2>/dev/null | grep -q .; then
        echo ""
        echo ">>> Checkpoint found after epoch 1! Killing container..."
        kill ${DOCKER_PID} 2>/dev/null || true
        wait ${DOCKER_PID} 2>/dev/null || true
        echo ">>> Container killed."
        break
    fi
    echo "  Waiting... (${i} polls, $((i*15))s elapsed)"
done

# Extract the experiment path from Run 1 logs
EXPERIMENT_PATH=$(grep "View detailed results here:" ${LOG1} | tail -1 | awk '{print $NF}')
# Map container path /ray_checkpoints → host path ${STORAGE}
EXPERIMENT_PATH_HOST="${STORAGE}${EXPERIMENT_PATH#/ray_checkpoints}"
# Map to container path for Run 2
EXPERIMENT_PATH_CONTAINER="/ray_checkpoints${EXPERIMENT_PATH#/ray_checkpoints}"

echo ""
echo "================================================"
echo " RUN 1 KILLED after epoch 1"
echo " Experiment path: ${EXPERIMENT_PATH_HOST}"
echo " Checkpoint:"
find ${STORAGE} -name "state.pt" 2>/dev/null
echo "================================================"
echo ""

if [ -z "${EXPERIMENT_PATH}" ]; then
    echo "ERROR: Could not extract experiment path from logs. Check ${LOG1}"
    exit 1
fi

# Step 3: Simulate recovery
echo "STEP 3: Simulating node recovery (10s pause)..."
sleep 10
echo ""

# Step 4: RUN 2 — restore from experiment path
echo "STEP 4: Starting RUN 2 with --restore_path=${EXPERIMENT_PATH_CONTAINER}"
echo "        Expecting: 'Resumed from checkpoint at epoch 1 — continuing from epoch 2'"
echo ""

${DOCKER_BASE} ${RAY_ARGS} --restore_path ${EXPERIMENT_PATH_CONTAINER} 2>&1 | tee ${LOG2}

echo ""
echo "================================================"
echo " DEMO COMPLETE"
echo ""
echo " Run 1 evidence (from ${LOG1}):"
grep -E "(Checkpoint saved after epoch|No checkpoint found|Resumed from)" ${LOG1} || true
echo ""
echo " Run 2 evidence (from ${LOG2}):"
grep -E "(RESTORING|Resumed from|No checkpoint found|Epoch [0-9])" ${LOG2} | head -5 || true
echo ""
echo " Key difference vs plain train.py:"
echo "   plain train.py killed mid-run -> restart from epoch 1 (all GPU time lost)"
echo "   Ray Train killed mid-run      -> resume from epoch 2 (only 1 epoch lost)"
echo "================================================"
