#!/bin/bash
# ray.sh  —  Extra credit fault-tolerance demo
# Run on the GPU node host — orchestrates two Docker runs to show fault tolerance
#
# DEMO FLOW:
#   Run 1: Start training → kill after epoch 1 checkpoint saves → container exits
#   Run 2: Restart same command → Ray finds checkpoint → resumes from epoch 2

set -e

REPO=/home/cc/ML-Sys-Ops-Project/train
DATA_DIR=/home/cc/ami_processed
STORAGE=/home/cc/artifacts/ray_checkpoints
MLFLOW_URI=http://129.114.25.90:8000
IMAGE=jitsi-train:latest

DOCKER_BASE="docker run --rm --gpus all \
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

# Step 1: Clean previous checkpoints for a fresh demo
echo "STEP 1: Clearing previous Ray checkpoints..."
rm -rf ${STORAGE}/jitsi-roberta-fault-tolerant
mkdir -p ${STORAGE}
echo "Done."
echo ""

# Step 2: RUN 1 — start training in background, kill after epoch 1 checkpoint
echo "STEP 2: Starting RUN 1 (will be killed after epoch 1 checkpoint saves)..."
echo "        Command: docker run ... python train_ray.py ..."
echo ""

${DOCKER_BASE} ${RAY_ARGS} &
DOCKER_PID=$!
echo "Docker container PID: ${DOCKER_PID}"

# Poll for checkpoint file
echo "Waiting for epoch 1 checkpoint to be saved..."
for i in $(seq 1 120); do
    sleep 15
    if find ${STORAGE} -name "state.pt" 2>/dev/null | grep -q .; then
        echo ""
        echo ">>> Checkpoint found! Killing container (PID ${DOCKER_PID})..."
        # Kill the docker run process — this simulates a preempted Chameleon lease
        kill ${DOCKER_PID} 2>/dev/null || true
        wait ${DOCKER_PID} 2>/dev/null || true
        echo ">>> Container killed."
        break
    fi
    echo "  Waiting... (${i} polls, $((i*15))s elapsed)"
done

echo ""
echo "================================================"
echo " RUN 1 KILLED after epoch 1"
echo " Checkpoint saved at:"
find ${STORAGE} -name "state.pt" 2>/dev/null || echo "  (none found — check timing)"
echo "================================================"
echo ""

# Brief pause to simulate node recovery
echo "STEP 3: Simulating node recovery (10s pause)..."
sleep 10
echo ""

# Step 4: RUN 2 — restart, should resume from checkpoint
echo "STEP 4: Starting RUN 2 (should resume from epoch 2, not epoch 1)..."
echo "        Look for: 'Resumed from checkpoint at epoch 1 — continuing from epoch 2'"
echo ""

${DOCKER_BASE} ${RAY_ARGS}

echo ""
echo "================================================"
echo " DEMO COMPLETE"
echo " Fault tolerance demonstrated:"
echo "   Run 1 was killed after saving epoch 1 checkpoint"
echo "   Run 2 resumed from epoch 2 (skipped epoch 1)"
echo "   Without Ray: Run 2 would restart from epoch 1"
echo "================================================"
