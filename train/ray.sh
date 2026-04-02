#!/bin/bash
# Fault tolerance demo: Ray Train resumes from checkpoint after simulated failure.
# Run 1 is killed after epoch 1. Run 2 resumes from epoch 2.

REPO=/home/cc/ML-Sys-Ops-Project/train
DATA_DIR=/home/cc/ami_processed
STORAGE=/home/cc/artifacts/ray_checkpoints
MLFLOW_URI=http://129.114.25.90:8000
IMAGE=jitsi-train:latest

DOCKER_RUN="docker run --rm --gpus all \
  --shm-size=10.24gb \
  -v ${REPO}:/app \
  -v ${DATA_DIR}:/data/ami_processed \
  -v ${STORAGE}:/ray_checkpoints \
  -e MLFLOW_TRACKING_URI=${MLFLOW_URI} \
  -e TOKENIZERS_PARALLELISM=false \
  -e GIT_SHA=$(git -C ${REPO} rev-parse HEAD) \
  --entrypoint python \
  ${IMAGE} /app/train_ray.py \
  --config /app/configs/roberta_base_full.yaml \
  --data_dir /data/ami_processed \
  --storage_path /ray_checkpoints \
  --num_workers 1"

# Clean up previous run
sudo rm -rf ${STORAGE}/jitsi-roberta-fault-tolerant
mkdir -p ${STORAGE}

echo "[run 1] starting training"
${DOCKER_RUN} &
DOCKER_PID=$!

# Wait for first checkpoint
echo "[run 1] waiting for epoch 1 checkpoint..."
for i in $(seq 1 120); do
    sleep 15
    if find ${STORAGE} -name "state.pt" 2>/dev/null | grep -q .; then
        echo "[run 1] checkpoint found — killing job"
        kill ${DOCKER_PID} 2>/dev/null || true
        wait ${DOCKER_PID} 2>/dev/null || true
        break
    fi
done

# Find the experiment directory — it's the TorchTrainer_* dir under jitsi-roberta-fault-tolerant
EXPERIMENT_DIR=$(find ${STORAGE}/jitsi-roberta-fault-tolerant -maxdepth 1 -type d -name "TorchTrainer_*" | head -1)
if [ -z "${EXPERIMENT_DIR}" ]; then
    echo "error: could not find experiment directory under ${STORAGE}/jitsi-roberta-fault-tolerant"
    exit 1
fi

# Map host path to container path
EXPERIMENT_DIR_CONTAINER="/ray_checkpoints/jitsi-roberta-fault-tolerant/$(basename ${EXPERIMENT_DIR})"
CHECKPOINT_COUNT=$(find ${STORAGE} -name "state.pt" | wc -l)

echo "[run 1] killed after saving ${CHECKPOINT_COUNT} checkpoint(s)"
echo "[run 1] experiment dir: ${EXPERIMENT_DIR}"
echo ""
echo "[run 2] restoring from ${EXPERIMENT_DIR_CONTAINER}"
sleep 5

${DOCKER_RUN} --restore_path ${EXPERIMENT_DIR_CONTAINER}

echo ""
echo "[done] check MLflow for ray-train-fresh-ep1 and ray-train-resume-ep2 runs"
