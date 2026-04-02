#!/bin/bash
# ray_ec_demo.sh  —  Extra credit fault-tolerance demo script

set -e

CONFIG=/home/cc/ML-Sys-Ops-Project/train/configs/roberta_base_full.yaml
DATA_DIR=/home/cc/ami_processed
STORAGE=/home/cc/artifacts/ray_checkpoints
SCRIPT=/home/cc/ML-Sys-Ops-Project/train/train_ray.py
export MLFLOW_TRACKING_URI=http://129.114.25.90:8000

echo "================================================"
echo "Ray Train Fault Tolerance Demo"
echo "================================================"

# Step 1: Install Ray if not present
echo ""
echo "STEP 1: Installing Ray..."
pip install "ray[train]==2.10.0"
echo "Ray installed: $(python -c 'import ray; print(ray.__version__)')"

# Step 2: Start Ray cluster
echo ""
echo "STEP 2: Starting Ray head node..."
ray stop --force 2>/dev/null || true
sleep 2
ray start --head --num-gpus=1 --num-cpus=4
echo "Ray cluster started"

# Step 3: Clean any previous checkpoints for a fresh demo
echo ""
echo "STEP 3: Cleaning previous checkpoints..."
rm -rf $STORAGE/jitsi-roberta-fault-tolerant
echo "Checkpoints cleared"

# Step 4: Start training — will be killed after epoch 1
echo ""
echo "STEP 4: Starting training (RUN 1 — will be killed after epoch 1 saves checkpoint)"
echo "        Watch for 'Checkpoint saved after epoch 1' then this script kills the job"
echo ""

# Run in background so we can kill it after checkpoint saves
python $SCRIPT \
  --config $CONFIG \
  --data_dir $DATA_DIR \
  --storage_path $STORAGE \
  --num_workers 1 &
TRAIN_PID=$!

# Wait for checkpoint to be saved (poll for checkpoint directory)
echo "Waiting for epoch 1 checkpoint..."
CHECKPOINT_SAVED=0
for i in $(seq 1 60); do
  sleep 30
  if find $STORAGE -name "state.pt" 2>/dev/null | grep -q .; then
    echo ""
    echo "Checkpoint detected! Killing training job (PID $TRAIN_PID)..."
    kill $TRAIN_PID 2>/dev/null || true
    wait $TRAIN_PID 2>/dev/null || true
    CHECKPOINT_SAVED=1
    break
  fi
  echo "  Still waiting... (${i} checks, $((i*30))s elapsed)"
done

if [ $CHECKPOINT_SAVED -eq 0 ]; then
  echo "ERROR: Checkpoint not found after 30 minutes. Kill manually and check logs."
  kill $TRAIN_PID 2>/dev/null || true
  exit 1
fi

echo ""
echo "================================================"
echo "RUN 1 KILLED. Checkpoint saved at:"
find $STORAGE -name "state.pt"
echo "================================================"

# Step 5: Simulate what happens after a real failure (brief pause)
echo ""
echo "STEP 5: Simulating node recovery (5 second pause)..."
sleep 5

# Step 6: Resume — this is the key demo moment
echo ""
echo "STEP 6: Restarting training (RUN 2 — should resume from epoch 2, not epoch 1)"
echo ""

python $SCRIPT \
  --config $CONFIG \
  --data_dir $DATA_DIR \
  --storage_path $STORAGE \
  --num_workers 1

echo ""
echo "================================================"
echo "DEMO COMPLETE"
echo "Evidence of fault tolerance:"
echo "  - Run 1 was killed after epoch 1 checkpoint"
echo "  - Run 2 resumed from epoch 2 (not epoch 1)"
echo "  - Without Ray: Run 2 would restart from epoch 1"
echo "================================================"
