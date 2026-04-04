#!/bin/bash
export GIT_SHA=$(git rev-parse HEAD)
echo "Git SHA: $GIT_SHA"

# Update runtime_ray.json on the fly with current SHA
cat > runtime_ray.json << EOF
{
  "env_vars": {
    "MLFLOW_TRACKING_URI": "http://129.114.25.90:8000",
    "RAY_STORAGE_PATH": "/ray_checkpoints",
    "GIT_SHA": "$GIT_SHA"
  }
}
EOF

# Submit the job from inside the container where Ray is running
docker exec -e GIT_SHA=$GIT_SHA ray-head \
  /bin/bash -c "cd /app/train && ray job submit \
    --address http://127.0.0.1:8265 \
    --runtime-env runtime_ray.json \
    --working-dir . \
    -- python train_ray.py --config configs/roberta_base_frozen.yaml"
