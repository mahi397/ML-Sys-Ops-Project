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

docker exec -e GIT_SHA=$GIT_SHA ray-head ray job submit \
  --working-dir /app \
  --runtime-env /app/runtime_ray.json \
  -- python train_ray.py --config configs/roberta_base_frozen.yaml
