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

ray job submit \
  --runtime-env runtime_ray.json \
  --working-dir . \
  -- python train_ray1.py --config configs/roberta_base_frozen.yaml
