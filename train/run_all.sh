# run_all.sh  —  Sequential commands to execute ALL deliverables on Chameleon.
# Run these inside your training container on the GPU node.
# Set MLFLOW_TRACKING_URI before starting.

set -e
export MLFLOW_TRACKING_URI=http://<YOUR_MLFLOW_FLOATING_IP>:8000

# ── STEP 0: Preprocess AMI corpus ──────────────────────────────────────────
python preprocess_ami.py \
  --ami_dir /data/ami_corpus \
  --output_dir /data/ami_processed \
  --seed 42

# ── STEP 1: Baseline (required simple baseline, ~5 min CPU) ────────────────
python train.py --config configs/baseline.yaml

# ── STEP 2: RoBERTa frozen backbone (~30-40 min on H100) ───────────────────
python train.py --config configs/roberta_base_frozen.yaml

# ── STEP 3: RoBERTa full fine-tune (~1-1.5h on H100) ───────────────────────
python train.py --config configs/roberta_base_full.yaml

# ── STEP 4: DistilRoBERTa full fine-tune (~1h on H100) ─────────────────────
python train.py --config configs/distilroberta_full.yaml

# ── STEP 5: Optuna hyperparameter sweep (~2-3h on H100, 20 trials) ─────────
python hparam_sweep.py \
  --base_config configs/roberta_base_full.yaml \
  --n_trials 20 \
  --epochs_per_trial 3 \
  --data_dir /data/ami_processed \
  --output_dir /artifacts/sweep

# Retrain best config from sweep for full 5 epochs
python train.py --config /artifacts/sweep/best_params.yaml

# ── STEP 6 (EXTRA CREDIT): Ray Train fault-tolerant training ───────────────
# Run on your Ray cluster (ray must be running: ray start --head)
python train_ray.py \
  --config configs/roberta_base_full.yaml \
  --num_workers 1 \
  --data_dir /data/ami_processed \
  --storage_path s3://ray

# ── STEP 7: Verify summarizer schema ───────────────────────────────────────
python summarizer_schema.py
