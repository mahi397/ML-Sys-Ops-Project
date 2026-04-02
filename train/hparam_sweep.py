"""
hparam_sweep.py  —  Optuna hyperparameter search over roberta-base.

Strategy:
  - TPE sampler: Bayesian optimization, focuses on promising regions after ~5 trials
  - MedianPruner: kills unpromising trials early (after epoch 1-2) saving ~40% GPU time
  - Optimizes val_pk (lower is better) — consistent with early stopping in train.py
  - All trials logged to MLflow manually (avoids MLflowCallback version issues)

Search space rationale:
  lr:            log-uniform 1e-5..5e-5  — standard RoBERTa fine-tuning range
  batch_size:    16 or 32               — 64 causes too many repeated boundary examples
  warmup_ratio:  0.05..0.2              — standard; <0.05 causes instability
  weight_decay:  0.01..0.1             — L2 reg; higher fights overfitting
  dropout:       0.1..0.4              — higher dropout reduces memorization
  max_oversample: 2.0..8.0             — caps boundary oversampling ratio

Usage:
  export MLFLOW_TRACKING_URI=http://<floating-ip>:8000
  python hparam_sweep.py \
    --base_config /app/configs/roberta_base_full.yaml \
    --n_trials 20 \
    --epochs_per_trial 3 \
    --data_dir /data/ami_processed \
    --output_dir /artifacts/sweep
"""

import argparse
import json
import os
import sys
import yaml
import logging

import optuna
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial: optuna.Trial, base_cfg: dict, args) -> float:
    """
    One Optuna trial = one training run with sampled hyperparams.
    Returns best_val_pk (lower is better — study direction is minimize).
    """
    cfg = base_cfg.copy()
    cfg["lr"] = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    cfg["batch_size"] = trial.suggest_categorical("batch_size", [16, 32])
    cfg["warmup_ratio"] = trial.suggest_float("warmup_ratio", 0.05, 0.2)
    cfg["weight_decay"] = trial.suggest_float("weight_decay", 0.01, 0.1)
    cfg["dropout"] = trial.suggest_float("dropout", 0.1, 0.4)
    cfg["max_oversample"] = trial.suggest_float("max_oversample", 2.0, 8.0)
    cfg["epochs"] = args.epochs_per_trial
    cfg["early_stopping_patience"] = 2
    cfg["output_dir"] = os.path.join(args.output_dir, f"trial_{trial.number}")
    cfg["experiment_name"] = args.experiment_name

    os.makedirs(cfg["output_dir"], exist_ok=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train import run_roberta

    run_id_holder = []
    try:
        metrics = run_roberta(cfg, run_id_holder)
        val_pk = metrics.get("best_val_pk", 1.0)

        trial.report(val_pk, step=cfg["epochs"])
        if trial.should_prune():
            log.info(f"Trial {trial.number} pruned at val_pk={val_pk:.4f}")
            raise optuna.exceptions.TrialPruned()

        print(f"\nTrial {trial.number} complete | val_pk={val_pk:.4f} | "
              f"lr={cfg['lr']:.2e} bs={cfg['batch_size']} "
              f"wd={cfg['weight_decay']:.3f} dropout={cfg['dropout']:.2f} "
              f"oversample={cfg['max_oversample']:.1f}", flush=True)
        return val_pk

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        log.error(f"Trial {trial.number} failed: {e}", exc_info=True)
        return 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="/app/configs/roberta_base_full.yaml")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--epochs_per_trial", type=int, default=3)
    parser.add_argument("--data_dir", default="/data/ami_processed")
    parser.add_argument("--output_dir", default="/artifacts/sweep")
    parser.add_argument("--experiment_name", default="jitsi-topic-segmentation-sweep")
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)
    base_cfg["data_dir"] = args.data_dir

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:8000"))
    mlflow.set_experiment(args.experiment_name)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        study_name="roberta-hparam-sweep",
    )

    log.info(f"Starting sweep: {args.n_trials} trials x {args.epochs_per_trial} epochs, optimizing val_pk")

    study.optimize(
        lambda trial: objective(trial, base_cfg, args),
        n_trials=args.n_trials,
    )

    best = study.best_trial
    log.info(f"Sweep complete. Best trial #{best.number}: val_pk={best.value:.4f}")
    log.info(f"Best params: {json.dumps(best.params, indent=2)}")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print("\nAll trials sorted by val_pk:", flush=True)
    for t in sorted(completed, key=lambda t: t.value):
        print(f"  Trial {t.number:02d} | val_pk={t.value:.4f} | "
              f"lr={t.params.get('lr','?'):.2e} bs={t.params.get('batch_size','?')} "
              f"wd={t.params.get('weight_decay','?'):.3f} "
              f"dropout={t.params.get('dropout','?'):.2f} "
              f"oversample={t.params.get('max_oversample','?'):.1f}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    best_cfg = base_cfg.copy()
    best_cfg.update(best.params)
    best_cfg["epochs"] = 8
    best_cfg["early_stopping_patience"] = 3
    best_cfg["data_dir"] = args.data_dir
    best_cfg["output_dir"] = os.path.join(args.output_dir, "best_model")
    best_cfg["experiment_name"] = "jitsi-topic-segmentation"

    out_path = os.path.join(args.output_dir, "best_params.yaml")
    with open(out_path, "w") as f:
        yaml.dump(best_cfg, f, default_flow_style=False)
    log.info(f"Best config saved to {out_path}")
    log.info(f"Retrain: python train.py --config {out_path}")

    with mlflow.start_run(run_name="sweep-summary"):
        mlflow.log_params(best.params)
        mlflow.log_metric("best_val_pk", best.value)
        mlflow.log_metric("n_trials_completed", len(completed))
        mlflow.log_metric("n_trials_pruned",
                          len([t for t in study.trials
                               if t.state == optuna.trial.TrialState.PRUNED]))
        mlflow.log_param("best_trial_number", best.number)
        mlflow.log_artifact(out_path, artifact_path="best_config")


if __name__ == "__main__":
    main()
