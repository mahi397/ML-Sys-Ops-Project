import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://localhost:5000"
BUCKET = "proj07-mlflow-artifacts"
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# ── Experiment definitions ──────────────────────────────────────────────────
EXPERIMENTS = {
    "1": "jitsi-topic-segmentation",
    "2": "jitsi-topic-segmentation-sweep",
}

# ── Run metadata from your training runs table ──────────────────────────────
# Format: (run_uuid, experiment_key, run_name, params, metrics)
RUNS = [
    (
        "80b690850b9944d6bf1cb09c6dfc739b", "1", "baseline-tfidf-logreg",
        {"ngram": "(1,2)", "C": "1.0", "class_weight": "balanced", "threshold": "0.40"},
        {"test_pk": 0.303, "test_windowdiff": 0.462, "test_recall": 0.409, "test_f1": 0.144},
    ),
    (
        "f30f3587634d4ac49be2328827742a17", "1", "roberta-base-frozen-backbone",
        {"lr": 2e-4, "batch_size": 32, "epochs": 5, "threshold": 0.20, "max_oversample": 5.0},
        {"test_pk": 0.253, "test_windowdiff": 0.393, "test_recall": 0.313, "test_f1": 0.127,
         "training_time_sec": 603.4, "peak_vram_gb": 1.28},
    ),
    (
        "dbd0cb5d052c42f5bae2e898684be6cc", "1", "distilroberta-base-full-finetune",
        {"lr": 3e-5, "batch_size": 32, "epochs": 8, "threshold": 0.35,
         "max_oversample": 5.0, "weight_decay": 0.05, "dropout": 0.2},
        {"test_pk": 0.286, "test_windowdiff": 0.479, "test_recall": 0.444, "test_f1": 0.208,
         "training_time_sec": 924.1, "peak_vram_gb": 5.06},
    ),
    (
        "bfc72621f8594e07943bdcb23d45762c", "1", "roberta-base-full-finetune",
        {"lr": 2e-5, "batch_size": 16, "epochs": 8, "threshold": 0.30,
         "max_oversample": 5.0, "weight_decay": 0.05, "dropout": 0.2},
        {"test_pk": 0.228, "test_windowdiff": 0.367, "test_recall": 0.374, "test_f1": 0.222,
         "training_time_sec": 1295.2, "peak_vram_gb": 5.53},
    ),
]

# ── Restore ─────────────────────────────────────────────────────────────────
exp_id_map = {}
for key, name in EXPERIMENTS.items():
    try:
        exp = client.create_experiment(
            name,
            artifact_location=f"s3://{BUCKET}/{key}"
        )
        exp_id_map[key] = exp
        print(f"Created experiment '{name}' -> id {exp}")
    except Exception as e:
        exp = client.get_experiment_by_name(name)
        exp_id_map[key] = exp.experiment_id
        print(f"Experiment '{name}' already exists -> id {exp.experiment_id}")

for run_uuid, exp_key, run_name, params, metrics in RUNS:
    exp_id = exp_id_map[exp_key]
    print(f"\nRestoring run {run_uuid} ({run_name})...")
    run = client.create_run(
        experiment_id=exp_id,
        run_name=run_name,
        tags={"mlflow.runName": run_name, "restored": "true"}
    )
    # Log params and metrics
    for k, v in params.items():
        client.log_param(run.info.run_id, k, v)
    for k, v in metrics.items():
        client.log_metric(run.info.run_id, k, v)
    client.set_terminated(run.info.run_id, status="FINISHED")
    print(f"  Done -> new run_id: {run.info.run_id}")
    print(f"  Artifacts at: s3://{BUCKET}/{exp_key}/{run_uuid}/artifacts/")

print("\n✓ Restore complete. Check http://localhost:5000")
