"""
train_ray.py — Fault-tolerant Ray Train wrapper for topic boundary detection.

Wraps the RoBERTa training logic from train.py with Ray TorchTrainer +
FailureConfig so that if a job is interrupted on Chameleon, it resumes from
the last per-epoch checkpoint — without losing MLflow run history.

Key innovation over the lab:
  The lab's fault tolerance demo used PyTorch Lightning with no MLflow
  integration. When a job resumes after failure, a naive implementation
  starts a NEW MLflow run, splitting epoch metrics across two runs (e.g.
  epochs 1-3 in run A, epochs 4-8 in run B). This makes the run unusable
  for comparing candidates.

  Fix: we persist the MLflow run_id inside the Ray checkpoint. On resume,
  we call mlflow.start_run(run_id=existing_id) to continue logging into
  the SAME run — all epochs appear under a single, continuous MLflow run.

Usage (submit from jupyter container on node-mltrain):
  ray job submit \\
    --runtime-env runtime_ray.json \\
    --working-dir . \\
    -- python train_ray.py --config configs/roberta_base_frozen.yaml

  # Override individual params:
  ray job submit --runtime-env runtime_ray.json --working-dir . \\
    -- python train_ray.py --config configs/roberta_base_frozen.yaml --lr 3e-5

Environment variables (set in runtime_ray.json):
  MLFLOW_TRACKING_URI   — your Chameleon MLflow instance
  RAY_STORAGE_PATH      — local path for checkpoints (default: /ray_checkpoints)
"""

import argparse
import json
import os
import time
import yaml
import logging
import platform
from collections import defaultdict
from typing import Dict

import numpy as np
import mlflow
import mlflow.pytorch
from sklearn.metrics import f1_score, precision_score, recall_score

import ray
import ray.train
from ray.train import RunConfig, ScalingConfig, FailureConfig
from ray.train.torch import TorchTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Default config (mirrors train.py) ─────────────────────────────────────────

DEFAULT_CONFIG = {
    "model_name": "roberta-base",
    "freeze_backbone": False,
    "lr": 2e-5,
    "batch_size": 16,
    "epochs": 5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_seq_len": 256,
    "dropout": 0.1,
    "early_stopping_patience": 2,
    "max_oversample": 5.0,
    "data_dir": "/data/ami_processed",
    "output_dir": "/artifacts/models",
    "experiment_name": "jitsi-topic-segmentation",
    "seed": 42,
}


def load_config(config_path: str, overrides: Dict) -> Dict:
    cfg = DEFAULT_CONFIG.copy()
    if config_path:
        with open(config_path) as f:
            cfg.update(yaml.safe_load(f))
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    for key in ("lr", "weight_decay", "warmup_ratio", "dropout"):
        if key in cfg and cfg[key] is not None:
            cfg[key] = float(cfg[key])
    for key in ("batch_size", "epochs", "max_seq_len", "early_stopping_patience"):
        if key in cfg and cfg[key] is not None:
            cfg[key] = int(cfg[key])
    return cfg


# ── Helpers (identical to train.py) ───────────────────────────────────────────

def format_window(window: list) -> str:
    parts = []
    for utt in sorted(window, key=lambda u: u["position"]):
        if utt["text"].strip():
            parts.append(f"[SPEAKER_{utt['speaker']}]: {utt['text']}")
    return " ".join(parts)


def load_jsonl(path: str):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def load_split(data_dir: str, split: str):
    examples = load_jsonl(os.path.join(data_dir, f"{split}.jsonl"))
    texts      = [format_window(e["window"]) for e in examples]
    labels     = [e["label"] for e in examples]
    meeting_ids = [e["meeting_id"] for e in examples]
    return texts, labels, meeting_ids


def compute_segmentation_metrics(true_labels, pred_labels, meeting_ids=None):
    try:
        from nltk.metrics.segmentation import windowdiff, pk as pk_metric
    except ImportError:
        return {"window_diff": -1.0, "pk": -1.0}

    if meeting_ids is None:
        return {"window_diff": -1.0, "pk": -1.0}

    meeting_true = defaultdict(list)
    meeting_pred = defaultdict(list)
    for mid, t, p in zip(meeting_ids, true_labels, pred_labels):
        meeting_true[mid].append(t)
        meeting_pred[mid].append(p)

    wd_scores, pk_scores = [], []
    for mid in meeting_true:
        ref = "".join(str(l) for l in meeting_true[mid])
        hyp = "".join(str(l) for l in meeting_pred[mid])
        if len(ref) < 4:
            continue
        k = max(2, len(ref) // 10)
        try:
            wd_scores.append(windowdiff(ref, hyp, k=k, boundary="1"))
            pk_scores.append(pk_metric(ref, hyp, k=k, boundary="1"))
        except Exception as e:
            log.warning(f"Segmentation metric error for {mid}: {e}")

    if not wd_scores:
        return {"window_diff": -1.0, "pk": -1.0}
    return {
        "window_diff": round(float(np.mean(wd_scores)), 4),
        "pk":          round(float(np.mean(pk_scores)), 4),
    }


THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

def sweep_thresholds(probs, true_labels, meeting_ids):
    probs       = np.array(probs)
    true_labels = np.array(true_labels)
    best_pk        = float("inf")
    best_threshold = 0.5
    best_metrics   = {}

    for thr in THRESHOLDS:
        preds  = (probs >= thr).astype(int)
        seg    = compute_segmentation_metrics(true_labels.tolist(), preds.tolist(), meeting_ids)
        pk_val = seg.get("pk", 1.0)
        if pk_val < best_pk:
            best_pk        = pk_val
            best_threshold = thr
            best_metrics   = {
                "f1":        f1_score(true_labels, preds, zero_division=0),
                "precision": precision_score(true_labels, preds, zero_division=0),
                "recall":    recall_score(true_labels, preds, zero_division=0),
                "pk":        seg.get("pk", 1.0),
                "window_diff": seg.get("window_diff", 1.0),
                "n_predicted": int(preds.sum()),
                "n_true":      int(true_labels.sum()),
            }

    preds_05 = (probs >= 0.5).astype(int)
    seg_05   = compute_segmentation_metrics(true_labels.tolist(), preds_05.tolist(), meeting_ids)
    return best_threshold, best_metrics, {
        "f1_at_0.5": f1_score(true_labels, preds_05, zero_division=0),
        "pk_at_0.5": seg_05.get("pk", 1.0),
    }


def log_environment():
    import torch
    env = {
        "python_version":  platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available":  str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        env["gpu_name"]    = torch.cuda.get_device_name(0)
        env["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        env["gpu_count"]   = torch.cuda.device_count()
    mlflow.log_params(env)


# ── Core training function — runs inside each Ray worker ──────────────────────

def train_func(config: Dict):
    """
    All training logic from run_roberta() in train.py, wrapped for Ray Train.

    MLflow continuity across fault tolerance:
    ─────────────────────────────────────────
    On the first run, we create a new MLflow run and save its run_id into the
    Ray checkpoint dict alongside model/optimizer state.

    On resume (after a failure), ray.train.get_checkpoint() returns the last
    saved checkpoint. We extract the run_id from it and call:
        mlflow.start_run(run_id=existing_run_id)
    This re-opens the SAME MLflow run, so epochs after the resume are logged
    to the same run as epochs before — producing a single, continuous run in
    the MLflow UI instead of two broken fragments.
    """
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Ray worker device: {device}")

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # ── Dataset ───────────────────────────────────────────────────────────────
    class WindowDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.encodings = tokenizer(
                texts, truncation=True, padding="max_length",
                max_length=max_len, return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids":      self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels":         self.labels[idx],
            }

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    special_tokens = [f"[SPEAKER_{s}]" for s in "ABCDEFGH"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    log.info("Loading and tokenizing data...")
    train_texts, train_labels, _               = load_split(config["data_dir"], "train")
    val_texts,   val_labels,   val_meeting_ids = load_split(config["data_dir"], "val")
    test_texts,  test_labels,  test_meeting_ids = load_split(config["data_dir"], "test")

    train_ds = WindowDataset(train_texts, train_labels, tokenizer, config["max_seq_len"])
    val_ds   = WindowDataset(val_texts,   val_labels,   tokenizer, config["max_seq_len"])
    test_ds  = WindowDataset(test_texts,  test_labels,  tokenizer, config["max_seq_len"])

    # ── WeightedRandomSampler (identical logic to train.py) ───────────────────
    n_pos           = sum(train_labels)
    n_neg           = len(train_labels) - n_pos
    max_oversample  = config.get("max_oversample", 5.0)
    actual_ratio    = n_neg / max(n_pos, 1)
    effective_ratio = min(actual_ratio, max_oversample)
    sample_weights  = [1.0 if l == 0 else effective_ratio for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    log.info(f"Class imbalance: {n_pos} pos / {n_neg} neg → oversample ratio capped at {effective_ratio:.1f}x")

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              sampler=sampler, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"] * 2, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=config["batch_size"] * 2, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=2,
        hidden_dropout_prob=config["dropout"],
        attention_probs_dropout_prob=config["dropout"],
    )
    model.resize_token_embeddings(len(tokenizer))

    if config["freeze_backbone"]:
        log.info("Freezing backbone — training classification head only")
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    model = model.to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_trainable:,}")

    loss_fn   = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"], weight_decay=config["weight_decay"],
    )
    total_steps  = len(train_loader) * config["epochs"]
    warmup_steps = int(config["warmup_ratio"] * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    run_name = (
        f"{config['model_name'].replace('/', '-')}"
        f"-{'frozen' if config['freeze_backbone'] else 'full'}"
        f"-lr{config['lr']}-bs{config['batch_size']}"
        f"-ray-ft"
    )
    os.makedirs(config["output_dir"], exist_ok=True)
    best_model_path = os.path.join(config["output_dir"], f"{run_name}_best.pt")

    # ── Training state (overwritten if resuming from checkpoint) ──────────────
    start_epoch          = 1
    best_val_pk          = float("inf")
    best_val_f1          = 0.0
    best_threshold       = 0.5
    prev_epoch_threshold = 0.5
    patience_counter     = 0
    existing_mlflow_run_id = None   # KEY: persisted across checkpoints

    # ── Restore from Ray checkpoint if recovering after failure ───────────────
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        log.info("Ray checkpoint found — restoring state after failure...")
        with checkpoint.as_directory() as ckpt_dir:
            state = torch.load(os.path.join(ckpt_dir, "ray_checkpoint.pt"),
                               map_location=device)

        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])

        start_epoch            = state["epoch"] + 1
        best_val_pk            = state["best_val_pk"]
        best_val_f1            = state["best_val_f1"]
        best_threshold         = state["best_threshold"]
        prev_epoch_threshold   = state["prev_epoch_threshold"]
        patience_counter       = state["patience_counter"]

        # ── MLflow continuity: re-open the SAME run ───────────────────────────
        # This is the key fix. Without this, mlflow.start_run() below would
        # create a new run, splitting epoch metrics across two separate runs.
        # By passing run_id=existing_mlflow_run_id we resume logging into
        # the original run — all epochs appear in one continuous run.
        existing_mlflow_run_id = state["mlflow_run_id"]

        log.info(
            f"Resumed from checkpoint: starting at epoch {start_epoch}, "
            f"best_val_pk={best_val_pk:.4f}, "
            f"MLflow run_id={existing_mlflow_run_id}"
        )

        # Also restore best model weights if they exist on disk
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            model = model.to(device)
            log.info("Best model weights restored from disk")
    else:
        log.info("No checkpoint found — starting from epoch 1")

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:8000"))
    mlflow.set_experiment(config["experiment_name"])

    # If resuming: re-open existing run.  If fresh: create new run.
    mlflow_start_kwargs = (
        {"run_id": existing_mlflow_run_id}
        if existing_mlflow_run_id
        else {"run_name": run_name}
    )

    total_train_start = time.time()

    with mlflow.start_run(**mlflow_start_kwargs) as mlflow_run:
        current_run_id = mlflow_run.info.run_id

        # Log params only on fresh start (re-logging on resume would error
        # because MLflow forbids overwriting existing params)
        if not existing_mlflow_run_id:
            mlflow.log_params(config)
            mlflow.log_params({
                "n_trainable_params": n_trainable,
                "ray_fault_tolerant":  True,
            })
            log_environment()
            mlflow.log_param("git_sha", os.environ.get("GIT_SHA", "unknown"))
            log.info(f"New MLflow run created: {current_run_id}")
        else:
            # On resume, just log a note so it's visible in MLflow
            mlflow.log_param("resumed_at_epoch", start_epoch)
            log.info(f"Resumed into existing MLflow run: {current_run_id}")

        # ── Training loop (identical logic to train.py) ───────────────────────
        for epoch in range(start_epoch, config["epochs"] + 1):
            model.train()
            epoch_start = time.time()
            train_loss  = 0.0

            for batch in train_loader:
                optimizer.zero_grad()
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss    = loss_fn(outputs.logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

            epoch_time = time.time() - epoch_start

            # ── Validation with threshold sweep ───────────────────────────────
            model.eval()
            val_probs, val_true = [], []
            with torch.no_grad():
                for batch in val_loader:
                    logits = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                    ).logits
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                    val_probs.extend(probs)
                    val_true.extend(batch["labels"].numpy())

            epoch_threshold, epoch_metrics, ref_metrics = sweep_thresholds(
                val_probs, val_true, val_meeting_ids)

            # Smooth threshold — identical logic to train.py
            if epoch > 1:
                epoch_threshold = float(np.clip(
                    epoch_threshold,
                    prev_epoch_threshold - 0.15,
                    prev_epoch_threshold + 0.15,
                ))
                smoothed_preds = (np.array(val_probs) >= epoch_threshold).astype(int)
                smoothed_seg   = compute_segmentation_metrics(
                    val_true, smoothed_preds.tolist(), val_meeting_ids)
                epoch_metrics = {
                    "f1":        f1_score(val_true, smoothed_preds, zero_division=0),
                    "precision": precision_score(val_true, smoothed_preds, zero_division=0),
                    "recall":    recall_score(val_true, smoothed_preds, zero_division=0),
                    "pk":        smoothed_seg.get("pk", 1.0),
                    "window_diff": smoothed_seg.get("window_diff", 1.0),
                    "n_predicted": int(smoothed_preds.sum()),
                    "n_true":      int(np.array(val_true).sum()),
                }
            prev_epoch_threshold = epoch_threshold

            val_f1 = epoch_metrics["f1"]
            val_pk = epoch_metrics["pk"]

            epoch_log = {
                "train_loss":        round(train_loss / len(train_loader), 4),
                "val_f1":            round(val_f1, 4),
                "val_precision":     round(epoch_metrics["precision"], 4),
                "val_recall":        round(epoch_metrics["recall"], 4),
                "val_pk":            round(val_pk, 4),
                "val_window_diff":   round(epoch_metrics["window_diff"], 4),
                "val_best_threshold": epoch_threshold,
                "val_n_predicted":   epoch_metrics["n_predicted"],
                "val_f1_at_0.5":     round(ref_metrics["f1_at_0.5"], 4),
                "val_pk_at_0.5":     round(ref_metrics["pk_at_0.5"], 4),
                "epoch_time_sec":    round(epoch_time, 1),
            }
            # Log to MLflow — goes into the same run whether fresh or resumed
            mlflow.log_metrics(epoch_log, step=epoch)

            print(
                f"Epoch {epoch}/{config['epochs']} | "
                f"loss={epoch_log['train_loss']:.4f} | "
                f"val_f1={val_f1:.4f} (thr={epoch_threshold:.2f}) | "
                f"val_pk={val_pk:.4f} | "
                f"predicted={epoch_metrics['n_predicted']} "
                f"true={epoch_metrics['n_true']} | "
                f"time={epoch_time:.1f}s | "
                f"mlflow_run={current_run_id[:8]}",  # short ID to confirm continuity
                flush=True,
            )

            # ── Early stopping on Pk (lower is better) ────────────────────────
            if val_pk < best_val_pk:
                prev_best    = best_val_pk
                best_val_pk  = val_pk
                best_val_f1  = val_f1
                best_threshold = epoch_threshold
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(
                    f"   New best val_pk: {val_pk:.4f} "
                    f"(improved from {prev_best:.4f}), "
                    f"threshold={best_threshold:.2f}, checkpoint saved",
                    flush=True,
                )
            else:
                if epoch_metrics["n_predicted"] > 0:
                    patience_counter += 1
                    print(
                        f"   val_pk did not improve "
                        f"({val_pk:.4f} vs best {best_val_pk:.4f}), "
                        f"patience {patience_counter}/{config['early_stopping_patience']}",
                        flush=True,
                    )
                else:
                    torch.save(model.state_dict(), best_model_path)
                    print("   No boundaries predicted yet, not counting patience", flush=True)

                if patience_counter >= config["early_stopping_patience"]:
                    print(
                        f"   Early stopping triggered at epoch {epoch}",
                        flush=True,
                    )
                    break

            # ── Save Ray checkpoint after every epoch ─────────────────────────
            # Includes mlflow_run_id so the resumed job can re-open the same run.
            with ray.train.tempdir_path() as tmpdir:
                torch.save(
                    {
                        "epoch":                epoch,
                        "model_state":          model.state_dict(),
                        "optimizer_state":      optimizer.state_dict(),
                        "scheduler_state":      scheduler.state_dict(),
                        "best_val_pk":          best_val_pk,
                        "best_val_f1":          best_val_f1,
                        "best_threshold":       best_threshold,
                        "prev_epoch_threshold": prev_epoch_threshold,
                        "patience_counter":     patience_counter,
                        # ── The key field for MLflow continuity ───────────────
                        "mlflow_run_id":        current_run_id,
                    },
                    os.path.join(tmpdir, "ray_checkpoint.pt"),
                )
                ray.train.report(
                    metrics={
                        "val_pk":  val_pk,
                        "val_f1":  val_f1,
                        "epoch":   epoch,
                    },
                    checkpoint=ray.train.Checkpoint.from_directory(tmpdir),
                )

        # ── Final test evaluation ─────────────────────────────────────────────
        # Load best weights found across all epochs (including pre-failure ones)
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        test_probs, test_true = [], []
        with torch.no_grad():
            for batch in test_loader:
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                test_probs.extend(probs)
                test_true.extend(batch["labels"].numpy())

        test_preds    = (np.array(test_probs) >= best_threshold).astype(int)
        test_seg      = compute_segmentation_metrics(
            test_true, test_preds.tolist(), test_meeting_ids)
        total_time    = time.time() - total_train_start

        final_metrics = {
            "test_f1":               round(f1_score(test_true, test_preds, zero_division=0), 4),
            "test_precision":        round(precision_score(test_true, test_preds, zero_division=0), 4),
            "test_recall":           round(recall_score(test_true, test_preds, zero_division=0), 4),
            "best_val_pk":           round(best_val_pk, 4),
            "best_val_f1":           round(best_val_f1, 4),
            "best_threshold":        best_threshold,
            "total_training_time_sec": round(total_time, 1),
            "test_pk":               round(test_seg.get("pk", -1.0), 4),
            "test_window_diff":      round(test_seg.get("window_diff", -1.0), 4),
        }
        if torch.cuda.is_available():
            final_metrics["peak_vram_gb"] = round(
                torch.cuda.max_memory_allocated() / 1e9, 2)

        mlflow.log_metrics(final_metrics)
        mlflow.log_param("best_threshold", best_threshold)
        log.info(
            f"Test F1: {final_metrics['test_f1']:.4f} | "
            f"Test Pk: {final_metrics['test_pk']:.4f} | "
            f"Best threshold: {best_threshold:.2f}"
        )

        mlflow.pytorch.log_model(
            model, artifact_path="model",
            pip_requirements=["transformers==4.40.0", "torch==2.2.0"],
        )
        tokenizer.save_pretrained(os.path.join(config["output_dir"], "tokenizer"))
        mlflow.log_artifacts(
            os.path.join(config["output_dir"], "tokenizer"),
            artifact_path="tokenizer",
        )

        print(f"\nMLflow run ID (single continuous run): {current_run_id}", flush=True)


# ── Launcher ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fault-tolerant Ray Train wrapper with MLflow run continuity"
    )
    parser.add_argument("--config",         default=None)
    parser.add_argument("--model_name",     default=None)
    parser.add_argument("--lr",             type=float, default=None)
    parser.add_argument("--batch_size",     type=int,   default=None)
    parser.add_argument("--epochs",         type=int,   default=None)
    parser.add_argument("--freeze_backbone", action="store_true", default=None)
    parser.add_argument("--max_seq_len",    type=int,   default=None)
    parser.add_argument("--data_dir",       default=None)
    parser.add_argument("--output_dir",     default=None)
    parser.add_argument("--experiment_name", default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    cfg       = load_config(args.config, overrides)

    # Storage path for Ray checkpoints — use persistent block volume on Chameleon
    storage_path = os.environ.get("RAY_STORAGE_PATH", "/ray_checkpoints")
    os.makedirs(storage_path, exist_ok=True)
    full_storage = os.path.join(storage_path, "jitsi-roberta-fault-tolerant")

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=cfg,
        scaling_config=ScalingConfig(
            num_workers=1,
            use_gpu=True,
            resources_per_worker={"GPU": 1, "CPU": 4},
        ),
        run_config=RunConfig(
            name="jitsi-roberta-fault-tolerant",
            storage_path=full_storage,
            failure_config=FailureConfig(max_failures=2),
        ),
    )

    print(f"\nStarting Ray Train job — checkpoints at: {full_storage}", flush=True)
    print(f"Training config: {json.dumps(cfg, indent=2)}\n", flush=True)

    result = trainer.fit()

    print(f"\nTraining complete.", flush=True)
    print(f"Best checkpoint: {result.checkpoint}", flush=True)
    print(f"Final metrics:   {result.metrics}",    flush=True)


if __name__ == "__main__":
    main()
