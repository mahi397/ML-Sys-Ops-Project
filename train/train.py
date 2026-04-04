"""
train.py  —  Single configurable training script for topic boundary detection.

All candidates (baseline, roberta-base frozen, roberta-base full, distilroberta)
are selected via config. 

Usage:
  # Run with a config file
  python train.py --config configs/roberta_base_frozen.yaml

  # Override individual params on CLI
  python train.py --config configs/roberta_base_frozen.yaml --lr 3e-5 --epochs 3

  # Baseline (no GPU needed)
  python train.py --config configs/baseline.yaml

MLflow tracking:
  Set MLFLOW_TRACKING_URI env var to your Chameleon MLflow instance before running.
  export MLFLOW_TRACKING_URI=http://<floating-ip>:8000
"""

import argparse
import json
import os
import time
import yaml
import logging
import platform
from pathlib import Path
from typing import Dict, Any

import nltk
from nltk.metrics.segmentation import windowdiff, pk as pk_metric
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from sklearn.metrics import f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Config loading ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "model_name": "roberta-base",      # or "distilroberta-base", "roberta-large", "baseline"
    "freeze_backbone": False,           # True = train head only
    "lr": 2e-5,
    "batch_size": 16,
    "epochs": 5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_seq_len": 256,
    "dropout": 0.1,
    "early_stopping_patience": 2,      # stop if val F1 doesn't improve for N epochs
    "data_dir": "/data/ami_processed", # path to JSONL splits from preprocess_ami.py
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
    # Ensure numeric types are correct — YAML sometimes reads scientific notation as str
    for key in ("lr", "weight_decay", "warmup_ratio", "dropout", "feedback_weight"):
        if key in cfg and cfg[key] is not None:
            cfg[key] = float(cfg[key])
    for key in ("batch_size", "epochs", "max_seq_len", "early_stopping_patience"):
        if key in cfg and cfg[key] is not None:
            cfg[key] = int(cfg[key])
    return cfg


# ── Tokenization — must be IDENTICAL to serving code ──────────────────────────
# Decision: speaker tag format [SPEAKER_X] is stored as a separate "speaker"
# field in the JSON so if we change the format (e.g. to <speaker=A>), we only
# change this function and the serving mirror — the JSON contract is unchanged.

def format_window(window: list) -> str:
    """
    Convert a 7-utterance window to the string fed to RoBERTa.
    Format: "[SPEAKER_A]: text [SPEAKER_B]: text ..."
    Empty utterances (padding) are skipped.
    """
    parts = []
    for utt in sorted(window, key=lambda u: u["position"]):
        if utt["text"].strip():
            parts.append(f"[SPEAKER_{utt['speaker']}]: {utt['text']}")
    return " ".join(parts)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_jsonl(path: str):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def load_split(data_dir: str, split: str):
    """Load a JSONL split and return (texts, labels, meeting_ids)."""
    examples = load_jsonl(os.path.join(data_dir, f"{split}.jsonl"))
    texts = [format_window(e["window"]) for e in examples]
    labels = [e["label"] for e in examples]
    meeting_ids = [e["meeting_id"] for e in examples]
    return texts, labels, meeting_ids


# ── Baseline: TF-IDF + Logistic Regression ────────────────────────────────────

def run_baseline(cfg: Dict, run_id_holder: list):
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    log.info("Running TF-IDF + Logistic Regression baseline")
    train_texts, train_labels, _ = load_split(cfg["data_dir"], "train")
    val_texts, val_labels, val_meeting_ids = load_split(cfg["data_dir"], "val")
    test_texts, test_labels, test_meeting_ids = load_split(cfg["data_dir"], "test")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, C=cfg.get("C", 1.0),
                                   class_weight="balanced")),
    ])

    t0 = time.time()
    pipe.fit(train_texts, train_labels)
    train_time = time.time() - t0

    # Baseline uses predict_proba for threshold sweeping
    val_probs = pipe.predict_proba(val_texts)[:, 1]
    test_probs = pipe.predict_proba(test_texts)[:, 1]

    # Sweep thresholds on val set
    best_threshold, val_metrics, ref_metrics = sweep_thresholds(
        val_probs, val_labels, val_meeting_ids)

    # Apply best threshold to test set
    test_preds = (test_probs >= best_threshold).astype(int)
    test_seg = compute_segmentation_metrics(
        test_labels, test_preds.tolist(), test_meeting_ids)

    metrics = {
        "val_f1": round(val_metrics["f1"], 4),
        "val_precision": round(val_metrics["precision"], 4),
        "val_recall": round(val_metrics["recall"], 4),
        "val_pk": round(val_metrics["pk"], 4),
        "val_window_diff": round(val_metrics["window_diff"], 4),
        "val_best_threshold": best_threshold,
        "val_f1_at_0.5": round(ref_metrics["f1_at_0.5"], 4),
        "test_f1": round(f1_score(test_labels, test_preds, zero_division=0), 4),
        "test_precision": round(precision_score(test_labels, test_preds, zero_division=0), 4),
        "test_recall": round(recall_score(test_labels, test_preds, zero_division=0), 4),
        "test_pk": round(test_seg.get("pk", -1.0), 4),
        "test_window_diff": round(test_seg.get("window_diff", -1.0), 4),
        "best_threshold": best_threshold,
        "total_training_time_sec": train_time,
        "gpu_used": 0,
    }

    with mlflow.start_run(run_name="baseline-tfidf-lr") as run:
        run_id_holder.append(run.info.run_id)
        mlflow.log_params({k: v for k, v in cfg.items()
                           if k in ("model_name", "epochs", "seed")})
        mlflow.log_params({"C": cfg.get("C", 1.0), "ngram_range": "1,2",
                            "max_features": 50000})
        log_environment()
        mlflow.log_param("git_sha", os.environ.get("GIT_SHA", "unknown"))
        mlflow.log_param("best_threshold", best_threshold)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipe, artifact_path="model")
        log.info(f"Baseline val F1: {metrics['val_f1']:.4f}")

    return metrics


# ── Segmentation metrics (WindowDiff and Pk) ──────────────────────────────────
# These are standard topic segmentation metrics beyond binary F1.
# WindowDiff penalizes off-by-one boundary placement, Pk measures probability
# of misclassifying a pair of utterances as in the same/different segment.

def compute_segmentation_metrics(true_labels, pred_labels, meeting_ids=None):
    """
    Compute WindowDiff and Pk averaged across meetings.
    Both metrics must be computed per meeting then averaged — computing them
    on the concatenated corpus sequence is incorrect because meeting boundaries
    are not real topic boundaries and corrupt the window-based calculations.
    Lower is better for both.
    """
    # try:
    #     import nltk
    #     nltk.download("punkt", quiet=True)
    #     from nltk.metrics.segmentation import windowdiff, pk as pk_metric
    # except ImportError:
    #     log.warning("nltk not installed — skipping WindowDiff/Pk.")
    #     return {"window_diff": -1.0, "pk": -1.0}

    if meeting_ids is None:
        # No meeting grouping available — skip rather than compute incorrectly
        log.warning("No meeting_ids provided — skipping WindowDiff/Pk.")
        return {"window_diff": -1.0, "pk": -1.0}

    # Group labels by meeting
    from collections import defaultdict
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
            continue  # too short for meaningful segmentation metrics
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
        "pk": round(float(np.mean(pk_scores)), 4),
    }



# ── Threshold sweeping ────────────────────────────────────────────────────────
# With severe class imbalance (~40:1), the model outputs low probabilities
# for boundaries (e.g. 0.10-0.25) rather than > 0.5. Using a fixed threshold
# of 0.5 predicts zero boundaries and gives F1=0.
# Threshold sweeping finds the best threshold on the val set, which is then
# logged to MLflow and handed to the serving team as the recommended value.
# The threshold is a serving hyperparameter, not a training one — it can be
# tuned without retraining.

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

def sweep_thresholds(probs, true_labels, meeting_ids):
    """
    Sweep probability thresholds and return the one with best val Pk.
    Also returns metrics at the best threshold and at fixed 0.5 for comparison.
    Pk is used as the sweep criterion (not F1) because:
      - Pk is threshold-independent in meaning — it measures segment quality
      - F1 is unstable under imbalance and can be gamed by a single threshold
      - Pk directly reflects what matters: are the resulting topic segments good?
    """
    probs = np.array(probs)
    true_labels = np.array(true_labels)

    best_pk = float("inf")
    best_threshold = 0.5
    best_metrics = {}

    for thr in THRESHOLDS:
        preds = (probs >= thr).astype(int)
        seg = compute_segmentation_metrics(
            true_labels.tolist(), preds.tolist(), meeting_ids)
        pk_val = seg.get("pk", 1.0)
        if pk_val < best_pk:
            best_pk = pk_val
            best_threshold = thr
            best_metrics = {
                "f1": f1_score(true_labels, preds, zero_division=0),
                "precision": precision_score(true_labels, preds, zero_division=0),
                "recall": recall_score(true_labels, preds, zero_division=0),
                "pk": seg.get("pk", 1.0),
                "window_diff": seg.get("window_diff", 1.0),
                "n_predicted": int(preds.sum()),
                "n_true": int(true_labels.sum()),
            }

    # Also compute metrics at fixed 0.5 for reference
    preds_05 = (probs >= 0.5).astype(int)
    seg_05 = compute_segmentation_metrics(
        true_labels.tolist(), preds_05.tolist(), meeting_ids)

    return best_threshold, best_metrics, {
        "f1_at_0.5": f1_score(true_labels, preds_05, zero_division=0),
        "pk_at_0.5": seg_05.get("pk", 1.0),
    }



def log_environment():
    """Log GPU info, Python version, and hostname to MLflow."""
    import torch
    env = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        env["gpu_count"] = torch.cuda.device_count()
    mlflow.log_params(env)


# ── RoBERTa training ───────────────────────────────────────────────────────────

def run_roberta(cfg: Dict, run_id_holder: list):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # ── Dataset ──────────────────────────────────────────────────────────────

    class WindowDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.labels[idx],
            }

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    # Add speaker tokens so they are not split by the tokenizer
    special_tokens = [f"[SPEAKER_{s}]" for s in "ABCDEFGH"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    log.info("Loading and tokenizing data...")
    train_texts, train_labels, _ = load_split(cfg["data_dir"], "train")
    val_texts, val_labels, val_meeting_ids = load_split(cfg["data_dir"], "val")
    test_texts, test_labels, test_meeting_ids = load_split(cfg["data_dir"], "test")

    train_ds = WindowDataset(train_texts, train_labels, tokenizer, cfg["max_seq_len"])
    val_ds = WindowDataset(val_texts, val_labels, tokenizer, cfg["max_seq_len"])
    test_ds = WindowDataset(test_texts, test_labels, tokenizer, cfg["max_seq_len"])

    # ── Oversample minority class via WeightedRandomSampler ───────────────────
    # WeightedRandomSampler with full inverse-frequency weighting (40:1) causes
    # the model to see the same ~480 boundary examples ~40x per epoch, leading
    # to memorization and val_f1 collapse after epoch 2.
    # Fix: cap the oversample ratio at 5x — enough signal without memorization.
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    max_oversample = cfg.get("max_oversample", 5.0)
    actual_ratio = n_neg / max(n_pos, 1)
    effective_ratio = min(actual_ratio, max_oversample)
    sample_weights = [1.0 if l == 0 else effective_ratio for l in train_labels]
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    log.info(f"Class imbalance: {n_pos} pos / {n_neg} neg → oversample ratio capped at {effective_ratio:.1f}x")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"] * 2, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"] * 2, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=2,
        hidden_dropout_prob=cfg["dropout"],
        attention_probs_dropout_prob=cfg["dropout"],
    )
    # Resize embeddings to accommodate [SPEAKER_X] tokens
    model.resize_token_embeddings(len(tokenizer))

    if cfg["freeze_backbone"]:
        # Decision: freeze all encoder layers, train classification head only.
        # This produces a faster-to-train, smaller-footprint model at the cost
        # of some task-specific representation quality. Good "fast" serving candidate.
        log.info("Freezing backbone — training classification head only")
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    model = model.to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_trainable:,}")

    # Standard loss — class balance handled by WeightedRandomSampler above
    loss_fn = torch.nn.CrossEntropyLoss()

    # ── Optimizer / scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    total_steps = len(train_loader) * cfg["epochs"]
    warmup_steps = int(cfg["warmup_ratio"] * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training loop ─────────────────────────────────────────────────────────

    run_name = (
        f"{cfg['model_name'].replace('/', '-')}"
        f"-{'frozen' if cfg['freeze_backbone'] else 'full'}"
        f"-lr{cfg['lr']}-bs{cfg['batch_size']}"
    )

    best_val_pk = float("inf")   # early stopping monitors Pk (lower = better)
    best_val_f1 = 0.0            # kept for logging only
    best_threshold = 0.5         # updated each epoch by threshold sweep
    prev_epoch_threshold = 0.5   # for smoothing
    patience_counter = 0
    best_model_path = os.path.join(cfg["output_dir"], f"{run_name}_best.pt")
    os.makedirs(cfg["output_dir"], exist_ok=True)

    total_train_start = time.time()

    with mlflow.start_run(run_name=run_name) as run:
        run_id_holder.append(run.info.run_id)

        # Log all config params
        mlflow.log_params(cfg)
        mlflow.log_params({"n_trainable_params": n_trainable})
        log_environment()
        mlflow.log_param("git_sha", os.environ.get("GIT_SHA", "unknown"))

        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            epoch_start = time.time()
            train_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

            epoch_time = time.time() - epoch_start

            # ── Validation with threshold sweep ──────────────────────────────
            model.eval()
            val_probs, val_true = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                    val_probs.extend(probs)
                    val_true.extend(batch["labels"].numpy())

            # Sweep thresholds — pick best by val Pk
            epoch_threshold, epoch_metrics, ref_metrics = sweep_thresholds(
                val_probs, val_true, val_meeting_ids)

            # Smooth threshold — don't allow jumps > 0.15 from the previous epoch.
            # Wild threshold jumps indicate the model's probability distribution
            # is unstable (overfitting symptom). Smoothing prevents the sweep
            # from chasing noise while still allowing gradual adaptation.
            if epoch > 1:
                max_jump = 0.15
                epoch_threshold = float(np.clip(
                    epoch_threshold,
                    prev_epoch_threshold - max_jump,
                    prev_epoch_threshold + max_jump,
                ))
                # Recompute metrics with the smoothed threshold
                smoothed_preds = (np.array(val_probs) >= epoch_threshold).astype(int)
                smoothed_seg = compute_segmentation_metrics(
                    val_true, smoothed_preds.tolist(), val_meeting_ids)
                epoch_metrics = {
                    "f1": f1_score(val_true, smoothed_preds, zero_division=0),
                    "precision": precision_score(val_true, smoothed_preds, zero_division=0),
                    "recall": recall_score(val_true, smoothed_preds, zero_division=0),
                    "pk": smoothed_seg.get("pk", 1.0),
                    "window_diff": smoothed_seg.get("window_diff", 1.0),
                    "n_predicted": int(smoothed_preds.sum()),
                    "n_true": int(np.array(val_true).sum()),
                }
            prev_epoch_threshold = epoch_threshold

            val_f1 = epoch_metrics["f1"]
            val_pk = epoch_metrics["pk"]

            epoch_log = {
                "train_loss": round(train_loss / len(train_loader), 4),
                "val_f1": round(val_f1, 4),
                "val_precision": round(epoch_metrics["precision"], 4),
                "val_recall": round(epoch_metrics["recall"], 4),
                "val_pk": round(val_pk, 4),
                "val_window_diff": round(epoch_metrics["window_diff"], 4),
                "val_best_threshold": epoch_threshold,
                "val_n_predicted": epoch_metrics["n_predicted"],
                "val_f1_at_0.5": round(ref_metrics["f1_at_0.5"], 4),
                "val_pk_at_0.5": round(ref_metrics["pk_at_0.5"], 4),
                "epoch_time_sec": round(epoch_time, 1),
            }
            mlflow.log_metrics(epoch_log, step=epoch)
            print(f"Epoch {epoch}/{cfg['epochs']} | "
                  f"loss={epoch_log['train_loss']:.4f} | "
                  f"f1={val_f1:.4f} (thr={epoch_threshold:.2f}) | "
                  f"pk={val_pk:.4f} | "
                  f"predicted={epoch_metrics['n_predicted']} true={epoch_metrics['n_true']} | "
                  f"time={epoch_time:.1f}s", flush=True)

            # ── Early stopping on Pk (lower is better) ────────────────────────
            if val_pk < best_val_pk:
                prev_best = best_val_pk
                best_val_pk = val_pk
                best_val_f1 = val_f1
                best_threshold = epoch_threshold
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"   New best val_pk: {val_pk:.4f} (improved from {prev_best:.4f}), "
                      f"threshold={best_threshold:.2f}, checkpoint saved", flush=True)
            else:
                # Don't count patience if model hasn't predicted any boundaries yet
                if epoch_metrics["n_predicted"] > 0:
                    patience_counter += 1
                    print(f"   val_pk did not improve ({val_pk:.4f} vs best {best_val_pk:.4f}), "
                          f"patience {patience_counter}/{cfg['early_stopping_patience']}", flush=True)
                else:
                    torch.save(model.state_dict(), best_model_path)
                    print(f"   No boundaries predicted yet, not counting patience", flush=True)
                if patience_counter >= cfg["early_stopping_patience"]:
                    print(f"   Early stopping triggered at epoch {epoch} ; "
                          f"val_pk did not improve for {patience_counter} epochs", flush=True)
                    break

        # ── Final test evaluation ─────────────────────────────────────────────
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        test_probs, test_true = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                test_probs.extend(probs)
                test_true.extend(batch["labels"].numpy())

        # Apply best threshold found on val set
        test_preds = (np.array(test_probs) >= best_threshold).astype(int)
        test_seg_metrics = compute_segmentation_metrics(
            test_true, test_preds.tolist(), test_meeting_ids)

        total_train_time = time.time() - total_train_start

        final_metrics = {
            "test_f1": round(f1_score(test_true, test_preds, zero_division=0), 4),
            "test_precision": round(precision_score(test_true, test_preds, zero_division=0), 4),
            "test_recall": round(recall_score(test_true, test_preds, zero_division=0), 4),
            "best_val_pk": round(best_val_pk, 4),
            "best_val_f1": round(best_val_f1, 4),
            "best_threshold": best_threshold,
            "total_training_time_sec": round(total_train_time, 1),
            "test_pk": round(test_seg_metrics.get("pk", -1.0), 4),
            "test_window_diff": round(test_seg_metrics.get("window_diff", -1.0), 4),
        }
        if torch.cuda.is_available():
            final_metrics["peak_vram_gb"] = round(
                torch.cuda.max_memory_allocated() / 1e9, 2)

        mlflow.log_metrics(final_metrics)
        mlflow.log_param("best_threshold", best_threshold)
        log.info(f"Test F1: {final_metrics['test_f1']:.4f} | "
                 f"Test Pk: {final_metrics['test_pk']:.4f} | "
                 f"Best threshold: {best_threshold:.2f}")

        # Log model + tokenizer as MLflow artifact
        mlflow.pytorch.log_model(model, artifact_path="model",
                                  pip_requirements=["transformers==4.40.0", "torch==2.2.0"])
        tokenizer.save_pretrained(os.path.join(cfg["output_dir"], "tokenizer"))
        mlflow.log_artifacts(os.path.join(cfg["output_dir"], "tokenizer"),
                              artifact_path="tokenizer")

    return final_metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    # Allow any config key to be overridden on CLI
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--freeze_backbone", action="store_true", default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--experiment_name", default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k not in ("config",)}
    cfg = load_config(args.config, overrides)

    # MLflow experiment
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:8000"))
    mlflow.set_experiment(cfg["experiment_name"])

    run_id_holder = []

    if cfg["model_name"] == "baseline":
        metrics = run_baseline(cfg, run_id_holder)
    else:
        metrics = run_roberta(cfg, run_id_holder)

    print(f"\nRun ID: {run_id_holder[0] if run_id_holder else 'N/A'}")
    print(f"Final metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
