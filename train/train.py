"""
train.py  —  Single configurable training script for topic boundary detection.

All candidates (baseline, roberta-base frozen, roberta-base full, distilroberta)
are selected via config. No one-off scripts.

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

import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pytorch

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
    """Load a JSONL split and return (texts, labels)."""
    examples = load_jsonl(os.path.join(data_dir, f"{split}.jsonl"))
    texts = [format_window(e["window"]) for e in examples]
    labels = [e["label"] for e in examples]
    return texts, labels


# ── Baseline: TF-IDF + Logistic Regression ────────────────────────────────────

def run_baseline(cfg: Dict, run_id_holder: list):
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_score, recall_score

    log.info("Running TF-IDF + Logistic Regression baseline")
    train_texts, train_labels = load_split(cfg["data_dir"], "train")
    val_texts, val_labels = load_split(cfg["data_dir"], "val")
    test_texts, test_labels = load_split(cfg["data_dir"], "test")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, C=cfg.get("C", 1.0),
                                   class_weight="balanced")),
    ])

    t0 = time.time()
    pipe.fit(train_texts, train_labels)
    train_time = time.time() - t0

    val_preds = pipe.predict(val_texts)
    test_preds = pipe.predict(test_texts)

    metrics = {
        "val_f1": f1_score(val_labels, val_preds, zero_division=0),
        "val_precision": precision_score(val_labels, val_preds, zero_division=0),
        "val_recall": recall_score(val_labels, val_preds, zero_division=0),
        "test_f1": f1_score(test_labels, test_preds, zero_division=0),
        "test_precision": precision_score(test_labels, test_preds, zero_division=0),
        "test_recall": recall_score(test_labels, test_preds, zero_division=0),
        "total_training_time_sec": train_time,
        "gpu_used": 0,
    }
    metrics.update(compute_segmentation_metrics(val_labels, val_preds))

    with mlflow.start_run(run_name="baseline-tfidf-lr") as run:
        run_id_holder.append(run.info.run_id)
        mlflow.log_params({k: v for k, v in cfg.items()
                           if k in ("model_name", "epochs", "seed")})
        mlflow.log_params({"C": cfg.get("C", 1.0), "ngram_range": "1,2",
                            "max_features": 50000})
        log_environment()
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipe, artifact_path="model")
        log.info(f"Baseline val F1: {metrics['val_f1']:.4f}")

    return metrics


# ── Segmentation metrics (WindowDiff and Pk) ──────────────────────────────────
# These are standard topic segmentation metrics beyond binary F1.
# WindowDiff penalizes off-by-one boundary placement, Pk measures probability
# of misclassifying a pair of utterances as in the same/different segment.

def compute_segmentation_metrics(true_labels, pred_labels, k=None):
    """
    Compute WindowDiff and Pk for predicted boundary sequences.
    Both operate on the full sequence of 0/1 labels for a meeting.
    Here we compute corpus-level averages across all windows.
    Lower is better for both.
    """
    try:
        import nltk
        nltk.download("punkt", quiet=True)
        from nltk.metrics.segmentation import windowdiff, pk
    except ImportError:
        log.warning("nltk not installed — skipping WindowDiff/Pk. pip install nltk")
        return {"window_diff": -1.0, "pk": -1.0}

    # Convert list of labels to boundary strings for nltk ("0"/"1")
    ref = "".join(str(l) for l in true_labels)
    hyp = "".join(str(l) for l in pred_labels)
    if k is None:
        k = max(2, len(ref) // 10)

    try:
        wd = windowdiff(ref, hyp, k=k, boundary="1")
        p = pk(ref, hyp, k=k, boundary="1")
        return {"window_diff": round(wd, 4), "pk": round(p, 4)}
    except Exception as e:
        log.warning(f"Segmentation metric error: {e}")
        return {"window_diff": -1.0, "pk": -1.0}


# ── Environment logging ────────────────────────────────────────────────────────

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
    from sklearn.metrics import f1_score, precision_score, recall_score

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
    train_texts, train_labels = load_split(cfg["data_dir"], "train")
    val_texts, val_labels = load_split(cfg["data_dir"], "val")
    test_texts, test_labels = load_split(cfg["data_dir"], "test")

    train_ds = WindowDataset(train_texts, train_labels, tokenizer, cfg["max_seq_len"])
    val_ds = WindowDataset(val_texts, val_labels, tokenizer, cfg["max_seq_len"])
    test_ds = WindowDataset(test_texts, test_labels, tokenizer, cfg["max_seq_len"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"] * 2, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"] * 2, num_workers=4)

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

    # ── Class weights for imbalanced data ─────────────────────────────────────
    # AMI corpus is ~2.4% boundary / 97.6% non-boundary.
    # Without weighting, the model learns to predict "no boundary" always
    # and gets high accuracy but near-zero F1 on the boundary class.
    # Weight = n_negative / n_positive ≈ 40x, capped at 20 to avoid instability.
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = min(n_neg / max(n_pos, 1), 20.0)
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    log.info(f"Class imbalance: {n_pos} pos / {n_neg} neg → pos_weight={pos_weight:.1f}")

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

    best_val_f1 = 0.0
    patience_counter = 0
    best_model_path = os.path.join(cfg["output_dir"], f"{run_name}_best.pt")
    os.makedirs(cfg["output_dir"], exist_ok=True)

    total_train_start = time.time()

    with mlflow.start_run(run_name=run_name) as run:
        run_id_holder.append(run.info.run_id)

        # Log all config params
        mlflow.log_params(cfg)
        mlflow.log_params({"n_trainable_params": n_trainable, "pos_weight": round(pos_weight, 2)})
        log_environment()

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

            # ── Validation ───────────────────────────────────────────────────
            model.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    preds = logits.argmax(dim=-1).cpu().numpy()
                    val_preds.extend(preds)
                    val_true.extend(batch["labels"].numpy())

            val_f1 = f1_score(val_true, val_preds, zero_division=0)
            val_prec = precision_score(val_true, val_preds, zero_division=0)
            val_rec = recall_score(val_true, val_preds, zero_division=0)
            seg_metrics = compute_segmentation_metrics(val_true, val_preds)

            epoch_metrics = {
                "train_loss": round(train_loss / len(train_loader), 4),
                "val_f1": round(val_f1, 4),
                "val_precision": round(val_prec, 4),
                "val_recall": round(val_rec, 4),
                "epoch_time_sec": round(epoch_time, 1),
                **{f"val_{k}": v for k, v in seg_metrics.items()},
            }
            mlflow.log_metrics(epoch_metrics, step=epoch)
            log.info(f"Epoch {epoch}/{cfg['epochs']} | "
                     f"loss={epoch_metrics['train_loss']:.4f} | "
                     f"val_f1={val_f1:.4f} | "
                     f"time={epoch_time:.1f}s")

            # Early stopping + checkpoint
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                log.info(f"  → New best val F1 {best_val_f1:.4f}, checkpoint saved")
            else:
                patience_counter += 1
                if patience_counter >= cfg["early_stopping_patience"]:
                    log.info(f"Early stopping at epoch {epoch}")
                    break

        # ── Final test evaluation ─────────────────────────────────────────────
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        test_preds, test_true, test_probs = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                test_preds.extend(preds)
                test_true.extend(batch["labels"].numpy())
                test_probs.extend(probs)

        total_train_time = time.time() - total_train_start
        test_seg_metrics = compute_segmentation_metrics(test_true, test_preds)

        final_metrics = {
            "test_f1": round(f1_score(test_true, test_preds, zero_division=0), 4),
            "test_precision": round(precision_score(test_true, test_preds, zero_division=0), 4),
            "test_recall": round(recall_score(test_true, test_preds, zero_division=0), 4),
            "best_val_f1": round(best_val_f1, 4),
            "total_training_time_sec": round(total_train_time, 1),
            **{f"test_{k}": v for k, v in test_seg_metrics.items()},
        }
        if torch.cuda.is_available():
            final_metrics["peak_vram_gb"] = round(
                torch.cuda.max_memory_allocated() / 1e9, 2)

        mlflow.log_metrics(final_metrics)
        log.info(f"Test F1: {final_metrics['test_f1']:.4f} | "
                 f"WindowDiff: {final_metrics.get('test_window_diff', '?')}")

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
