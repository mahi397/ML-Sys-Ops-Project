"""
train_ray.py  —  Ray Train wrapper for fault-tolerant RoBERTa fine-tuning.

hows Ray Train making training more robust than plain train.py
by automatically resuming from checkpoints after worker failure.

Usage:
  # Install Ray
  pip install "ray[train]==2.10.0" --break-system-packages

  # Start a single-node Ray cluster
  ray start --head --num-gpus=1

  export MLFLOW_TRACKING_URI=http://129.114.25.90:8000

  # Run (and kill after epoch 1 to demo fault tolerance)
  python train_ray.py \
    --config /home/cc/ML-Sys-Ops-Project/train/configs/roberta_base_full.yaml \
    --data_dir /home/cc/ami_processed \
    --storage_path /home/cc/artifacts/ray_checkpoints

  # Rerun same command — resumes from checkpoint
  python train_ray.py \
    --config /home/cc/ML-Sys-Ops-Project/train/configs/roberta_base_full.yaml \
    --data_dir /home/cc/ami_processed \
    --storage_path /home/cc/artifacts/ray_checkpoints
"""

import argparse
import json
import os
import sys
import time
import yaml
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def format_window(window):
    parts = []
    for utt in sorted(window, key=lambda u: u["position"]):
        if utt["text"].strip():
            parts.append(f"[SPEAKER_{utt['speaker']}]: {utt['text']}")
    return " ".join(parts)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def load_split(data_dir, split):
    examples = load_jsonl(os.path.join(data_dir, f"{split}.jsonl"))
    texts = [format_window(e["window"]) for e in examples]
    labels = [e["label"] for e in examples]
    return texts, labels


def train_func(config):
    """
    This function runs inside a Ray worker.
    Ray handles:
      - Checkpointing to persistent storage after each epoch
      - Resuming from last checkpoint if worker is killed and restarted
      - Fault tolerance: max_failures=3 means Ray retries up to 3 times
    """
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    from sklearn.metrics import f1_score
    import ray.train

    device = ray.train.torch.get_device()
    data_dir = config["data_dir"]

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    special_tokens = [f"[SPEAKER_{s}]" for s in "ABCDEFGH"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    class WindowDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.encodings = tokenizer(
                texts, truncation=True, padding="max_length",
                max_length=max_len, return_tensors="pt")
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self): return len(self.labels)
        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.labels[idx],
            }

    train_texts, train_labels = load_split(data_dir, "train")
    val_texts, val_labels = load_split(data_dir, "val")

    # WeightedRandomSampler for class imbalance
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    max_oversample = config.get("max_oversample", 5.0)
    effective_ratio = min(n_neg / max(n_pos, 1), max_oversample)
    sample_weights = [1.0 if l == 0 else effective_ratio for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_ds = WindowDataset(train_texts, train_labels, tokenizer, config["max_seq_len"])
    val_ds = WindowDataset(val_texts, val_labels, tokenizer, config["max_seq_len"])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"] * 2, num_workers=2)

    # Prepare loaders for distributed training
    train_loader = ray.train.torch.prepare_data_loader(train_loader)
    val_loader = ray.train.torch.prepare_data_loader(val_loader)

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=2,
        hidden_dropout_prob=config.get("dropout", 0.2),
        attention_probs_dropout_prob=config.get("dropout", 0.2),
    )
    model.resize_token_embeddings(len(tokenizer))

    # Prepare model for distributed training
    model = ray.train.torch.prepare_model(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.05),
    )
    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(config.get("warmup_ratio", 0.1) * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    # ── Resume from checkpoint if one exists ─────────────────────────────────
    # This is the key fault-tolerance feature.
    # If this worker was killed after epoch N, Ray finds the checkpoint
    # saved at epoch N and resumes here — skipping epochs 1..N entirely.
    start_epoch = 1
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as ckpt_dir:
            ckpt = torch.load(os.path.join(ckpt_dir, "state.pt"), map_location=device)
            model.module.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed from checkpoint at epoch {ckpt['epoch']} "
                  f"— continuing from epoch {start_epoch}", flush=True)
    else:
        print("No checkpoint found — starting from epoch 1", flush=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, config["epochs"] + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                preds = (probs >= 0.3).astype(int)
                val_preds.extend(preds)
                val_true.extend(batch["labels"].cpu().numpy())

        val_f1 = f1_score(val_true, val_preds, zero_division=0)
        avg_loss = train_loss / len(train_loader)
        epoch_time = time.time() - t0

        print(f"Epoch {epoch}/{config['epochs']} | loss={avg_loss:.4f} | "
              f"val_f1={val_f1:.4f} | time={epoch_time:.1f}s", flush=True)

        # ── Save checkpoint to persistent storage after every epoch ───────────
        # Even if the worker is killed immediately after this line,
        # the next worker will find this checkpoint and resume from epoch+1.
        ckpt_data = {
            "epoch": epoch,
            "model": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }

        # Write checkpoint to a temp dir then pass to Ray
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            torch.save(ckpt_data, os.path.join(tmp, "state.pt"))
            checkpoint = ray.train.Checkpoint.from_directory(tmp)
            ray.train.report(
                metrics={
                    "epoch": epoch,
                    "train_loss": round(avg_loss, 4),
                    "val_f1": round(val_f1, 4),
                },
                checkpoint=checkpoint,
            )
        print(f"  Checkpoint saved after epoch {epoch} "
              f"(kill the job now to demo fault tolerance)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", default="/home/cc/ami_processed")
    parser.add_argument("--storage_path", default="/home/cc/artifacts/ray_checkpoints")
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["data_dir"] = args.data_dir

    # Ensure numeric types
    cfg["lr"] = float(cfg["lr"])
    cfg["batch_size"] = int(cfg["batch_size"])
    cfg["epochs"] = int(cfg.get("epochs", 8))
    cfg["max_seq_len"] = int(cfg.get("max_seq_len", 256))

    import ray
    import ray.train
    from ray.train import RunConfig, ScalingConfig, FailureConfig
    from ray.train.torch import TorchTrainer

    # Local mode — Ray starts its own single-node cluster inside the container.
    # No separate `ray start --head` needed.
    ray.init(ignore_reinit_error=True)
    log.info(f"Ray initialized: {ray.cluster_resources()}")

    trainer = TorchTrainer(
        train_func,
        train_loop_config=cfg,
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=True,
            resources_per_worker={"GPU": 1, "CPU": 2},
        ),
        run_config=RunConfig(
            name="jitsi-roberta-fault-tolerant",
            storage_path=args.storage_path,
            # Key: automatically retry up to 3 times on worker failure
            # Each retry loads the last saved checkpoint
            failure_config=FailureConfig(max_failures=3),
        ),
    )

    log.info("Starting Ray Train job — kill after epoch 1 checkpoint to demo fault tolerance")
    result = trainer.fit()
    print(f"\nFinal metrics: {result.metrics}", flush=True)
    print(f"Checkpoint path: {result.checkpoint}", flush=True)


if __name__ == "__main__":
    main()
