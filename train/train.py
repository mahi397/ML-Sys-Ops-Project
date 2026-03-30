import argparse
import json
import os
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score
from nltk.metrics.segmentation import pk, windowdiff
from datetime import datetime


# ---------- args ----------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--model_name", default="roberta-base")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--k", type=int, default=7)
    p.add_argument("--pos_weight", type=float, default=20.0)
    p.add_argument("--freeze_encoder", action="store_true", default=False)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--output_dir", default="checkpoints")
    return p.parse_args()


# ---------- dataset ----------

class WindowDataset(Dataset):
    def __init__(self, path, tokenizer, max_length, k=7):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.k = k
        self.samples = []
        with open(path) as f:
            for line in f:
                self.samples.append(json.loads(line))

    def format_window(self, window):
    """
    For any k, take k utterances centered on transition_index.
    For k=7: positions 0-6 (3 before, transition, 3 after)
    For k=5: positions 1-5 (2 before, transition, 2 after)
    For k=3: positions 2-4 (1 before, transition, 1 after)
    """
    sorted_utts = sorted(window, key=lambda x: x["position"])
    total = len(sorted_utts)  # always 7 in the JSON
    center = total // 2       # always 3

    half = self.k // 2
    start = center - half
    end = center + half + (self.k % 2)  # handles odd k correctly
    selected = sorted_utts[start:end]

    parts = []
    for utt in selected:
        if not utt["text"]:
            continue
        if utt["speaker"]:
            speaker_tag = f"[SPEAKER_{utt['speaker']}]:"
        else:
            speaker_tag = "[PAD]:"
        parts.append(f"{speaker_tag} {utt['text']}")
    return " ".join(parts)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = self.format_window(sample["window"])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "meeting_id": sample["meeting_id"]
        }


# ---------- segmentation metrics ----------

def compute_seg_metrics(labels, preds, meeting_ids):
    meetings = {}
    for mid, label, pred in zip(meeting_ids, labels, preds):
        if mid not in meetings:
            meetings[mid] = {"labels": [], "preds": []}
        meetings[mid]["labels"].append(label)
        meetings[mid]["preds"].append(pred)

    pk_scores, wd_scores = [], []
    for mid, data in meetings.items():
        ref = "".join(str(l) for l in data["labels"])
        hyp = "".join(str(p) for p in data["preds"])
        if len(ref) < 2:
            continue
        k_val = max(2, len(ref) // 10)
        try:
            pk_scores.append(pk(ref, hyp, k=k_val))
            wd_scores.append(windowdiff(ref, hyp, k=k_val))
        except Exception:
            pass

    return {
        "pk": float(np.mean(pk_scores)) if pk_scores else 1.0,
        "windowdiff": float(np.mean(wd_scores)) if wd_scores else 1.0
    }


# ---------- train ----------

def train_epoch(model, loader, optimizer, scheduler, device, pos_weight):
    model.train()
    total_loss = 0
    weight = torch.tensor([1.0, pos_weight], device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ---------- evaluate ----------

def evaluate(model, loader, device):
    model.eval()
    all_labels, all_mids, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            all_mids.extend(batch["meeting_id"])

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # sweep thresholds to find best F1
    best_f1, best_threshold = 0.0, 0.5
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        preds = (all_probs >= threshold).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    best_preds = (all_probs >= best_threshold).astype(int)
    seg_metrics = compute_seg_metrics(
        all_labels.tolist(), best_preds.tolist(), all_mids)

    preds_05 = (all_probs >= 0.5).astype(int)
    f1_05 = f1_score(all_labels, preds_05, zero_division=0)

    return {
        "f1": float(f1_score(all_labels, best_preds, zero_division=0)),
        "precision": float(precision_score(all_labels, best_preds,
                                           zero_division=0)),
        "recall": float(recall_score(all_labels, best_preds,
                                     zero_division=0)),
        "best_threshold": best_threshold,
        "f1_at_0.5": float(f1_05),
        "n_predicted_boundaries": int(best_preds.sum()),
        "n_true_boundaries": int(all_labels.sum()),
        "pk": seg_metrics["pk"],
        "windowdiff": seg_metrics["windowdiff"]
    }


# ---------- save run ----------

def save_run(args, metrics_per_epoch, best_pk):
    run = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "params": vars(args),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available()
               else "cpu",
        "epochs": metrics_per_epoch,
        "best_val_pk": best_pk
    }
    os.makedirs("runs", exist_ok=True)
    path = f"runs/run_{run['run_id']}.json"
    with open(path, "w") as f:
        json.dump(run, f, indent=2)
    print(f"Run saved to {path}")
    return path


# ---------- main ----------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    special_tokens = [f"[SPEAKER_{s}]:" for s in ["A", "B", "C", "D", "E"]]
    special_tokens.append("[PAD]:")
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    train_ds = WindowDataset(
        f"{args.data_dir}/train.jsonl", tokenizer, args.max_length, args.k)
    val_ds = WindowDataset(
        f"{args.data_dir}/val.jsonl", tokenizer, args.max_length, args.k)

    n_train_pos = sum(s["label"] for s in train_ds.samples)
    n_val_pos = sum(s["label"] for s in val_ds.samples)
    imbalance_ratio = (len(train_ds) - n_train_pos) / max(n_train_pos, 1)

    print(f"Train: {len(train_ds)} samples | "
          f"{n_train_pos} boundaries ({100*n_train_pos/len(train_ds):.1f}%)")
    print(f"Val:   {len(val_ds)} samples | "
          f"{n_val_pos} boundaries ({100*n_val_pos/len(val_ds):.1f}%)")
    print(f"pos_weight: {args.pos_weight} | "
          f"Imbalance ratio: {imbalance_ratio:.1f}x")

    if args.pos_weight < imbalance_ratio * 0.3:
        print(f"WARNING: pos_weight may be too low for {imbalance_ratio:.1f}x "
              f"imbalance. Consider --pos_weight {imbalance_ratio:.0f}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=2)

    # load model with dropout to reduce overfitting
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
        classifier_dropout=0.3
    )
    model.resize_token_embeddings(len(tokenizer))

    # optionally freeze encoder — useful when positive examples are scarce
    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Encoder frozen. Trainable: {trainable:,} / {total:,} "
              f"({100*trainable/total:.1f}%)")
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"Full fine-tune. Total params: {total:,}")

    model = model.to(device)

    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    metrics_per_epoch = []
    best_pk = 1.0
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, args.pos_weight)
        val_metrics = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0

        row = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_f1": round(val_metrics["f1"], 4),
            "val_precision": round(val_metrics["precision"], 4),
            "val_recall": round(val_metrics["recall"], 4),
            "val_f1_at_0.5": round(val_metrics["f1_at_0.5"], 4),
            "best_threshold": val_metrics["best_threshold"],
            "n_predicted_boundaries": val_metrics["n_predicted_boundaries"],
            "n_true_boundaries": val_metrics["n_true_boundaries"],
            "val_pk": round(val_metrics["pk"], 4),
            "val_windowdiff": round(val_metrics["windowdiff"], 4),
            "epoch_time_seconds": round(epoch_time, 1)
        }
        metrics_per_epoch.append(row)

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"loss={train_loss:.4f} | "
              f"f1={val_metrics['f1']:.3f} "
              f"(thr={val_metrics['best_threshold']:.2f}) | "
              f"precision={val_metrics['precision']:.3f} | "
              f"recall={val_metrics['recall']:.3f} | "
              f"predicted={val_metrics['n_predicted_boundaries']} "
              f"true={val_metrics['n_true_boundaries']} | "
              f"pk={val_metrics['pk']:.3f} | "
              f"wd={val_metrics['windowdiff']:.3f} | "
              f"time={epoch_time:.0f}s")

        if val_metrics["pk"] < best_pk:
            best_pk = val_metrics["pk"]
            epochs_without_improvement = 0
            model.save_pretrained(f"{args.output_dir}/best_model")
            tokenizer.save_pretrained(f"{args.output_dir}/best_model")
            print(f"  → saved best model (pk={best_pk:.3f})")
        else:
            epochs_without_improvement += 1
            print(f"  → no improvement "
                  f"({epochs_without_improvement}/{args.patience})")
            if epochs_without_improvement >= args.patience:
                print(f"  → early stopping at epoch {epoch+1}")
                break

    save_run(args, metrics_per_epoch, best_pk)
    print(f"\nDone. Best val Pk: {best_pk:.3f}")
    print(f"Target to beat (unsupervised BERT baseline on AMI): Pk=0.331")


if __name__ == "__main__":
    main()
