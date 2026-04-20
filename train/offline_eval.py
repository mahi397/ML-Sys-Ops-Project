"""
offline_eval.py — Standalone offline evaluation for the jitsi-topic-segmenter

Evaluates any registered model version (or a local checkpoint) against the
held-out test set. Logs results to MLflow under the 'offline-evaluation'
experiment. Can be run:
  - After a retrain to double-check before promoting candidate → production
  - Against a new dataset version without retraining
  - On demand to audit any historical model version

Usage:
  # Evaluate the current production model on the default test set
  python offline_eval.py

  # Evaluate a specific model alias
  python offline_eval.py --model_alias candidate

  # Evaluate against a specific local data dir
  python offline_eval.py --data_dir /mnt/block/roberta_stage1/v2

  # Evaluate and compare against production
  python offline_eval.py --model_alias candidate --compare_to production
"""

import argparse
import json
import logging
import os
import time
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import mlflow
import mlflow.pytorch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "model_registry_name": "jitsi-topic-segmenter",
    "model_alias": "production",
    "data_dir": "/mnt/block/roberta_stage1/v2",
    "batch_size": 64,
    "max_seq_len": 128,
    "experiment_name": "offline-evaluation",
    # Gates (same as retrain.py)
    "gate_min_f1": 0.20,
    "gate_max_pk": 0.25,
    "gate_max_windowdiff": 0.40,
    "slice_gate_max_pk": 0.40,
}

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


# ── Data loading ─────────────────────────────────────────────────────────────

def format_window(window: list) -> str:
    parts = []
    for utt in sorted(window, key=lambda u: u["position"]):
        if utt.get("is_padding", False):
            continue
        if utt["text"] and utt["text"].strip():
            speaker = utt["speaker"] or "UNK"
            parts.append(f"[SPEAKER_{speaker}]: {utt['text']}")
    return " ".join(parts)


def load_split_with_metadata(data_dir: str, split: str):
    import json as _json
    path = os.path.join(data_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        log.warning(f"Split file not found: {path}")
        return [], [], [], []
    with open(path) as f:
        examples = [_json.loads(line) for line in f]
    texts = [format_window(e["input"]["window"]) for e in examples]
    labels = [e["output"]["label"] for e in examples]
    meeting_ids = [e["input"]["meeting_id"] for e in examples]
    n_speakers = [
        len(set(u["speaker"] for u in e["input"]["window"] if u["speaker"] is not None))
        for e in examples
    ]
    return texts, labels, meeting_ids, n_speakers


class WindowDataset(torch.utils.data.Dataset):
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
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ── Metrics ──────────────────────────────────────────────────────────────────

def _pk_single(ref, hyp, k):
    if len(ref) != len(hyp):
        return 1.0
    err, total = 0, 0
    for i in range(len(ref) - k):
        if (ref[i] != ref[i + k]) != (hyp[i] != hyp[i + k]):
            err += 1
        total += 1
    return err / total if total > 0 else 0.0


def _windowdiff_single(ref, hyp, k):
    if len(ref) != len(hyp):
        return 1.0
    err, total = 0, 0
    for i in range(len(ref) - k):
        if ref[i:i+k+1].count("1") != hyp[i:i+k+1].count("1"):
            err += 1
        total += 1
    return err / total if total > 0 else 0.0


def compute_segmentation_metrics(true_labels, pred_labels, meeting_ids) -> Dict:
    if not meeting_ids:
        return {"window_diff": -1.0, "pk": -1.0}
    meeting_true: Dict[str, List] = defaultdict(list)
    meeting_pred: Dict[str, List] = defaultdict(list)
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
            wd_scores.append(_windowdiff_single(ref, hyp, k=k))
            pk_scores.append(_pk_single(ref, hyp, k=k))
        except Exception:
            pass
    if not wd_scores:
        return {"window_diff": -1.0, "pk": -1.0}
    return {
        "window_diff": round(float(np.mean(wd_scores)), 4),
        "pk": round(float(np.mean(pk_scores)), 4),
    }


def sweep_thresholds(probs, true_labels, meeting_ids):
    probs = np.array(probs)
    true_labels = np.array(true_labels)
    best_pk, best_threshold, best_metrics = float("inf"), 0.5, {}
    for thr in THRESHOLDS:
        preds = (probs >= thr).astype(int)
        seg = compute_segmentation_metrics(true_labels.tolist(), preds.tolist(), meeting_ids)
        pk_val = seg.get("pk", 1.0)
        if pk_val < best_pk:
            best_pk = pk_val
            best_threshold = thr
            best_metrics = {
                "f1": f1_score(true_labels, preds, zero_division=0),
                "precision": precision_score(true_labels, preds, zero_division=0),
                "recall": recall_score(true_labels, preds, zero_division=0),
                "pk": pk_val,
                "window_diff": seg.get("window_diff", 1.0),
            }
    return best_threshold, best_metrics


def compute_slice_metrics(true_labels, test_probs, meeting_ids, n_speakers, threshold) -> Dict:
    preds = (np.array(test_probs) >= threshold).astype(int).tolist()
    meeting_size: Dict[str, int] = defaultdict(int)
    for mid in meeting_ids:
        meeting_size[mid] += 1

    def size_bucket(mid):
        n = meeting_size[mid]
        if n < 15:    return "short_lt15"
        elif n <= 40: return "medium_15to40"
        else:         return "long_gt40"

    def speaker_bucket(n):
        if n == 1:   return "single_speaker"
        elif n == 2: return "two_speaker"
        else:        return "multi_speaker_3plus"

    slices: Dict[str, Dict[str, List]] = defaultdict(
        lambda: {"true": [], "pred": [], "probs": [], "mids": []}
    )
    for t, p, prob, mid, ns in zip(true_labels, preds, test_probs, meeting_ids, n_speakers):
        for key in (size_bucket(mid), speaker_bucket(ns)):
            slices[key]["true"].append(t)
            slices[key]["pred"].append(p)
            slices[key]["probs"].append(prob)
            slices[key]["mids"].append(mid)

    results = {}
    for slice_name, data in slices.items():
        if len(data["true"]) < 10:
            results[slice_name] = {"pk": -1.0, "f1": -1.0, "note": "too_few_examples",
                                   "n_examples": len(data["true"])}
            continue
        seg = compute_segmentation_metrics(data["true"], data["pred"], data["mids"])
        results[slice_name] = {
            "pk": seg.get("pk", -1.0),
            "f1": round(f1_score(data["true"], data["pred"], zero_division=0), 4),
            "window_diff": seg.get("window_diff", -1.0),
            "n_examples": len(data["true"]),
            "n_meetings": len(set(data["mids"])),
        }
        log.info(f"  [{slice_name:25s}] Pk={results[slice_name]['pk']:.4f}  "
                 f"F1={results[slice_name]['f1']:.4f}")
    return results


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_from_registry(registry_name: str, alias: str, model_name: str = "roberta-base"):
    """Load model and tokenizer from MLflow registry by alias."""
    log.info(f"Loading model from registry: {registry_name}@{alias}")
    model_uri = f"models:/{registry_name}@{alias}"

    # Load model
    model = mlflow.pytorch.load_model(model_uri)

    # Load tokenizer — try from registry artifact first, fall back to base
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = [f"[SPEAKER_{s}]" for s in "ABCDEFGH"] + ["[SPEAKER_X]", "[SPEAKER_Y]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Try to get best_threshold from model version tag
    client = mlflow.tracking.MlflowClient()
    best_threshold = 0.5
    try:
        versions = client.get_model_version_by_alias(registry_name, alias)
        threshold_tag = versions.tags.get("best_threshold")
        if threshold_tag:
            best_threshold = float(threshold_tag)
            log.info(f"Using best_threshold={best_threshold} from model registry tag")
    except Exception as e:
        log.warning(f"Could not read best_threshold tag: {e} — using 0.5")

    return model, tokenizer, best_threshold


# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(model, tokenizer, texts, labels, cfg: Dict, device):
    ds = WindowDataset(texts, labels, tokenizer, cfg["max_seq_len"])
    loader = DataLoader(ds, batch_size=cfg["batch_size"] * 2, num_workers=0)
    probs_list, true_list = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            probs = torch.softmax(
                model(input_ids=batch["input_ids"].to(device),
                      attention_mask=batch["attention_mask"].to(device)).logits,
                dim=-1
            )[:, 1].cpu().numpy()
            probs_list.extend(probs)
            true_list.extend(batch["labels"].numpy())
    return probs_list, true_list


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(cfg: Dict, compare_alias: Optional[str] = None) -> Dict:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(cfg["experiment_name"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Load model
    model, tokenizer, best_threshold = load_model_from_registry(
        cfg["model_registry_name"], cfg["model_alias"]
    )
    model = model.to(device).eval()

    # Load test data
    log.info(f"Loading test data from {cfg['data_dir']}")
    test_texts, test_labels, test_meeting_ids, test_n_speakers = \
        load_split_with_metadata(cfg["data_dir"], "test")

    if not test_texts:
        log.error(f"No test data found at {cfg['data_dir']}")
        return {}

    log.info(f"Test set: {len(test_texts)} examples, {len(set(test_meeting_ids))} meetings")

    # Run inference
    log.info("Running inference on test set...")
    test_probs, test_true = run_inference(model, tokenizer, test_texts, test_labels, cfg, device)

    # Sweep thresholds
    best_threshold_swept, best_metrics = sweep_thresholds(test_probs, test_true, test_meeting_ids)
    log.info(f"Best threshold (val sweep): {best_threshold}")
    log.info(f"Best threshold (test sweep): {best_threshold_swept}")

    # Use registry threshold for final metrics
    test_preds = (np.array(test_probs) >= best_threshold).astype(int)
    test_seg = compute_segmentation_metrics(test_true, test_preds.tolist(), test_meeting_ids)
    test_f1  = f1_score(test_true, test_preds, zero_division=0)
    test_pk  = test_seg.get("pk", 1.0)
    test_wd  = test_seg.get("window_diff", 1.0)

    log.info(f"Aggregate — F1={test_f1:.4f}, Pk={test_pk:.4f}, WD={test_wd:.4f}")

    # Gates
    agg_passed = (
        test_f1 >= cfg["gate_min_f1"]
        and test_pk <= cfg["gate_max_pk"]
        and test_wd <= cfg["gate_max_windowdiff"]
    )
    log.info(f"Aggregate gates: {'PASS' if agg_passed else 'FAIL'}")

    # Slice evaluation
    log.info("Running slice evaluation...")
    slice_metrics = compute_slice_metrics(
        test_true, test_probs, test_meeting_ids, test_n_speakers, best_threshold
    )
    fairness_failures = [
        f"{k}: Pk={v['pk']:.4f}" for k, v in slice_metrics.items()
        if v.get("note") != "too_few_examples" and v.get("pk", -1) > cfg["slice_gate_max_pk"]
    ]
    fairness_passed = len(fairness_failures) == 0
    gates_passed = agg_passed and fairness_passed

    log.info(f"Fairness gate: {'PASS' if fairness_passed else 'FAIL'}")
    if fairness_failures:
        log.warning(f"Fairness failures: {fairness_failures}")

    # Speaker relabeling invariance test
    log.info("Running speaker relabeling invariance test...")
    relabeled_texts = [
        t.replace("[SPEAKER_A]", "[SPEAKER_X]").replace("[SPEAKER_B]", "[SPEAKER_Y]")
        for t in test_texts
    ]
    relabel_probs, _ = run_inference(model, tokenizer, relabeled_texts, test_labels, cfg, device)
    relabel_preds = (np.array(relabel_probs) >= best_threshold).astype(int)
    relabel_seg = compute_segmentation_metrics(test_true, relabel_preds.tolist(), test_meeting_ids)
    relabel_pk = relabel_seg.get("pk", 1.0)
    relabel_warn = relabel_pk > 0.30
    log.info(f"Speaker relabeling Pk={relabel_pk:.4f} {'WARN' if relabel_warn else 'OK'}")

    # Log to MLflow
    run_name = f"offline-eval-{cfg['model_alias']}-{int(time.time())}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "model_alias": cfg["model_alias"],
            "model_registry": cfg["model_registry_name"],
            "data_dir": cfg["data_dir"],
            "best_threshold": best_threshold,
            "n_test_examples": len(test_texts),
            "n_test_meetings": len(set(test_meeting_ids)),
        })
        mlflow.log_metrics({
            "test_f1": round(test_f1, 4),
            "test_pk": round(test_pk, 4),
            "test_window_diff": round(test_wd, 4),
            "test_precision": round(precision_score(test_true, test_preds, zero_division=0), 4),
            "test_recall": round(recall_score(test_true, test_preds, zero_division=0), 4),
            "gates_passed": int(gates_passed),
            "fairness_gate_passed": int(fairness_passed),
            "fm_pk_speaker_relabel_invariance": round(relabel_pk, 4),
        })
        for slice_name, sm in slice_metrics.items():
            if sm.get("pk", -1) >= 0:
                mlflow.log_metric(f"slice_pk_{slice_name}", sm["pk"])
            if sm.get("f1", -1) >= 0:
                mlflow.log_metric(f"slice_f1_{slice_name}", sm["f1"])

        # Log summary as artifact
        summary = {
            "model_alias": cfg["model_alias"],
            "model_registry": cfg["model_registry_name"],
            "data_dir": cfg["data_dir"],
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "aggregate": {
                "f1": round(test_f1, 4),
                "pk": round(test_pk, 4),
                "window_diff": round(test_wd, 4),
                "gates_passed": gates_passed,
            },
            "fairness": {
                "gate_passed": fairness_passed,
                "failures": fairness_failures,
                "slices": slice_metrics,
            },
            "robustness": {
                "speaker_relabel_pk": round(relabel_pk, 4),
                "speaker_relabel_warn": relabel_warn,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = os.path.join(tmpdir, "eval_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            mlflow.log_artifact(summary_path)

        log.info(f"MLflow run: {run.info.run_id}")
        log.info(f"View at: {mlflow.get_tracking_uri()}/#/experiments/"
                 f"{run.info.experiment_id}/runs/{run.info.run_id}")

    log.info("=" * 60)
    log.info(f"OFFLINE EVAL RESULT: {'PASS' if gates_passed else 'FAIL'}")
    log.info(f"  F1={test_f1:.4f}  Pk={test_pk:.4f}  WD={test_wd:.4f}")
    log.info(f"  Fairness: {'PASS' if fairness_passed else 'FAIL'}")
    log.info(f"  Speaker relabeling Pk={relabel_pk:.4f} {'(WARN)' if relabel_warn else '(OK)'}")
    log.info("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Offline evaluation for jitsi-topic-segmenter")
    parser.add_argument("--model_alias", default="production",
                        help="MLflow model alias to evaluate (default: production)")
    parser.add_argument("--data_dir", default=None,
                        help="Local path to test data dir (default: from config)")
    parser.add_argument("--compare_to", default=None,
                        help="Also evaluate this alias and print a comparison")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg["model_alias"] = args.model_alias
    if args.data_dir:
        cfg["data_dir"] = args.data_dir

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    result = evaluate(cfg)

    if args.compare_to:
        log.info(f"\nComparing against {args.compare_to}...")
        cfg_compare = cfg.copy()
        cfg_compare["model_alias"] = args.compare_to
        result_compare = evaluate(cfg_compare)

        log.info("\n" + "=" * 60)
        log.info("COMPARISON")
        log.info("=" * 60)
        log.info(f"{'Metric':<20} {args.model_alias:>15} {args.compare_to:>15}")
        log.info("-" * 50)
        for metric in ["f1", "pk", "window_diff"]:
            v1 = result.get("aggregate", {}).get(metric, "N/A")
            v2 = result_compare.get("aggregate", {}).get(metric, "N/A")
            log.info(f"{metric:<20} {str(v1):>15} {str(v2):>15}")

    return 0 if result.get("aggregate", {}).get("gates_passed") else 1


if __name__ == "__main__":
    exit(main())
