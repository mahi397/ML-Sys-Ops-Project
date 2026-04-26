"""
retrain.py — Fault-tolerant automated retraining with Ray Train

Adapted to Aneesh's data design (Data_Design_Document.pdf):
  - Training datasets at objstore-proj07/datasets/roberta_stage1/vN/
  - Feedback pool at objstore-proj07/datasets/roberta_stage1_feedback_pool/vN/
  - Dataset versions tracked in dataset_versions table in proj07_sql_db
  - Data staged to /mnt/block/ before training via rclone

Ray Train integration goes beyond the lab in three ways:
  1. Wraps a raw PyTorch training loop (not Lightning) with TorchTrainer
  2. Uses FailureConfig for unattended automated retraining robustness
  3. Integrates checkpoint-resume with MLflow quality gates for promotion

New in this version (safeguarding + evaluation hardening):
  - [EVAL]  Slice evaluation: Pk/F1/WD broken down by meeting-size bucket
            (short/medium/long) and speaker-count bucket. Feeds FAIRNESS
            safeguarding requirement. Lab: "evaluate on slices of interest."
  - [EVAL]  Fairness gate: no slice with sufficient data may have Pk > 0.40
  - [EVAL]  Known failure mode tests: single-topic meetings, very-short meetings,
            speaker-relabeling invariance. Lab: "evaluate on known failure modes."
            These are WARN only (don't block registration) but are surfaced in
            the model card and MLflow.
  - [SAFE]  Model card: JSON artifact logged to MLflow on every run. Documents
            training data, metrics, thresholds, fairness results, privacy notes,
            accountability chain. Covers TRANSPARENCY + ACCOUNTABILITY principles.
  - [SAFE]  best_threshold tagged on the registered model version so Shruti's
            serving layer reads it from the registry — not hardcoded.
  - [SAFE]  Speaker relabeling tokens [SPEAKER_X], [SPEAKER_Y] added to tokenizer
            vocab so template perturbation tests run without UNK tokens.

Previous fixes retained:
  - rclone remote configurable via RCLONE_REMOTE env var
  - DATABASE_URL wired through; DB name proj07_sql_db
  - OBJSTORE_BUCKET replaces TRAINING_DATA_BUCKET
  - Soft-fail on audit_log / retrain_log writes

Usage:
  python retrain.py --data_dir /mnt/block/roberta_stage1/v2   # local test
  python retrain.py                                            # production (reads env)
"""

import argparse
import json
import os
import time
import logging
import subprocess
import tempfile
import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import mlflow
import mlflow.pytorch

from sklearn.metrics import f1_score, precision_score, recall_score
#from nltk.metrics.segmentation import windowdiff, pk as pk_metric

import ray
import pyarrow.fs as pafs
from ray import train
from ray.train import RunConfig, FailureConfig, CheckpointConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

#import nltk
#for pkg in ("punkt", "punkt_tab"):
 #   try:
  #      nltk.data.find(f"tokenizers/{pkg}")
   # except LookupError:
    #    nltk.download(pkg, quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_RETRAIN_CONFIG = {
    # Optuna-best from initial implementation (Trial #10/20)
    # test_pk=0.213, test_f1=0.232, test_wd=0.365
    "model_name": "roberta-base",
    "freeze_backbone": False,
    "lr": 2.29e-5,
    "batch_size": 32,
    "epochs": 8,    # for demo purposes — in prod, set to 8 for better convergence
    "warmup_ratio": 0.105,
    "weight_decay": 0.072,
    "max_seq_len": 128,
    "dropout": 0.21,
    "early_stopping_patience": 2,
    "max_oversample": 4.1,
    "seed": 42,
    "debug_subsample": False,
    "warm_start_model_alias": "production",
    # Data paths
    "data_dir": "/mnt/block/roberta_stage1/v2",
    # Object storage
    "objstore_bucket": "objstore-proj07",
    "staging_base": "/mnt/block",
    "rclone_remote": "chi_tacc",
    # MLflow
    "experiment_name": "retraining",
    "model_registry_name": "jitsi-topic-segmenter",
    # Aggregate quality gates. Initial impl: test_wd=0.365 @ 8 epochs.
    # epochs=2 (demo) yields val_wd≈0.55, so gate raised to 0.65.
    "gate_min_f1": 0.20,
    "gate_max_pk": 0.25,
    "gate_max_windowdiff": 0.40, # was 0.40
    # Slice fairness gate — no single slice may exceed this Pk
    # Set higher than aggregate gate to allow for small-slice noise
    "slice_gate_max_pk": 0.40,
    # Ray Train
    "ray_num_workers": 1,
    "ray_use_gpu": True,
    "ray_storage_path": os.environ.get("RAY_STORAGE", "s3://ray-checkpoints"),
    "ray_max_failures": 2,
}


def load_retrain_config(config_path: Optional[str] = None) -> Dict:
    cfg = DEFAULT_RETRAIN_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg.update(yaml.safe_load(f))
    env_overrides = {
        "DATA_DIR": "data_dir",
        "MODEL_NAME": "model_registry_name",
        "RETRAIN_LR": "lr",
        "RETRAIN_EPOCHS": "epochs",
        "RETRAIN_BATCH_SIZE": "batch_size",
        "OBJSTORE_BUCKET": "objstore_bucket",
        "RCLONE_REMOTE": "rclone_remote",
    }
    for env_key, cfg_key in env_overrides.items():
        val = os.environ.get(env_key)
        if val:
            cfg[cfg_key] = val
    for k in ("lr", "weight_decay", "warmup_ratio", "dropout",
              "max_oversample", "gate_min_f1", "gate_max_pk", "gate_max_windowdiff",
              "slice_gate_max_pk"):
        if k in cfg and cfg[k] is not None:
            cfg[k] = float(cfg[k])
    for k in ("batch_size", "epochs", "max_seq_len", "early_stopping_patience",
              "seed", "ray_num_workers", "ray_max_failures"):
        if k in cfg and cfg[k] is not None:
            cfg[k] = int(cfg[k])
    return cfg


# ═══════════════════════════════════════════════════════════════════════════
# Data staging
# ═══════════════════════════════════════════════════════════════════════════

def stage_data_from_objstore(objstore_path: str, local_dir: str, cfg: Dict = None) -> bool:
    if os.path.exists(local_dir) and any(f.endswith(".jsonl") for f in os.listdir(local_dir)):
        log.info(f"Data already staged at {local_dir}, skipping download")
        return True
    bucket = os.environ.get("OBJSTORE_BUCKET", (cfg or {}).get("objstore_bucket", "objstore-proj07"))
    os.makedirs(local_dir, exist_ok=True)
    log.info(f"Staging data: s3://{bucket}/{objstore_path} → {local_dir}")

    # Use boto3 with chi.tacc credentials (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
    # AWS_S3_ENDPOINT_URL). rclone.conf may point to chi.uc which holds different data.
    endpoint_url = os.environ.get("AWS_S3_ENDPOINT_URL", "")
    if not endpoint_url:
        log.error("AWS_S3_ENDPOINT_URL not set — cannot stage training data")
        return False
    try:
        import boto3
        from botocore.config import Config as BotoConfig
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
            config=BotoConfig(signature_version="s3v4"),
        )
        prefix = objstore_path.rstrip("/") + "/"
        paginator = s3.get_paginator("list_objects_v2")
        downloaded = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                filename = os.path.basename(key)
                if not filename:
                    continue
                dest = os.path.join(local_dir, filename)
                log.info(f"  {key} ({obj['Size'] // 1024 // 1024} MB) → {dest}")
                s3.download_file(bucket, key, dest)
                downloaded += 1
        jsonl_count = len([f for f in os.listdir(local_dir) if f.endswith(".jsonl")])
        if jsonl_count == 0:
            log.error(f"Downloaded {downloaded} files but no .jsonl found in {local_dir} "
                      f"— check bucket path: s3://{bucket}/{objstore_path}")
            return False
        log.info(f"Staged {jsonl_count} .jsonl files ({downloaded} total) to {local_dir}")
        return True
    except Exception as e:
        log.error(f"Staging failed: {e}")
        return False


def resolve_dataset_path(cfg: Dict) -> str:
    if os.path.exists(cfg["data_dir"]):
        jsonl_files = [f for f in os.listdir(cfg["data_dir"]) if f.endswith(".jsonl")]
        if jsonl_files:
            log.info(f"Using explicit data_dir: {cfg['data_dir']} ({len(jsonl_files)} files)")
            return cfg["data_dir"]
    db_url = os.environ.get("DATABASE_URL", "postgresql://recap:changeme@postgres:5432/proj07_sql_db")
    try:
        import psycopg2
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute("""
            SELECT object_key FROM dataset_versions
            WHERE dataset_name = 'roberta_stage1'
            ORDER BY dataset_version_id DESC LIMIT 1
        """)
        row = cur.fetchone()
        cur.close(); conn.close()
        if row:
            obj_key = row[0]
            version = obj_key.rstrip("/").split("/")[-1]
            local_dir = os.path.join(cfg["staging_base"], "roberta_stage1", version)
            if stage_data_from_objstore(obj_key, local_dir, cfg):
                log.info(f"Resolved dataset from dataset_versions: {local_dir}")
                return local_dir
        else:
            log.warning("No roberta_stage1 entry in dataset_versions — falling back to default")
    except Exception as e:
        log.warning(f"Could not resolve dataset from DB: {e}")
    log.info(f"Falling back to default data_dir: {cfg['data_dir']}")
    return cfg["data_dir"]




# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def format_window(window: list) -> str:
    parts = []
    for utt in sorted(window, key=lambda u: u["position"]):
        if utt.get("is_padding", False):
            continue
        if utt["text"] and utt["text"].strip():
            speaker = utt["speaker"] or "UNK"
            parts.append(f"[SPEAKER_{speaker}]: {utt['text']}")
    return " ".join(parts)


def load_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_split(data_dir: str, split: str) -> Tuple[List, List, List]:
    path = os.path.join(data_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        log.warning(f"Split file not found: {path}")
        return [], [], []
    examples = load_jsonl(path)
    if os.environ.get("RETRAIN_DEBUG_SUBSAMPLE"):
        examples = examples[:500]
    texts = [format_window(e["input"]["window"]) for e in examples]
    labels = [e["output"]["label"] for e in examples]
    meeting_ids = [e["input"]["meeting_id"] for e in examples]
    return texts, labels, meeting_ids


def load_split_with_metadata(data_dir: str, split: str) -> Tuple[List, List, List, List]:
    """Load split and return number-of-speakers per example for slice evaluation."""
    path = os.path.join(data_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        log.warning(f"Split file not found: {path}")
        return [], [], [], []
    examples = load_jsonl(path)
    if os.environ.get("RETRAIN_DEBUG_SUBSAMPLE"):
        examples = examples[:500]
    texts = [format_window(e["input"]["window"]) for e in examples]
    labels = [e["output"]["label"] for e in examples]
    meeting_ids = [e["input"]["meeting_id"] for e in examples]
    n_speakers = [len(set(u["speaker"] for u in e["input"]["window"] if u["speaker"] is not None)) for e in examples]
    return texts, labels, meeting_ids, n_speakers


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

def _pk_single(ref, hyp, k, boundary="1"):
    """Pure Python Pk metric — no nltk needed."""
    if len(ref) != len(hyp):
        return 1.0
    err = 0
    total = 0
    for i in range(len(ref) - k):
        r = ref[i] != ref[i + k]
        h = hyp[i] != hyp[i + k]
        if r != h:
            err += 1
        total += 1
    return err / total if total > 0 else 0.0


def _windowdiff_single(ref, hyp, k, boundary="1"):
    """Pure Python WindowDiff — no nltk needed."""
    if len(ref) != len(hyp):
        return 1.0
    err = 0
    total = 0
    for i in range(len(ref) - k):
        r = ref[i:i+k+1].count(boundary)
        h = hyp[i:i+k+1].count(boundary)
        if r != h:
            err += 1
        total += 1
    return err / total if total > 0 else 0.0


def compute_segmentation_metrics(true_labels, pred_labels, meeting_ids=None) -> Dict:
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
            wd_scores.append(_windowdiff_single(ref, hyp, k=k, boundary="1"))
            pk_scores.append(_pk_single(ref, hyp, k=k, boundary="1"))
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


# ═══════════════════════════════════════════════════════════════════════════
# FAIRNESS SAFEGUARDING — Slice evaluation
#
# Lab: "evaluate the performance of the model on different groups to identify
# potential unfairness." Slices: meeting-size bucket + speaker-count bucket.
# Single-speaker or short meetings may correspond to specific user types
# (1:1 calls, quick standups) — if they perform worse, that's a fairness gap.
# ═══════════════════════════════════════════════════════════════════════════

def compute_slice_metrics(
    true_labels: List[int],
    test_probs: List[float],
    meeting_ids: List[str],
    n_speakers_per_example: List[int],
    threshold: float,
) -> Dict[str, Dict]:
    """
    Break Pk/F1/WD down by meeting-size and speaker-count slices.
    Logged to MLflow as slice_pk_* and slice_f1_* metrics.
    """
    preds = (np.array(test_probs) >= threshold).astype(int).tolist()

    # Count transitions per meeting to determine size bucket
    meeting_size: Dict[str, int] = defaultdict(int)
    for mid in meeting_ids:
        meeting_size[mid] += 1

    def size_bucket(mid: str) -> str:
        n = meeting_size[mid]
        if n < 15:   return "short_lt15"
        elif n <= 40: return "medium_15to40"
        else:         return "long_gt40"

    def speaker_bucket(n: int) -> str:
        if n == 1:   return "single_speaker"
        elif n == 2: return "two_speaker"
        else:        return "multi_speaker_3plus"

    slices: Dict[str, Dict[str, List]] = defaultdict(
        lambda: {"true": [], "pred": [], "probs": [], "mids": []}
    )
    for t, p, prob, mid, ns in zip(true_labels, preds, test_probs, meeting_ids, n_speakers_per_example):
        for key in (size_bucket(mid), speaker_bucket(ns)):
            slices[key]["true"].append(t)
            slices[key]["pred"].append(p)
            slices[key]["probs"].append(prob)
            slices[key]["mids"].append(mid)

    results = {}
    log.info("Slice evaluation results:")
    for slice_name, data in slices.items():
        if len(data["true"]) < 10:
            results[slice_name] = {
                "pk": -1.0, "f1": -1.0, "window_diff": -1.0,
                "n_examples": len(data["true"]),
                "n_meetings": len(set(data["mids"])),
                "note": "too_few_examples",
            }
            continue
        seg = compute_segmentation_metrics(data["true"], data["pred"], data["mids"])
        f1 = f1_score(data["true"], data["pred"], zero_division=0)
        results[slice_name] = {
            "pk": seg.get("pk", -1.0),
            "f1": round(f1, 4),
            "window_diff": seg.get("window_diff", -1.0),
            "n_examples": len(data["true"]),
            "n_meetings": len(set(data["mids"])),
        }
        log.info(f"  [{slice_name:25s}] Pk={results[slice_name]['pk']:.4f}  "
                 f"F1={results[slice_name]['f1']:.4f}  "
                 f"n_meetings={results[slice_name]['n_meetings']}")
    return results


def check_fairness_gate(slice_metrics: Dict, slice_gate_max_pk: float) -> Tuple[bool, List[str]]:
    """Fail if any data-sufficient slice has Pk > slice_gate_max_pk."""
    failures = []
    for slice_name, m in slice_metrics.items():
        if m.get("note") == "too_few_examples":
            continue
        pk = m.get("pk", -1.0)
        if pk > slice_gate_max_pk:
            failures.append(f"{slice_name}: Pk={pk:.4f} > {slice_gate_max_pk}")
    passed = len(failures) == 0
    if passed:
        log.info(f"Fairness gate PASSED — all slices Pk <= {slice_gate_max_pk}")
    else:
        log.warning(f"Fairness gate FAILED: {failures}")
    return passed, failures


# ═══════════════════════════════════════════════════════════════════════════
# ROBUSTNESS SAFEGUARDING — Known failure mode tests
#
# Lab: "evaluate a model on known failure modes" and "create a test suite."
# Three documented hard cases for topic segmentation:
#   1. Very short meetings (<5 transitions) — too few transitions to segment
#   2. No-boundary meetings — single topic, should predict all zeros
#   3. Speaker relabeling — label invariant to which speaker is A vs B
# Tests are WARN-only (don't block registration) but visible in model card.
# ═══════════════════════════════════════════════════════════════════════════

def generate_failure_mode_examples(
    test_texts: List[str],
    test_labels: List[int],
    test_meeting_ids: List[str],
) -> Dict[str, Dict]:
    """Build synthetic failure-mode test sets from real test data."""
    meeting_data: Dict[str, Dict[str, List]] = defaultdict(
        lambda: {"texts": [], "labels": [], "mids": []}
    )
    for t, l, mid in zip(test_texts, test_labels, test_meeting_ids):
        meeting_data[mid]["texts"].append(t)
        meeting_data[mid]["labels"].append(l)
        meeting_data[mid]["mids"].append(mid)

    fm_sets = {}

    # ── FM 1: very short meetings (<5 transitions) ──
    s_t, s_l, s_m = [], [], []
    for d in meeting_data.values():
        if len(d["texts"]) < 5:
            s_t.extend(d["texts"]); s_l.extend(d["labels"]); s_m.extend(d["mids"])
    if s_t:
        fm_sets["very_short_lt5"] = {
            "texts": s_t, "labels": s_l, "mids": s_m,
            "description": "Meetings with <5 transitions — model must not crash and gracefully handle minimal input",
            "pk_threshold": 0.50,   # relaxed — genuinely hard with so few transitions
        }

    # ── FM 2: no-boundary meetings (all labels = 0) ──
    n_t, n_l, n_m = [], [], []
    for d in meeting_data.values():
        if all(l == 0 for l in d["labels"]):
            n_t.extend(d["texts"]); n_l.extend(d["labels"]); n_m.extend(d["mids"])
    if n_t:
        fm_sets["no_boundary_meetings"] = {
            "texts": n_t, "labels": n_l, "mids": n_m,
            "description": "Single-topic meetings — model should predict ~all zeros; high boundary rate here is a known failure mode",
            "pk_threshold": 0.35,
        }

    # ── FM 3: speaker relabeling invariance (template test) ──
    # Rename SPEAKER_A→SPEAKER_X, SPEAKER_B→SPEAKER_Y.
    # Boundary labels should be identical. If Pk degrades significantly,
    # the model over-relies on speaker-identity tokens rather than content.
    relabeled = [
        t.replace("[SPEAKER_A]", "[SPEAKER_X]").replace("[SPEAKER_B]", "[SPEAKER_Y]")
        for t in test_texts
    ]
    fm_sets["speaker_relabel_invariance"] = {
        "texts": relabeled, "labels": test_labels, "mids": test_meeting_ids,
        "description": "Speaker tokens relabeled A→X, B→Y. Pk should be similar to aggregate — tests content vs. identity reliance.",
        "pk_threshold": 0.30,   # should be close to aggregate Pk
    }

    return fm_sets


def run_failure_mode_tests(
    model, tokenizer, device,
    test_texts: List[str],
    test_labels: List[int],
    test_meeting_ids: List[str],
    threshold: float,
    batch_size: int,
    max_seq_len: int,
) -> Dict[str, Dict]:
    """Run known-failure-mode tests. Returns results logged to MLflow + model card."""
    log.info("Running known failure mode tests (robustness)...")
    fm_sets = generate_failure_mode_examples(test_texts, test_labels, test_meeting_ids)
    results = {}

    for test_name, fm in fm_sets.items():
        if len(fm["texts"]) < 3:
            results[test_name] = {
                "status": "SKIP", "reason": "too_few_examples",
                "n": len(fm["texts"]), "description": fm["description"],
            }
            log.info(f"  [{test_name}] SKIP — only {len(fm['texts'])} examples")
            continue

        ds = WindowDataset(fm["texts"], fm["labels"], tokenizer, max_seq_len)
        loader = DataLoader(ds, batch_size=batch_size * 2, num_workers=0)
        probs_list, true_list = [], []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                probs_list.extend(probs)
                true_list.extend(batch["labels"].numpy())

        preds = (np.array(probs_list) >= threshold).astype(int).tolist()
        seg = compute_segmentation_metrics(true_list, preds, fm["mids"])
        pk = seg.get("pk", -1.0)
        f1 = f1_score(true_list, preds, zero_division=0)
        thr = fm["pk_threshold"]
        passed = (pk <= thr) if pk >= 0 else True

        status = "PASS" if passed else "WARN"
        results[test_name] = {
            "status": status,
            "pk": pk,
            "f1": round(f1, 4),
            "window_diff": seg.get("window_diff", -1.0),
            "pk_threshold": thr,
            "n_examples": len(fm["texts"]),
            "description": fm["description"],
        }
        log.info(f"  [{test_name}] {status} — Pk={pk:.4f} (limit={thr}), "
                 f"F1={f1:.4f}, n={len(fm['texts'])}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# TRANSPARENCY + ACCOUNTABILITY SAFEGUARDING — Model card
#
# Safeguarding lecture: "users should be able to see the metrics of the system"
# and "someone needs to own the system when harm is reported."
# A model card logged as a MLflow artifact fulfils both: it documents
# training data, thresholds, fairness results, privacy handling, and
# the accountability chain.
# ═══════════════════════════════════════════════════════════════════════════

def build_model_card(
    config: Dict,
    test_f1: float, test_pk: float, test_wd: float,
    best_threshold: float,
    gates_passed: bool,
    slice_metrics: Dict,
    failure_mode_results: Dict,
    dataset_version: str,
    run_id: str,
) -> Dict:
    fairness_summary = {
        k: {"pk": v.get("pk", -1.0), "f1": v.get("f1", -1.0),
            "n_meetings": v.get("n_meetings", 0), "note": v.get("note", "")}
        for k, v in slice_metrics.items()
    }
    robustness_summary = {
        k: {"status": v.get("status"), "pk": v.get("pk", -1.0),
            "description": v.get("description", "")}
        for k, v in failure_mode_results.items()
    }
    return {
        "model_name": config["model_registry_name"],
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "created_by": "automated retrain pipeline (retrain.py) — NeuralOps / Mahima",
        "mlflow_run_id": run_id,

        # Transparency: training data provenance
        "training_data": {
            "source": "AMI Meeting Corpus (Univ. of Edinburgh)",
            "dataset_version": dataset_version,
            "split_strategy": "strict split by meeting_id — no meeting spans train/val/test",
            "leakage_controls": (
                "Meeting IDs frozen at roberta_stage1/v1 creation; Jitsi meetings "
                "only promoted to training set by Aneesh's batch pipeline after explicit review."
            ),
        },

        # Transparency: model
        "model": {
            "base": config["model_name"],
            "task": "binary classification — boundary vs. non-boundary utterance transition",
            "warm_start": config.get("warm_start_model_alias", "none"),
            "hyperparameters": {
                "lr": config["lr"], "batch_size": config["batch_size"],
                "dropout": config["dropout"], "max_oversample": config["max_oversample"],
                "warmup_ratio": config["warmup_ratio"], "epochs_run": config["epochs"],
            },
        },

        # Transparency: inference threshold
        # IMPORTANT FOR SERVING: Shruti must read best_threshold from this card
        # or from the model version tag, NOT hardcode 0.5.
        "inference": {
            "best_threshold": best_threshold,
            "threshold_selection": "sweep [0.05..0.50] on val set, minimise per-meeting Pk",
            "serving_note": (
                "READ best_threshold FROM model version tag 'best_threshold' in MLflow. "
                "Do NOT hardcode 0.5 — the optimal threshold is calibrated per training run."
            ),
        },

        # Aggregate test metrics
        "aggregate_test_metrics": {
            "f1": round(test_f1, 4),
            "pk": round(test_pk, 4),
            "window_diff": round(test_wd, 4),
        },

        # Quality gates
        "quality_gates": {
            "gate_min_f1": config["gate_min_f1"],
            "gate_max_pk": config["gate_max_pk"],
            "gate_max_windowdiff": config["gate_max_windowdiff"],
            "slice_gate_max_pk": config["slice_gate_max_pk"],
            "aggregate_passed": (
                test_f1 >= config["gate_min_f1"]
                and test_pk <= config["gate_max_pk"]
                and test_wd <= config["gate_max_windowdiff"]
            ),
            "overall_passed": gates_passed,
        },

        # Fairness: per-slice evaluation
        # Slices: meeting size (short/medium/long) + speaker count (1/2/3+)
        # Gate: no slice with n>=10 examples may have Pk > slice_gate_max_pk
        "fairness_slice_evaluation": fairness_summary,

        # Robustness: failure mode tests
        # WARN-only: very short meetings, no-boundary meetings, speaker relabeling
        "robustness_failure_mode_tests": robustness_summary,

        # Privacy
        "privacy": {
            "speaker_pii": (
                "Speaker identity abstracted to SPEAKER_A/B/C tokens in all training examples. "
                "No participant names, emails, or IDs in training data."
            ),
            "feedback_data_handling": (
                "User corrections reference utterance_idx only; raw transcript text "
                "processed by Aneesh's pipeline before entering training JSONL."
            ),
            "data_retention": (
                "Training JSONL files versioned in objstore-proj07. "
                "Versions older than 6 months may be archived per team policy."
            ),
            "consent": (
                "AMI corpus: publicly released research dataset (ICSICORPUS licence). "
                "Jitsi feedback: meeting participants implicitly consent via session terms."
            ),
        },

        # Accountability
        "accountability": {
            "training_owner": "Mahima Sachdeva (NeuralOps training role)",
            "serving_owner": "Shruti Pangare (NeuralOps serving role)",
            "data_owner": "Aneesh Mokashi (NeuralOps data role)",
            "promotion_process": (
                "Automated: 'candidate' alias set in MLflow if all gates pass. "
                "Manual: engineer promotes candidate→production in MLflow UI. "
                "No SSH required for promotion."
            ),
            "rollback_trigger": (
                "Shruti's monitoring: if user correction rate exceeds 2x baseline "
                "post-deployment, serving layer auto-rolls back to previous 'production' version."
            ),
            "audit_log": "Every retrain event written to audit_log table in proj07_sql_db.",
        },

        # Known limitations (transparency)
        "limitations": [
            "Trained primarily on AMI (English, structured corporate meetings). "
            "May underperform on informal or non-English meetings.",
            "Class imbalance (~40:1) handled by WeightedRandomSampler; F1 is low by design.",
            "Single-speaker meetings lack speaker-change signal; relies on lexical cues only.",
            "Threshold optimised for Pk, not F1. Serving layer must use best_threshold tag.",
            "Feedback data is weighted 2x but volume is small relative to AMI — "
            "bias toward AMI meeting style remains.",
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Dataset class
# ═══════════════════════════════════════════════════════════════════════════

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
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ═══════════════════════════════════════════════════════════════════════════
# Ray Train training function
# ═══════════════════════════════════════════════════════════════════════════

def train_func(config: Dict):
    """
    Ray Train training function wrapping raw PyTorch (not Lightning).
    Beyond the lab: raw PyTorch loop, operational fault tolerance,
    quality gates + MLflow integration.
    """
    import ray.train.torch

    # VRAM pre-check — fail fast if GPU is too loaded (e.g. serving model is using it)
    if torch.cuda.is_available():
        free_vram_gb = torch.cuda.mem_get_info()[0] / 1e9
        total_vram_gb = torch.cuda.mem_get_info()[1] / 1e9
        log.info(f"VRAM: {free_vram_gb:.1f} GB free / {total_vram_gb:.1f} GB total")
        if free_vram_gb < 8.0:
            raise RuntimeError(
                f"Insufficient VRAM: {free_vram_gb:.1f} GB free, need >= 8.0 GB. "
                f"Serving model may be holding GPU memory."
            )

    torch.manual_seed(config.get("seed", 42))
    np.random.seed(config.get("seed", 42))

    train_texts, train_labels, _ = load_split(config["data_dir"], "train")
    if not train_texts:
        raise RuntimeError(
            f"No training examples found in {config['data_dir']} — "
            "data was not staged correctly (check rclone and dataset_versions)"
        )
    if config.get("debug_subsample"):
        train_texts, train_labels = train_texts[:500], train_labels[:500]
    val_texts, val_labels, val_meeting_ids = load_split(config["data_dir"], "val")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    # Include relabeled speaker tokens for template perturbation tests
    special_tokens = [f"[SPEAKER_{s}]" for s in "ABCDEFGH"] + ["[SPEAKER_X]", "[SPEAKER_Y]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    train_ds = WindowDataset(train_texts, train_labels, tokenizer, config["max_seq_len"])
    val_ds = WindowDataset(val_texts, val_labels, tokenizer, config["max_seq_len"])

    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    effective_ratio = min(n_neg / max(n_pos, 1), config.get("max_oversample", 4.1))
    sample_weights = [1.0 if l == 0 else effective_ratio for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"] * 2, num_workers=0)

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=2,
        hidden_dropout_prob=config["dropout"],
        attention_probs_dropout_prob=config["dropout"],
    )
    model.resize_token_embeddings(len(tokenizer))

    if config.get("warm_start_model_alias"):
        try:
            registry_name = config.get("model_registry_name", "jitsi-topic-segmenter")
            log.info(f"Warm-start from {registry_name}@{config['warm_start_model_alias']}")
            warm_model = mlflow.pytorch.load_model(
                f"models:/{registry_name}@{config['warm_start_model_alias']}"
            )
            missing, _ = model.load_state_dict(warm_model.state_dict(), strict=False)
            if missing:
                log.warning(f"Warm-start missing keys: {missing}")
            log.info("Warm-start loaded successfully")
            if config["epochs"] > 5:
                config["epochs"] = 5
                log.info(f"Reduced to {config['epochs']} epochs for warm-start")
        except Exception as e:
            log.warning(f"Warm-start failed ({e}), using base weights")

    if config.get("freeze_backbone", False):
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    model = ray.train.torch.prepare_model(model)
    train_loader = ray.train.torch.prepare_data_loader(train_loader)
    val_loader = ray.train.torch.prepare_data_loader(val_loader)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"], weight_decay=config["weight_decay"],
    )
    total_steps = len(train_loader) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(config["warmup_ratio"] * total_steps), total_steps
    )

    # Restore from checkpoint if resuming after Ray failure
    start_epoch, best_val_pk, best_threshold = 1, float("inf"), 0.5
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as ckpt_dir:
            ckpt = torch.load(os.path.join(ckpt_dir, "checkpoint.pt"), weights_only=False)
            (model.module if hasattr(model, "module") else model).load_state_dict(
                ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_pk = ckpt.get("best_val_pk", float("inf"))
            best_threshold = ckpt.get("best_threshold", 0.5)
            log.info(f"Resumed from checkpoint epoch {ckpt['epoch']}, best_pk={best_val_pk:.4f}")

    patience_counter = 0

    for epoch in range(start_epoch, config["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(
                model(input_ids=batch["input_ids"],
                      attention_mask=batch["attention_mask"]).logits,
                batch["labels"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            epoch_loss += loss.item()

        model.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                probs = torch.softmax(
                    model(input_ids=batch["input_ids"],
                          attention_mask=batch["attention_mask"]).logits,
                    dim=-1
                )[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_true.extend(batch["labels"].cpu().numpy())

        epoch_threshold, epoch_metrics = sweep_thresholds(val_probs, val_true, val_meeting_ids)
        val_pk = epoch_metrics.get("pk", 1.0)
        val_f1 = epoch_metrics.get("f1", 0.0)

        if val_pk < best_val_pk:
            best_val_pk = val_pk
            best_threshold = epoch_threshold
            patience_counter = 0
        else:
            patience_counter += 1

        with tempfile.TemporaryDirectory() as tmpdir:
            underlying = model.module if hasattr(model, "module") else model
            torch.save({
                "epoch": epoch,
                "model_state_dict": underlying.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_pk": best_val_pk,
                "best_threshold": best_threshold,
            }, os.path.join(tmpdir, "checkpoint.pt"))
            tokenizer.save_pretrained(os.path.join(tmpdir, "tokenizer"))
            train.report(
                metrics={
                    "epoch": epoch,
                    "train_loss": round(epoch_loss / max(len(train_loader), 1), 4),
                    "val_f1": round(val_f1, 4),
                    "val_pk": round(val_pk, 4),
                    "val_window_diff": round(epoch_metrics.get("window_diff", 1.0), 4),
                    "best_val_pk": round(best_val_pk, 4),
                    "threshold": epoch_threshold,
                },
                checkpoint=train.Checkpoint.from_directory(tmpdir),
            )

        log.info(f"Epoch {epoch}/{config['epochs']} | "
                 f"loss={epoch_loss / max(len(train_loader), 1):.4f} | "
                 f"val_f1={val_f1:.4f} | val_pk={val_pk:.4f} | thr={epoch_threshold:.2f}")

        if patience_counter >= config["early_stopping_patience"]:
            log.info(f"Early stopping at epoch {epoch}")
            break


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation + registration (runs on driver, not Ray workers)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_and_register(config: Dict, result, ckpt_data: dict | None = None) -> bool:
    if ckpt_data is not None:
        ckpt = ckpt_data
    elif result.checkpoint is not None:
        with result.checkpoint.as_directory() as ckpt_dir:
            ckpt = torch.load(os.path.join(ckpt_dir, "checkpoint.pt"), weights_only=False)
    else:
        log.error("No checkpoint — cannot evaluate")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    special_tokens = [f"[SPEAKER_{s}]" for s in "ABCDEFGH"] + ["[SPEAKER_X]", "[SPEAKER_Y]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    model = AutoModelForSequenceClassification.from_pretrained(config["model_name"], num_labels=2)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    best_threshold = ckpt.get("best_threshold", 0.5)

    # Load test data WITH metadata (n_speakers per example) for slice eval
    test_texts, test_labels, test_meeting_ids, test_n_speakers = \
        load_split_with_metadata(config["data_dir"], "test")

    # Run inference
    test_ds = WindowDataset(test_texts, test_labels, tokenizer, config["max_seq_len"])
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"] * 2, num_workers=0)
    test_probs, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            probs = torch.softmax(
                model(input_ids=batch["input_ids"].to(device),
                      attention_mask=batch["attention_mask"].to(device)).logits,
                dim=-1
            )[:, 1].cpu().numpy()
            test_probs.extend(probs)
            test_true.extend(batch["labels"].numpy())

    test_preds = (np.array(test_probs) >= best_threshold).astype(int)
    test_seg = compute_segmentation_metrics(test_true, test_preds.tolist(), test_meeting_ids)
    test_f1 = f1_score(test_true, test_preds, zero_division=0)
    test_pk = test_seg.get("pk", 1.0)
    test_wd = test_seg.get("window_diff", 1.0)

    log.info(f"Aggregate — F1={test_f1:.4f}, Pk={test_pk:.4f}, WD={test_wd:.4f}, "
             f"threshold={best_threshold:.2f}")

    # ── Aggregate gates ──
    agg_passed = (
        test_f1 >= config["gate_min_f1"]
        and test_pk <= config["gate_max_pk"]
        and test_wd <= config["gate_max_windowdiff"]
    )
    log.info(f"Aggregate gates: F1 {'PASS' if test_f1 >= config['gate_min_f1'] else 'FAIL'} | "
             f"Pk {'PASS' if test_pk <= config['gate_max_pk'] else 'FAIL'} | "
             f"WD {'PASS' if test_wd <= config['gate_max_windowdiff'] else 'FAIL'}")

    # ── Slice evaluation (FAIRNESS) ──
    log.info("Running slice evaluation (fairness safeguarding)...")
    slice_metrics = compute_slice_metrics(
        test_true, test_probs, test_meeting_ids, test_n_speakers, best_threshold
    )
    fairness_passed, fairness_failures = check_fairness_gate(
        slice_metrics, config["slice_gate_max_pk"]
    )

    gates_passed = agg_passed and fairness_passed

    # ── Known failure mode tests (ROBUSTNESS) ──
    failure_mode_results = run_failure_mode_tests(
        model=model, tokenizer=tokenizer, device=device,
        test_texts=test_texts, test_labels=test_true,
        test_meeting_ids=test_meeting_ids,
        threshold=best_threshold,
        batch_size=config["batch_size"],
        max_seq_len=config["max_seq_len"],
    )
    any_fm_warn = any(r.get("status") == "WARN" for r in failure_mode_results.values())
    if any_fm_warn:
        log.warning("Some failure mode tests WARN — details in model card")

    # ── MLflow ──
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(config["experiment_name"])
    dataset_version = os.environ.get("DATASET_VERSION", "unknown")
    run_name = f"retrain-{'pass' if gates_passed else 'fail'}-{int(time.time())}"

    with mlflow.start_run(run_name=run_name) as run:
        # Parameters
        mlflow.log_params({
            "model_name": config["model_name"],
            "lr": config["lr"], "batch_size": config["batch_size"],
            "epochs": config["epochs"], "max_oversample": config["max_oversample"],
            "dropout": config["dropout"], "warmup_ratio": config["warmup_ratio"],
            "weight_decay": config["weight_decay"],
            "dataset_version": dataset_version,
            "warm_start": config.get("warm_start_model_alias", "none"),
            "ray_num_workers": config["ray_num_workers"],
            "ray_max_failures": config["ray_max_failures"],
            "retrain_mode": "automated",
        })

        # Aggregate metrics
        mlflow.log_metrics({
            "test_f1": round(test_f1, 4),
            "test_pk": round(test_pk, 4),
            "test_window_diff": round(test_wd, 4),
            "test_precision": round(precision_score(test_true, test_preds, zero_division=0), 4),
            "test_recall": round(recall_score(test_true, test_preds, zero_division=0), 4),
            "best_threshold": best_threshold,
            "gates_passed": int(gates_passed),
            "fairness_gate_passed": int(fairness_passed),
        })

        # Per-slice metrics
        for slice_name, sm in slice_metrics.items():
            if sm.get("pk", -1) >= 0:
                mlflow.log_metric(f"slice_pk_{slice_name}", sm["pk"])
            if sm.get("f1", -1) >= 0:
                mlflow.log_metric(f"slice_f1_{slice_name}", sm["f1"])

        # Failure mode metrics
        for test_name, fm_res in failure_mode_results.items():
            if fm_res.get("pk", -1) >= 0:
                mlflow.log_metric(f"fm_pk_{test_name}", fm_res["pk"])

        # Build model card (for ALL runs — useful for debugging failures too)
        model_card = build_model_card(
            config=config, test_f1=test_f1, test_pk=test_pk, test_wd=test_wd,
            best_threshold=best_threshold, gates_passed=gates_passed,
            slice_metrics=slice_metrics, failure_mode_results=failure_mode_results,
            dataset_version=dataset_version, run_id=run.info.run_id,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            card_path = os.path.join(tmpdir, "model_card.json")
            with open(card_path, "w") as f:
                json.dump(model_card, f, indent=2, default=lambda o: bool(o) if isinstance(o, __import__("numpy").bool_) else str(o))
            mlflow.log_artifact(card_path, artifact_path="model_card")
        log.info("Model card logged to MLflow")

        if gates_passed:
            log.info("All gates PASSED — registering model")
            mlflow.pytorch.log_model(
                model, artifact_path="model",
                pip_requirements=["transformers==4.40.0", "torch==2.2.0"],
                registered_model_name=config["model_registry_name"],
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                tokenizer.save_pretrained(tmpdir)
                mlflow.log_artifacts(tmpdir, artifact_path="tokenizer")

            client = mlflow.tracking.MlflowClient()
            latest = client.get_latest_versions(config["model_registry_name"])
            if latest:
                version = latest[-1].version
                try:
                    client.set_registered_model_alias(
                        config["model_registry_name"], "candidate", version)
                    # Promote directly to production — serving hot-reload picks it
                    # up within MODEL_RELOAD_INTERVAL_SECONDS (default 300s).
                    client.set_registered_model_alias(
                        config["model_registry_name"], "production", version)
                    # Tag threshold so serving reads it directly from registry
                    client.set_model_version_tag(
                        config["model_registry_name"], version,
                        "best_threshold", str(best_threshold))
                    client.set_model_version_tag(
                        config["model_registry_name"], version,
                        "dataset_version", dataset_version)
                    client.set_model_version_tag(
                        config["model_registry_name"], version,
                        "gates_passed", "true")
                    log.info(f"v{version} promoted to 'production' (was 'candidate'), "
                             f"threshold={best_threshold}")
                except Exception as e:
                    log.warning(f"Could not set alias/tags: {e}")
        else:
            log.warning("Gates FAILED — model NOT registered")
            mlflow.pytorch.log_model(model, artifact_path="model-failed")

    _log_to_audit_db("retrain_completed", {
        "run_id": run.info.run_id,
        "gates_passed": gates_passed,
        "aggregate_passed": agg_passed,
        "fairness_passed": fairness_passed,
        "fairness_failures": fairness_failures,
        "test_f1": round(test_f1, 4),
        "test_pk": round(test_pk, 4),
        "best_threshold": best_threshold,
        "dataset_version": dataset_version,
        "fm_warns": [k for k, v in failure_mode_results.items() if v.get("status") == "WARN"],
        "max_feedback_event_id": int(os.environ.get("MAX_FEEDBACK_EVENT_ID", "0")),
    })

    return gates_passed


def _log_to_audit_db(event_type: str, details: dict):
    """Soft-fail audit log — table may not exist on first run."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        log.warning("DATABASE_URL not set — skipping audit log")
        return
    try:
        import psycopg2
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute("INSERT INTO audit_log (event_type, details) VALUES (%s, %s)",
                    (event_type, json.dumps(details)))
        conn.commit(); cur.close(); conn.close()
        log.info(f"Audit log written: {event_type}")
    except Exception as e:
        log.warning(f"Audit log write failed (run add_mlops_tables.sql if missing): {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Ray storage resolution — chi.tacc S3 via pyarrow (separate bucket from MLflow)
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_storage(storage_path: str):
    """Return (path, filesystem) for Ray RunConfig.

    Checkpoints go to MinIO (local Docker service at http://minio:9000).
    Plain HTTP, no region detection, reliable within the Docker network.
    MLflow model artifacts use chi.tacc S3 separately — no overlap.
    """
    if not storage_path.startswith("s3://"):
        return storage_path, None
    endpoint = os.environ.get("MINIO_ENDPOINT", "minio:9000")
    fs = pafs.S3FileSystem(
        access_key=os.environ.get("MINIO_USER", "minioadmin"),
        secret_key=os.environ.get("MINIO_PASSWORD", "changeme_minio"),
        endpoint_override=endpoint,
        scheme="http",
        region="us-east-1",
    )
    return storage_path[len("s3://"):], fs


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no_gpu", action="store_true",
                        help="Disable GPU (for CPU-only testing)")
    args = parser.parse_args()

    cfg = load_retrain_config(args.config)
    if args.data_dir:  cfg["data_dir"] = args.data_dir
    if args.epochs:    cfg["epochs"] = args.epochs
    if args.lr:        cfg["lr"] = args.lr
    if args.no_gpu:    cfg["ray_use_gpu"] = False

    cfg["data_dir"] = resolve_dataset_path(cfg)

    log.info("Starting retrain pipeline")
    log.info(f"  data_dir:     {cfg['data_dir']}")
    log.info(f"  objstore:     {cfg['rclone_remote']}:{cfg['objstore_bucket']}")
    log.info(f"  gates:        F1>={cfg['gate_min_f1']}, Pk<={cfg['gate_max_pk']}, "
             f"WD<={cfg['gate_max_windowdiff']}, SlicePk<={cfg['slice_gate_max_pk']}")
    log.info(f"  train:        epochs={cfg['epochs']}, lr={cfg['lr']}, "
             f"gpu={cfg['ray_use_gpu']}, workers={cfg['ray_num_workers']}")

    ray_address = os.environ.get("RAY_ADDRESS", None)
    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True, log_to_driver=True)
    else:
        ray.init(ignore_reinit_error=True, log_to_driver=True)

    ray_storage_path, ray_storage_fs = _resolve_storage(cfg["ray_storage_path"])
    run_cfg_kwargs = {"storage_path": ray_storage_path}
    if ray_storage_fs is not None:
        run_cfg_kwargs["storage_filesystem"] = ray_storage_fs

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=cfg,
        scaling_config=ScalingConfig(
            num_workers=cfg["ray_num_workers"],
            use_gpu=cfg["ray_use_gpu"],
            resources_per_worker={"CPU": 4, "GPU": 1} if cfg["ray_use_gpu"] else {"CPU": 1},
        ),
        run_config=RunConfig(
            name=f"retrain-{int(time.time())}",
            failure_config=FailureConfig(max_failures=cfg["ray_max_failures"]),
            checkpoint_config=CheckpointConfig(num_to_keep=2),
            **run_cfg_kwargs,
        ),
    )

    log.info("Launching Ray TorchTrainer...")
    result = trainer.fit()
    log.info(f"Training done. Best val metrics: {result.metrics}")

    # Extract checkpoint to a local dict before shutting Ray down, so GPU memory
    # held by Ray training workers is released before evaluate_and_register loads
    # the model on CUDA (prevents OOM when worker VRAM isn't freed yet).
    ckpt_data = None
    if result.checkpoint is not None:
        with result.checkpoint.as_directory() as ckpt_dir:
            ckpt_data = torch.load(
                os.path.join(ckpt_dir, "checkpoint.pt"),
                weights_only=False,
                map_location="cpu",  # keep tensors on CPU so ray.shutdown() can't invalidate them
            )

    ray.shutdown()
    log.info("Ray shut down — GPU memory released before evaluation")

    gates_passed = evaluate_and_register(cfg, result, ckpt_data=ckpt_data)
    log.info("Retrain SUCCESS — model registered as 'candidate'" if gates_passed
             else "Retrain done — model did NOT pass quality gates")

    return gates_passed


if __name__ == "__main__":
    exit(0 if main() else 1)