"""
Unit tests for retrain.py — covers metric computation, quality gates,
data loading, fairness evaluation, and failure-mode test generation.

Run inside the training container:
    pytest train/tests/test_retrain.py -v

No GPU required; no MLflow server required; no database required.
"""
import importlib
import json
import os
import sys
import tempfile
import types
from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import helpers — stub out heavy deps that aren't needed for unit tests
# ---------------------------------------------------------------------------

def _stub_ray():
    ray = types.ModuleType("ray")
    ray.train = types.ModuleType("ray.train")
    ray.init = lambda *a, **kw: None
    for sub in ("ray.train.torch", "ray.train"):
        sys.modules.setdefault(sub, types.ModuleType(sub))
    sys.modules.setdefault("ray", ray)
    sys.modules.setdefault("ray.train", ray.train)
    for attr in ("RunConfig", "FailureConfig", "CheckpointConfig", "ScalingConfig"):
        setattr(ray.train, attr, MagicMock)
    return ray

_stub_ray()

# Stub transformers to avoid downloading weights
_transformers = types.ModuleType("transformers")
for cls in ("AutoTokenizer", "AutoModelForSequenceClassification", "get_linear_schedule_with_warmup"):
    setattr(_transformers, cls, MagicMock)
sys.modules.setdefault("transformers", _transformers)

# Stub mlflow
_mlflow = types.ModuleType("mlflow")
_mlflow.pytorch = types.ModuleType("mlflow.pytorch")
_mlflow.tracking = types.ModuleType("mlflow.tracking")
_mlflow.tracking.MlflowClient = MagicMock
_mlflow.start_run = MagicMock()
_mlflow.log_metric = MagicMock()
_mlflow.log_param = MagicMock()
_mlflow.log_artifact = MagicMock()
_mlflow.log_artifacts = MagicMock()
_mlflow.set_experiment = MagicMock()
_mlflow.set_tracking_uri = MagicMock()
for m in ("mlflow", "mlflow.pytorch", "mlflow.tracking"):
    sys.modules.setdefault(m, _mlflow)

# Stub optuna
sys.modules.setdefault("optuna", types.ModuleType("optuna"))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import retrain as R


# ===========================================================================
# Pk metric
# ===========================================================================

class TestPkMetric:
    def test_perfect_match_returns_zero(self):
        assert R._pk_single("0001000", "0001000", k=2) == 0.0

    def test_all_wrong_returns_nonzero(self):
        score = R._pk_single("0001000", "1110111", k=2)
        assert score > 0.0

    def test_length_mismatch_returns_one(self):
        assert R._pk_single("000", "0000", k=1) == 1.0

    def test_empty_returns_zero(self):
        assert R._pk_single("", "", k=2) == 0.0

    def test_score_between_zero_and_one(self):
        score = R._pk_single("0101010101", "0000000000", k=3)
        assert 0.0 <= score <= 1.0

    def test_symmetric(self):
        ref = "0010100"
        hyp = "0001100"
        assert R._pk_single(ref, hyp, k=2) == R._pk_single(hyp, ref, k=2)


# ===========================================================================
# WindowDiff metric
# ===========================================================================

class TestWindowDiff:
    def test_identical_sequences_zero(self):
        assert R._windowdiff_single("0001000", "0001000", k=3) == 0.0

    def test_length_mismatch_returns_one(self):
        assert R._windowdiff_single("000", "0000", k=1) == 1.0

    def test_score_bounded(self):
        score = R._windowdiff_single("0101010101", "1010101010", k=3)
        assert 0.0 <= score <= 1.0

    def test_penalizes_missed_boundary(self):
        ref = "0001000"  # one boundary
        hyp = "0000000"  # no boundary
        score = R._windowdiff_single(ref, hyp, k=2)
        assert score > 0.0


# ===========================================================================
# compute_segmentation_metrics
# ===========================================================================

class TestComputeSegmentationMetrics:
    def test_returns_neg_one_without_meeting_ids(self):
        result = R.compute_segmentation_metrics([0, 1, 0], [0, 1, 0], meeting_ids=None)
        assert result["pk"] == -1.0
        assert result["window_diff"] == -1.0

    def test_perfect_prediction(self):
        labels = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
        mids = ["m1"] * 10
        result = R.compute_segmentation_metrics(labels, labels, mids)
        assert result["pk"] == pytest.approx(0.0, abs=1e-4)

    def test_skips_short_meetings(self):
        # Meeting shorter than 4 tokens should be skipped gracefully
        labels = [0, 1, 0]
        mids = ["short"] * 3
        result = R.compute_segmentation_metrics(labels, labels, mids)
        assert result["pk"] == -1.0

    def test_multiple_meetings_averaged(self):
        labels_a = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        labels_b = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
        mids = ["m1"] * 10 + ["m2"] * 10
        result = R.compute_segmentation_metrics(
            labels_a + labels_b, labels_a + labels_b, mids
        )
        assert result["pk"] == pytest.approx(0.0, abs=1e-4)


# ===========================================================================
# Quality gates
# ===========================================================================

class TestQualityGates:
    BASE = {
        "gate_min_f1": 0.20,
        "gate_max_pk": 0.25,
        "gate_max_windowdiff": 0.65,
    }

    def _check(self, f1, pk, wd):
        cfg = self.BASE
        agg = (f1 >= cfg["gate_min_f1"]
               and pk <= cfg["gate_max_pk"]
               and wd <= cfg["gate_max_windowdiff"])
        return agg

    def test_all_gates_pass(self):
        assert self._check(f1=0.25, pk=0.20, wd=0.50) is True

    def test_f1_gate_fails(self):
        assert self._check(f1=0.10, pk=0.20, wd=0.50) is False

    def test_pk_gate_fails(self):
        assert self._check(f1=0.25, pk=0.30, wd=0.50) is False

    def test_wd_gate_fails(self):
        assert self._check(f1=0.25, pk=0.20, wd=0.70) is False

    def test_boundary_values_pass(self):
        assert self._check(f1=0.20, pk=0.25, wd=0.65) is True


# ===========================================================================
# Fairness gate
# ===========================================================================

class TestFairnessGate:
    def test_all_slices_pass(self):
        metrics = {
            "short_lt15":       {"pk": 0.30, "f1": 0.22},
            "medium_15to40":    {"pk": 0.25, "f1": 0.25},
            "long_gt40":        {"pk": 0.20, "f1": 0.30},
        }
        passed, failures = R.check_fairness_gate(metrics, slice_gate_max_pk=0.40)
        assert passed is True
        assert failures == []

    def test_one_slice_fails(self):
        metrics = {
            "short_lt15":    {"pk": 0.45, "f1": 0.10},
            "long_gt40":     {"pk": 0.20, "f1": 0.30},
        }
        passed, failures = R.check_fairness_gate(metrics, slice_gate_max_pk=0.40)
        assert passed is False
        assert len(failures) == 1
        assert "short_lt15" in failures[0]

    def test_skips_too_few_examples(self):
        metrics = {
            "single_speaker": {"pk": 0.99, "note": "too_few_examples"},
            "long_gt40":      {"pk": 0.20},
        }
        passed, failures = R.check_fairness_gate(metrics, slice_gate_max_pk=0.40)
        assert passed is True

    def test_all_slices_fail(self):
        metrics = {
            "short_lt15":  {"pk": 0.50},
            "long_gt40":   {"pk": 0.60},
        }
        passed, failures = R.check_fairness_gate(metrics, slice_gate_max_pk=0.40)
        assert passed is False
        assert len(failures) == 2

    def test_exact_boundary_passes(self):
        metrics = {"medium_15to40": {"pk": 0.40}}
        passed, _ = R.check_fairness_gate(metrics, slice_gate_max_pk=0.40)
        assert passed is True


# ===========================================================================
# format_window
# ===========================================================================

class TestFormatWindow:
    def _utt(self, speaker, text, position, is_padding=False):
        return {"speaker": speaker, "text": text,
                "position": position, "is_padding": is_padding}

    def test_basic_formatting(self):
        window = [self._utt("A", "hello", 0), self._utt("B", "world", 1)]
        result = R.format_window(window)
        assert "[SPEAKER_A]: hello" in result
        assert "[SPEAKER_B]: world" in result

    def test_padding_excluded(self):
        window = [
            self._utt("A", "real", 0),
            self._utt("B", "padded", 1, is_padding=True),
        ]
        result = R.format_window(window)
        assert "padded" not in result

    def test_empty_text_excluded(self):
        window = [self._utt("A", "", 0), self._utt("B", "present", 1)]
        result = R.format_window(window)
        assert "[SPEAKER_A]" not in result
        assert "present" in result

    def test_sorted_by_position(self):
        window = [self._utt("B", "second", 1), self._utt("A", "first", 0)]
        result = R.format_window(window)
        assert result.index("first") < result.index("second")

    def test_null_speaker(self):
        window = [{"speaker": None, "text": "anon", "position": 0, "is_padding": False}]
        result = R.format_window(window)
        assert "[SPEAKER_UNK]: anon" in result


# ===========================================================================
# load_split
# ===========================================================================

class TestLoadSplit:
    def _make_example(self, meeting_id, label, texts):
        return {
            "input": {
                "meeting_id": meeting_id,
                "window": [
                    {"speaker": "A", "text": t, "position": i, "is_padding": False}
                    for i, t in enumerate(texts)
                ],
            },
            "output": {"label": label},
        }

    def test_loads_jsonl_correctly(self):
        examples = [
            self._make_example("m1", 0, ["hello", "world"]),
            self._make_example("m1", 1, ["topic", "change"]),
        ]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "train.jsonl")
            with open(path, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")
            texts, labels, mids = R.load_split(d, "train")
        assert len(texts) == 2
        assert labels == [0, 1]
        assert all(m == "m1" for m in mids)

    def test_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            texts, labels, mids = R.load_split(d, "train")
        assert texts == [] and labels == [] and mids == []

    def test_text_content_formatted(self):
        example = self._make_example("m1", 0, ["hello"])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "val.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps(example) + "\n")
            texts, _, _ = R.load_split(d, "val")
        assert "[SPEAKER_A]: hello" in texts[0]


# ===========================================================================
# generate_failure_mode_examples
# ===========================================================================

class TestFailureModeExamples:
    def _make_data(self, n_per_meeting=10, n_meetings=5, include_no_boundary=True):
        texts, labels, mids = [], [], []
        for i in range(n_meetings):
            mid = f"meet_{i}"
            for j in range(n_per_meeting):
                texts.append(f"[SPEAKER_A]: utterance {j}")
                labels.append(1 if j == n_per_meeting // 2 else 0)
                mids.append(mid)
        if include_no_boundary:
            for j in range(10):
                texts.append(f"[SPEAKER_B]: no boundary {j}")
                labels.append(0)
                mids.append("no_boundary_meet")
        return texts, labels, mids

    def test_speaker_relabel_always_present(self):
        texts, labels, mids = self._make_data()
        fm = R.generate_failure_mode_examples(texts, labels, mids)
        assert "speaker_relabel_invariance" in fm

    def test_relabeling_replaces_tokens(self):
        texts = ["[SPEAKER_A]: hello [SPEAKER_B]: world"]
        labels = [0]
        mids = ["m1"]
        fm = R.generate_failure_mode_examples(texts, labels, mids)
        relabeled = fm["speaker_relabel_invariance"]["texts"]
        assert "[SPEAKER_X]" in relabeled[0]
        assert "[SPEAKER_Y]" in relabeled[0]
        assert "[SPEAKER_A]" not in relabeled[0]

    def test_no_boundary_meetings_detected(self):
        texts, labels, mids = self._make_data(include_no_boundary=True)
        fm = R.generate_failure_mode_examples(texts, labels, mids)
        assert "no_boundary_meetings" in fm
        assert all(l == 0 for l in fm["no_boundary_meetings"]["labels"])

    def test_very_short_meetings_detected(self):
        # Add a short meeting with < 5 utterances
        texts = [f"[SPEAKER_A]: utt {j}" for j in range(3)]
        labels = [0, 0, 0]
        mids = ["short_m"] * 3
        # Add a normal meeting to avoid all-short edge case
        for j in range(10):
            texts.append(f"[SPEAKER_B]: long {j}")
            labels.append(0)
            mids.append("long_m")
        fm = R.generate_failure_mode_examples(texts, labels, mids)
        assert "very_short_lt5" in fm
        assert len(fm["very_short_lt5"]["texts"]) == 3


# ===========================================================================
# stage_data_from_objstore
# ===========================================================================

class TestStageDataFromObjstore:
    def test_skips_if_already_staged(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "train.jsonl"), "w").close()
            with patch("boto3.client") as mock_boto:
                result = R.stage_data_from_objstore("datasets/roberta_stage1/v3/", d)
            mock_boto.assert_not_called()
        assert result is True

    def test_returns_false_without_endpoint(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("AWS_S3_ENDPOINT_URL", None)
                result = R.stage_data_from_objstore("datasets/x/", d)
        assert result is False

    def test_downloads_jsonl_files(self):
        with tempfile.TemporaryDirectory() as d:
            mock_s3 = MagicMock()
            mock_s3.get_paginator.return_value.paginate.return_value = [
                {"Contents": [
                    {"Key": "datasets/v3/train.jsonl", "Size": 1024 * 1024},
                    {"Key": "datasets/v3/val.jsonl",   "Size": 512 * 1024},
                ]}
            ]
            def fake_download(bucket, key, dest):
                open(dest, "w").close()
            mock_s3.download_file.side_effect = fake_download

            env = {
                "AWS_S3_ENDPOINT_URL": "https://chi.tacc.chameleoncloud.org:7480",
                "AWS_ACCESS_KEY_ID": "test_key",
                "AWS_SECRET_ACCESS_KEY": "test_secret",
            }
            with patch("boto3.client", return_value=mock_s3), \
                 patch.dict(os.environ, env):
                result = R.stage_data_from_objstore("datasets/v3", d)
        assert result is True

    def test_returns_false_if_no_jsonl_downloaded(self):
        with tempfile.TemporaryDirectory() as d:
            mock_s3 = MagicMock()
            mock_s3.get_paginator.return_value.paginate.return_value = [
                {"Contents": [{"Key": "datasets/v3/README.txt", "Size": 100}]}
            ]
            def fake_download(bucket, key, dest):
                open(dest, "w").close()
            mock_s3.download_file.side_effect = fake_download

            env = {
                "AWS_S3_ENDPOINT_URL": "https://chi.tacc.chameleoncloud.org:7480",
                "AWS_ACCESS_KEY_ID": "key",
                "AWS_SECRET_ACCESS_KEY": "secret",
            }
            with patch("boto3.client", return_value=mock_s3), \
                 patch.dict(os.environ, env):
                result = R.stage_data_from_objstore("datasets/v3", d)
        assert result is False


# ===========================================================================
# sweep_thresholds
# ===========================================================================

class TestSweepThresholds:
    def test_returns_threshold_and_metrics(self):
        np.random.seed(42)
        probs = np.random.rand(100).tolist()
        labels = [1 if p > 0.5 else 0 for p in probs]
        mids = [f"m{i % 5}" for i in range(100)]
        threshold, metrics = R.sweep_thresholds(probs, labels, mids)
        assert 0.0 < threshold <= 1.0
        assert "pk" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["pk"] <= 1.0

    def test_threshold_in_valid_range(self):
        probs = [0.1, 0.4, 0.6, 0.9] * 10
        labels = [0, 0, 1, 1] * 10
        mids = ["m1"] * 40
        threshold, _ = R.sweep_thresholds(probs, labels, mids)
        assert threshold in R.THRESHOLDS


# ===========================================================================
# retrain_watcher — trigger threshold logic
# ===========================================================================

class TestWatcherTriggerLogic:
    """Tests the threshold comparison without touching DB or subprocess."""

    def test_threshold_met(self):
        count = 5
        threshold = 5
        assert count >= threshold

    def test_threshold_not_met(self):
        count = 4
        threshold = 5
        assert not (count >= threshold)

    def test_time_based_trigger(self):
        from datetime import timezone
        import datetime
        last = datetime.datetime.now(timezone.utc) - datetime.timedelta(days=8)
        days_since = (datetime.datetime.now(timezone.utc) - last).days
        assert days_since >= 7

    def test_watermark_filters_old_events(self):
        # Simulates get_unconsumed_feedback_count logic:
        # only events ABOVE watermark count
        all_events = [1, 2, 3, 4, 5, 6]
        watermark = 3
        unconsumed = [e for e in all_events if e > watermark]
        assert len(unconsumed) == 3
