from __future__ import annotations

import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EPSILON = 1e-6


@dataclass(frozen=True)
class DriftGateConfig:
    psi_threshold: float = 0.2
    max_drift_share: float = 0.35
    min_feature_samples: int = 25
    numeric_bin_count: int = 10


def _normalize_category(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "__NULL__"
    return str(value)


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    weight = position - lower
    return lower_value + (upper_value - lower_value) * weight


def _strict_numeric_edges(values: list[float], bin_count: int) -> list[float]:
    if not values:
        return []

    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [low - 0.5, high + 0.5]

    sorted_values = sorted(values)
    raw_edges = [low]
    for idx in range(1, bin_count):
        raw_edges.append(_quantile(sorted_values, idx / bin_count))
    raw_edges.append(high)

    edges = [raw_edges[0]]
    for edge in raw_edges[1:]:
        if edge <= edges[-1]:
            edge = edges[-1] + 1e-9
        edges.append(edge)
    return edges


def _numeric_bin_proportions(values: list[float], edges: list[float]) -> list[float]:
    if len(edges) < 2:
        return []

    counts = [0] * (len(edges) - 1)
    if not values:
        return [0.0 for _ in counts]

    for value in values:
        bucket = bisect_right(edges, value) - 1
        bucket = max(0, min(bucket, len(counts) - 1))
        counts[bucket] += 1
    total = float(len(values))
    return [count / total for count in counts]


def _categorical_proportions(values: list[Any]) -> dict[str, float]:
    normalized = [_normalize_category(value) for value in values]
    if not normalized:
        return {}
    counts = Counter(normalized)
    total = float(len(normalized))
    return {key: count / total for key, count in sorted(counts.items())}


def _psi(reference: list[float], current: list[float]) -> float:
    if not reference or not current or len(reference) != len(current):
        return 0.0
    score = 0.0
    for ref_share, cur_share in zip(reference, current):
        ref = max(ref_share, EPSILON)
        cur = max(cur_share, EPSILON)
        score += (cur - ref) * math.log(cur / ref)
    return score


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(values),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "mean": round(_safe_mean(values), 4),
    }


def extract_stage1_feature_columns(
    rows: list[dict],
    *,
    include_label: bool = True,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    columns: dict[str, list[Any]] = defaultdict(list)
    meeting_window_counts: Counter[str] = Counter()
    meeting_speakers: dict[str, set[str]] = defaultdict(set)
    meeting_utterances: dict[str, dict[str, str]] = defaultdict(dict)

    for row in rows:
        input_payload = row.get("input") or {}
        meeting_id = str(input_payload.get("meeting_id") or "")
        window = input_payload.get("window") or []
        transition_index = int(input_payload.get("transition_index") or 0)
        real_entries = [entry for entry in window if not entry.get("is_padding")]

        texts = [str(entry.get("text") or "") for entry in real_entries]
        speakers = [str(entry.get("speaker") or "") for entry in real_entries if entry.get("speaker")]
        left_entry = window[transition_index] if 0 <= transition_index < len(window) else {}
        right_entry = window[transition_index + 1] if 0 <= transition_index + 1 < len(window) else {}

        left_text = "" if left_entry.get("is_padding") else str(left_entry.get("text") or "")
        right_text = "" if right_entry.get("is_padding") else str(right_entry.get("text") or "")
        left_speaker = "" if left_entry.get("is_padding") else str(left_entry.get("speaker") or "")
        right_speaker = "" if right_entry.get("is_padding") else str(right_entry.get("speaker") or "")

        columns["window_real_utterances"].append(len(real_entries))
        columns["window_unique_speakers"].append(len(set(speakers)))
        columns["window_total_chars"].append(sum(len(text) for text in texts))
        columns["window_total_tokens"].append(sum(len(text.split()) for text in texts))
        columns["left_chars"].append(len(left_text))
        columns["right_chars"].append(len(right_text))
        columns["meeting_offset_seconds"].append(float(input_payload.get("meeting_offset_seconds") or 0.0))
        columns["speaker_change"].append(bool(left_speaker and right_speaker and left_speaker != right_speaker))

        if include_label and row.get("output") is not None:
            columns["label"].append(row["output"].get("label"))

        if not meeting_id:
            continue

        meeting_window_counts[meeting_id] += 1
        for entry in real_entries:
            speaker = str(entry.get("speaker") or "")
            if speaker:
                meeting_speakers[meeting_id].add(speaker)

            model_utterance_id = str(entry.get("model_utterance_id") or "")
            text = str(entry.get("text") or "")
            if model_utterance_id and model_utterance_id not in meeting_utterances[meeting_id]:
                meeting_utterances[meeting_id][model_utterance_id] = text

    for meeting_id, window_count in meeting_window_counts.items():
        utterance_texts = list(meeting_utterances[meeting_id].values())
        char_lengths = [len(text) for text in utterance_texts]
        token_lengths = [len(text.split()) for text in utterance_texts]

        columns["meeting_window_count"].append(window_count)
        columns["meeting_model_utterance_count"].append(len(utterance_texts))
        columns["meeting_unique_speakers"].append(len(meeting_speakers[meeting_id]))
        columns["meeting_avg_chars_per_utterance"].append(_safe_mean(char_lengths))
        columns["meeting_avg_tokens_per_utterance"].append(_safe_mean(token_lengths))

    metadata = {
        "row_count": len(rows),
        "meeting_count": len(meeting_window_counts),
        "feature_count": len(columns),
    }
    return dict(columns), metadata


def extract_live_feature_columns(
    model_utterances_by_meeting: dict[str, list[dict]],
    *,
    window_size: int,
    transition_index: int,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    columns: dict[str, list[Any]] = defaultdict(list)

    for meeting_id, utterances in model_utterances_by_meeting.items():
        if not utterances:
            continue

        speakers = {str(utterance.get("speaker_label") or "") for utterance in utterances if utterance.get("speaker_label")}
        texts = [str(utterance.get("text") or "") for utterance in utterances]
        char_lengths = [len(text) for text in texts]
        token_lengths = [len(text.split()) for text in texts]

        columns["meeting_model_utterance_count"].append(len(utterances))
        columns["meeting_unique_speakers"].append(len(speakers))
        columns["meeting_avg_chars_per_utterance"].append(_safe_mean(char_lengths))
        columns["meeting_avg_tokens_per_utterance"].append(_safe_mean(token_lengths))
        columns["meeting_window_count"].append(max(0, len(utterances) - 1))

        for left_idx in range(len(utterances) - 1):
            right_idx = left_idx + 1
            start_index = left_idx - transition_index
            window_entries = []

            for pos in range(window_size):
                idx = start_index + pos
                if 0 <= idx < len(utterances):
                    window_entries.append(utterances[idx])

            left = utterances[left_idx]
            right = utterances[right_idx]
            window_texts = [str(entry.get("text") or "") for entry in window_entries]
            window_speakers = {
                str(entry.get("speaker_label") or "")
                for entry in window_entries
                if entry.get("speaker_label")
            }

            columns["window_real_utterances"].append(len(window_entries))
            columns["window_unique_speakers"].append(len(window_speakers))
            columns["window_total_chars"].append(sum(len(text) for text in window_texts))
            columns["window_total_tokens"].append(sum(len(text.split()) for text in window_texts))
            columns["left_chars"].append(len(str(left.get("text") or "")))
            columns["right_chars"].append(len(str(right.get("text") or "")))
            columns["meeting_offset_seconds"].append(float(left.get("start_time_sec") or 0.0))
            columns["speaker_change"].append(
                bool(
                    left.get("speaker_label")
                    and right.get("speaker_label")
                    and left.get("speaker_label") != right.get("speaker_label")
                )
            )

    metadata = {
        "row_count": len(columns.get("window_real_utterances", [])),
        "meeting_count": len(model_utterances_by_meeting),
        "feature_count": len(columns),
    }
    return dict(columns), metadata


def build_reference_profile(
    feature_columns: dict[str, list[Any]],
    *,
    metadata: dict[str, Any] | None = None,
    bin_count: int = 10,
) -> dict[str, Any]:
    profile_features: dict[str, dict[str, Any]] = {}

    for feature_name, values in sorted(feature_columns.items()):
        if not values:
            continue

        numeric_values: list[float] = []
        is_numeric = True
        for value in values:
            try:
                numeric_values.append(float(value))
            except (TypeError, ValueError):
                is_numeric = False
                break

        if is_numeric:
            edges = _strict_numeric_edges(numeric_values, bin_count=max(2, bin_count))
            profile_features[feature_name] = {
                "type": "numeric",
                "summary": _summary(numeric_values),
                "bins": {
                    "edges": [round(edge, 6) for edge in edges],
                    "reference_proportions": [
                        round(prop, 6) for prop in _numeric_bin_proportions(numeric_values, edges)
                    ],
                },
            }
            continue

        proportions = _categorical_proportions(values)
        profile_features[feature_name] = {
            "type": "categorical",
            "summary": {"count": len(values), "cardinality": len(proportions)},
            "categories": {key: round(value, 6) for key, value in proportions.items()},
        }

    return {
        "metadata": metadata or {},
        "feature_count": len(profile_features),
        "features": profile_features,
    }


def compare_feature_columns_to_reference(
    *,
    reference_profile: dict[str, Any] | None,
    current_feature_columns: dict[str, list[Any]],
    current_metadata: dict[str, Any] | None = None,
    config: DriftGateConfig | None = None,
) -> dict[str, Any]:
    gate = config or DriftGateConfig()
    current_metadata = current_metadata or {}

    if not reference_profile:
        return {
            "status": "skipped",
            "reason": "missing_reference_profile",
            "gate": {
                "psi_threshold": gate.psi_threshold,
                "max_drift_share": gate.max_drift_share,
                "min_feature_samples": gate.min_feature_samples,
            },
            "current_metadata": current_metadata,
            "feature_reports": {},
            "drifted_feature_count": 0,
            "total_feature_count": 0,
            "share_drifted_features": 0.0,
        }

    feature_reports: dict[str, dict[str, Any]] = {}
    compared_feature_count = 0
    drifted_feature_count = 0

    for feature_name, reference_feature in sorted((reference_profile.get("features") or {}).items()):
        current_values = current_feature_columns.get(feature_name)
        if not current_values:
            continue

        if len(current_values) < gate.min_feature_samples:
            feature_reports[feature_name] = {
                "status": "skipped",
                "reason": "insufficient_current_samples",
                "current_sample_count": len(current_values),
            }
            continue

        if reference_feature.get("type") == "numeric":
            numeric_values = [float(value) for value in current_values]
            edges = [float(edge) for edge in reference_feature["bins"]["edges"]]
            current_props = _numeric_bin_proportions(numeric_values, edges)
            reference_props = [
                float(prop) for prop in reference_feature["bins"]["reference_proportions"]
            ]
            psi_score = _psi(reference_props, current_props)
            drifted = psi_score > gate.psi_threshold
            compared_feature_count += 1
            drifted_feature_count += int(drifted)
            feature_reports[feature_name] = {
                "status": "evaluated",
                "type": "numeric",
                "psi": round(psi_score, 6),
                "threshold": gate.psi_threshold,
                "drifted": drifted,
                "reference_summary": reference_feature.get("summary", {}),
                "current_summary": _summary(numeric_values),
                "current_proportions": [round(prop, 6) for prop in current_props],
            }
            continue

        current_props = _categorical_proportions(current_values)
        reference_categories = {
            key: float(value)
            for key, value in (reference_feature.get("categories") or {}).items()
        }
        all_categories = sorted(set(reference_categories) | set(current_props))
        reference_vector = [reference_categories.get(key, 0.0) for key in all_categories]
        current_vector = [current_props.get(key, 0.0) for key in all_categories]
        psi_score = _psi(reference_vector, current_vector)
        drifted = psi_score > gate.psi_threshold
        compared_feature_count += 1
        drifted_feature_count += int(drifted)
        feature_reports[feature_name] = {
            "status": "evaluated",
            "type": "categorical",
            "psi": round(psi_score, 6),
            "threshold": gate.psi_threshold,
            "drifted": drifted,
            "reference_categories": reference_categories,
            "current_categories": {key: round(value, 6) for key, value in current_props.items()},
        }

    share_drifted = (
        drifted_feature_count / compared_feature_count if compared_feature_count else 0.0
    )
    status = "failed" if compared_feature_count and share_drifted > gate.max_drift_share else "passed"
    if compared_feature_count == 0:
        status = "skipped"

    return {
        "status": status,
        "gate": {
            "psi_threshold": gate.psi_threshold,
            "max_drift_share": gate.max_drift_share,
            "min_feature_samples": gate.min_feature_samples,
        },
        "reference_metadata": reference_profile.get("metadata", {}),
        "current_metadata": current_metadata,
        "feature_reports": feature_reports,
        "drifted_feature_count": drifted_feature_count,
        "total_feature_count": compared_feature_count,
        "share_drifted_features": round(share_drifted, 6),
    }


def load_reference_profile(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None
