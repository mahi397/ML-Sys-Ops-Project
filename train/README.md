# Training Subsystem — NeuralOps / Jitsi Topic Segmentation

**Role:** Model Training & Retraining  
**Stack:** PyTorch + HuggingFace Transformers + Ray Train + MLflow  
**Node:** CHI@UC RTX 6000 bare-metal GPU

---

## Overview

This folder owns the full training and automated retraining pipeline for the **RoBERTa-based topic boundary classifier** that powers Jitsi meeting recap. The model predicts, for each utterance transition in a meeting transcript, whether a new topic segment begins at that point.

The pipeline has two modes:

- **Initial training** — run manually to produce a candidate model from the AMI corpus. All historical runs are tracked in MLflow under the `jitsi-topic-segmentation` experiment.
- **Automated retraining** — triggered by `retrain_watcher.py` when enough user boundary corrections accumulate in the `feedback_events` table. Runs unattended, logs to the `retraining` MLflow experiment, and registers a new `candidate` model if quality gates pass.

---

## Files

| File | Purpose |
|---|---|
| `retrain.py` | Main training script. Runs a full PyTorch fine-tuning loop via Ray Train. Evaluates on aggregate metrics + per-slice fairness + known failure modes. Registers model to MLflow if gates pass. |
| `retrain_watcher.py` | Long-running daemon. Polls `feedback_events` table every 5 minutes. Triggers `retrain.py` when correction count exceeds threshold or time since last retrain exceeds max days. |
| `train.py` | Original standalone training script (initial implementation). Kept for reference. |
| `hparam_sweep.py` | Optuna hyperparameter sweep. Used to produce the current production model (Trial #10, test_pk=0.213). |
| `best_params_sweep.yaml` | Best hyperparameter config from Optuna sweep. |
| `configs/` | Additional YAML configs for different training scenarios. |
| `Dockerfile` | Container image for both `retrain-watcher` and `retrain-job` services. |
| `setup_train.sh` | One-time setup script for a fresh RTX node. See Setup below. |

---

## Architecture

```
feedback_events (Postgres)
        │
        ▼
retrain_watcher.py          ← always running, polls every 5 min
        │  (threshold met)
        ▼
retrain.py                  ← Ray Train, single GPU worker
        │
        ├── resolve_dataset_path()   ← queries dataset_versions table
        │         └── rclone copy chi_tacc:objstore-proj07/datasets/roberta_stage1/vN/
        │
        ├── resolve_feedback_path()  ← queries dataset_versions table
        │         └── rclone copy chi_tacc:objstore-proj07/datasets/roberta_stage1_feedback_pool/vN/
        │
        ├── train_func()             ← Ray Train worker (GPU)
        │         ├── AMI examples + feedback examples (2x weighted)
        │         ├── WeightedRandomSampler (handles ~40:1 class imbalance)
        │         ├── AdamW + linear warmup scheduler
        │         └── per-epoch threshold sweep on val set (minimise Pk)
        │
        └── evaluate_and_register()  ← runs on driver after training
                  ├── Aggregate gates: F1 ≥ 0.20, Pk ≤ 0.25, WD ≤ 0.40
                  ├── Fairness gates: per-slice Pk ≤ 0.40 (meeting size + speaker count)
                  ├── Robustness tests: very short meetings, no-boundary meetings, speaker relabeling
                  ├── Model card: logged as MLflow artifact
                  └── If gates pass → registered as 'candidate' in MLflow registry
```

---

## Model Registry

Three aliases in MLflow under `jitsi-topic-segmenter`:

| Alias | Description | test_pk | test_f1 |
|---|---|---|---|
| `production` | Optuna best (Trial #10/20) | 0.213 | 0.232 |
| `fallback` | distilroberta-base full fine-tune | 0.228 | 0.222 |
| `candidate` | Set automatically after a passing retrain | — | — |

Promotion from `candidate` → `production` is a **manual step** in the MLflow UI. Rollback is handled by Shruti's serving layer — if correction rate spikes post-deployment, the serving API switches back to `fallback`.

---

## Quality Gates

Gates are calibrated to the initial training run results and applied in `evaluate_and_register()`:

**Aggregate gates** (all must pass):
- `test_f1 ≥ 0.20`
- `test_pk ≤ 0.25` (Pk: probability of misclassifying a random utterance pair)
- `test_window_diff ≤ 0.40`

**Fairness gate** (safeguarding):
- No data-sufficient slice may have `Pk > 0.40`
- Slices: meeting size (short/medium/long) × speaker count (1/2/3+)
- Ensures the model does not degrade significantly on specific meeting types

**Robustness tests** (warn only, do not block registration):
- Very short meetings (<5 transitions)
- No-boundary meetings (single topic)
- Speaker relabeling invariance (SPEAKER_A→SPEAKER_X should not degrade Pk)

If any aggregate or fairness gate fails, the model is logged to MLflow as `model-failed` (not registered). All runs produce a model card regardless of gate outcome.

---

## Retraining Trigger Logic

`retrain_watcher.py` triggers a retrain when **any** of these conditions are met:

1. `COUNT(feedback_events WHERE event_type IN ('merge_segments', 'split_segment') AND event_source = 'user' AND feedback_event_id > watermark) >= RETRAIN_THRESHOLD`
2. Days since last successful retrain ≥ `MAX_DAYS_BETWEEN_RETRAINS`
3. No retrain has ever run and at least one correction exists

After each retrain (pass or fail), the watermark is advanced to the current max `feedback_event_id` so the same events are never counted twice.

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `RETRAIN_THRESHOLD` | `5` (demo) / `500` (prod) | Feedback events needed to trigger |
| `RETRAIN_CHECK_INTERVAL_SECONDS` | `300` | Poll interval |
| `MAX_DAYS_BETWEEN_RETRAINS` | `30` | Time-based trigger |
| `DATABASE_URL` | see compose | Postgres connection |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server |
| `TOKENIZERS_PARALLELISM` | `false` | Avoids Ray fork warnings |
| `DATASET_VERSION` | set by watcher | Passed to retrain.py for audit |

---

## Dataset

Training data is pulled from CHI@TACC object storage (`chi_tacc` rclone remote) at runtime:

| Dataset | Path | Description |
|---|---|---|
| `roberta_stage1/v2` | `chi_tacc:objstore-proj07/datasets/roberta_stage1/v2/` | AMI corpus + synthetic augmentation. Used for retraining baseline. |
| `roberta_stage1_feedback_pool/v1` | `chi_tacc:objstore-proj07/datasets/roberta_stage1_feedback_pool/v1/` | User boundary corrections converted to training examples. |

Dataset versions are tracked in the `dataset_versions` table in `proj07_sql_db`. `retrain.py` always queries this table for the latest version — when Aneesh's data pipeline produces a new snapshot, it inserts a new row and the next retrain automatically picks it up.

Data split is **strictly by `meeting_id`** (70/15/15 train/val/test) to prevent leakage. AMI meetings are permanently assigned to splits at v1 creation and never reassigned.

---

## Safeguarding

| Principle | Implementation |
|---|---|
| **Fairness** | Slice evaluation (meeting size + speaker count buckets). Fairness gate blocks registration if any slice has Pk > 0.40. |
| **Robustness** | Known failure mode tests on short meetings, single-topic meetings, speaker relabeling. Results in model card. |
| **Transparency** | Model card JSON artifact logged on every run. Documents training data, thresholds, fairness results, privacy notes, accountability chain. |
| **Accountability** | Every retrain event written to `audit_log` and `retrain_log` tables with watermark, correction count, gates passed, and MLflow run ID. |
| **Privacy** | Speaker identity abstracted to `[SPEAKER_A/B/C]` tokens. No participant names or IDs in training data. |
| **Threshold** | `best_threshold` tag set on registered model version. Serving reads it from registry — never hardcoded. |

---

## Setup (Fresh Node)

```bash
# 1. Clone repo and fill in .env
git clone https://github.com/mahi397/ML-Sys-Ops-Project.git
cd ML-Sys-Ops-Project
cp .env.example .env
# edit .env: FLOATING_IP, AWS keys, POSTGRES_PASSWORD, MINIO_PASSWORD

# 2. Run unified setup script (from repo root)
chmod +x setup.sh
bash setup.sh
```

`setup.sh` handles: Docker install, rclone install + validation, NVIDIA toolkit, block storage layout, data staging from object storage, ML model downloads (RoBERTa + Mistral), Postgres init + schema migrations, `dataset_versions` seeding, MLflow startup + model registry restore, full `docker compose up -d`. Optionally deploys Jitsi with `DEPLOY_JITSI=true bash setup.sh`.

**Prerequisites before running setup:**
- `~/.config/rclone/rclone.conf` configured with `chi_tacc` remote (CHI@TACC S3)
- `~/restore_mlflow.py` present (for model registry restore)
- `.env` filled in

---

## Running a Manual Retrain

```bash
# From inside the retrain-watcher container
docker compose exec retrain-watcher python /app/retrain.py

# With custom args
docker compose exec retrain-watcher python /app/retrain.py --epochs 3 --data_dir /mnt/block/roberta_stage1/v2
```

---

## Testing the Retraining Trigger

```bash
# 1. Insert a test meeting
docker compose exec postgres psql -U proj07_user -d proj07_sql_db -c "
INSERT INTO meetings (meeting_id, source_type, source_name, started_at, ended_at)
VALUES ('test-001', 'jitsi', 'Test Meeting', NOW(), NOW());"

# 2. Insert 5 user feedback events (event_source='user' only counts toward trigger)
docker compose exec postgres psql -U proj07_user -d proj07_sql_db -c "
INSERT INTO feedback_events (meeting_id, event_type, event_source, before_payload, after_payload)
VALUES
  ('test-001', 'merge_segments', 'user', '{\"transition_index\": 1}', '{\"label\": 0}'),
  ('test-001', 'merge_segments', 'user', '{\"transition_index\": 2}', '{\"label\": 0}'),
  ('test-001', 'split_segment',  'user', '{\"transition_index\": 3}', '{\"label\": 1}'),
  ('test-001', 'merge_segments', 'user', '{\"transition_index\": 4}', '{\"label\": 0}'),
  ('test-001', 'split_segment',  'user', '{\"transition_index\": 5}', '{\"label\": 1}');"

# 3. Restart watcher to trigger immediately (otherwise waits up to 5 min)
docker compose restart retrain-watcher
docker compose logs retrain-watcher -f
```

---

## Verifying a Completed Retrain

```bash
# Check retrain_log
docker compose exec postgres psql -U proj07_user -d proj07_sql_db -c \
  "SELECT retrain_id, dataset_version, corrections_used, passed_gates, high_watermark_event_id, finished_at FROM retrain_log ORDER BY retrain_id DESC LIMIT 5;"

# Check audit_log
docker compose exec postgres psql -U proj07_user -d proj07_sql_db -c \
  "SELECT event_type, details FROM audit_log ORDER BY audit_id DESC LIMIT 5;"

# Check MLflow — new run in 'retraining' experiment
# http://<FLOATING_IP>:5000
# If gates passed: jitsi-topic-segmenter 'candidate' alias set in Model Registry
```

---

## Promoting a Candidate to Production

After a passing retrain, `candidate` alias is set automatically. Promotion to `production` is a **manual approval step**:

1. Open MLflow at `http://<FLOATING_IP>:5000`
2. Go to **Models → jitsi-topic-segmenter**
3. Find the version with alias `candidate`
4. Review metrics and model card artifact
5. Set alias `production` on that version
6. Serving API picks up the new model on next request (no restart needed)

---

## Integration Boundaries

| Dependency | Owner | Contract |
|---|---|---|
| `feedback_events` table | Aneesh (Data) | `retrain_watcher.py` reads `merge_segments`/`split_segment` events with `event_source='user'` |
| `dataset_versions` table | Aneesh (Data) | `retrain.py` queries latest `roberta_stage1` row for training data path |
| MLflow model registry | Shared | `retrain.py` writes `candidate` alias; Shruti's serving reads `production` alias |
| `retrain_log` / `audit_log` | Training (writes) | Shruti's monitoring reads correction rate to detect model degradation |
| Ray cluster | Shared | Single `ray-head` container. Shruti's Ray Serve + this Ray Train share the same cluster and dashboard. |

---

## Evaluation Test Suite

All evaluation runs automatically inside `evaluate_and_register()` after every training run — initial or retrain. Results are logged to MLflow as metrics and captured in the model card artifact. The test suite follows three layers aligned with the offline evaluation lab practices.

---

### Layer 1 — Aggregate Metrics (held-out test set)

Evaluated on the held-out test split (15% of AMI meetings, strictly split by `meeting_id` — no meeting spans train/val/test).

| Metric | What it measures | Gate threshold |
|---|---|---|
| **Pk** | Probability of misclassifying a random pair of utterances as same/different topic segment. Lower is better; random baseline = 0.5. | ≤ 0.25 (**blocks registration if fails**) |
| **WindowDiff** | Penalises off-by-one boundary placement errors, stricter than Pk for near-miss predictions. | ≤ 0.40 (**blocks registration if fails**) |
| **F1** | Per-transition boundary/non-boundary classification score. Low by design due to ~40:1 class imbalance. | ≥ 0.20 (**blocks registration if fails**) |
| **Precision** | Of predicted boundaries, how many are real boundaries. | Logged, no gate |
| **Recall** | Of real boundaries, how many were predicted. | Logged, no gate |
| **best_threshold** | Probability cutoff that minimises Pk on the val set (swept over [0.05..0.50]). Tagged on the registered model version so serving never hardcodes 0.5. | Logged + tagged |

Metrics are computed per-meeting then averaged, not globally, to avoid long-meeting dominance.

---

### Layer 2 — Slice Evaluation (fairness safeguarding)

Following the offline evaluation lab principle of evaluating on slices of interest to identify potential unfairness. Pk and F1 are broken down across two dimensions:

**Meeting size slices:**

| Slice | Definition | Rationale |
|---|---|---|
| `short_lt15` | Meetings with < 15 utterance transitions | Short standups / 1:1 calls — fewer transitions means less signal for segmentation |
| `medium_15to40` | 15–40 transitions | Typical meeting length |
| `long_gt40` | > 40 transitions | Long all-hands or design reviews |

**Speaker count slices:**

| Slice | Definition | Rationale |
|---|---|---|
| `single_speaker` | 1 unique speaker | Monologue-style meetings — model relies entirely on lexical cues, no speaker-change signal |
| `two_speaker` | 2 unique speakers | Standard 1:1 |
| `multi_speaker_3plus` | 3+ unique speakers | Panel discussions, team meetings |

**Fairness gate:** No data-sufficient slice (≥ 10 examples) may have Pk > 0.40. This is set higher than the aggregate gate (0.25) to account for small-slice noise while still catching systematic failures on specific meeting types. If any slice fails, the model is **not registered**.

All slice metrics are logged to MLflow as `slice_pk_<name>` and `slice_f1_<name>`.

---

### Layer 3 — Known Failure Mode Tests (robustness)

Following the offline evaluation lab principle of evaluating on known failure modes and creating an automated test suite. Three documented hard cases for topic segmentation are tested on every run. These are **warn-only** — they do not block registration but are surfaced in the model card and MLflow so degradation is visible over time.

**FM 1 — Very short meetings (`very_short_lt5`)**
- **What:** Meetings with fewer than 5 utterance transitions, synthesised from the test set.
- **Why:** The model has almost no context to work with. A known failure mode is predicting all boundaries or no boundaries on these inputs.
- **Threshold:** Pk ≤ 0.50 (relaxed, genuine difficulty with so few transitions). Warn if exceeded.
- **MLflow metric:** `fm_pk_very_short_lt5`

**FM 2 — No-boundary meetings (`no_boundary_meetings`)**
- **What:** Meetings where all gold labels are 0 (single-topic, no real boundaries), synthesised from test set.
- **Why:** The model should predict all-zero for these. Predicting many false boundaries here is a known failure mode — it maps to users seeing over-segmented recaps for focused meetings.
- **Threshold:** Pk ≤ 0.35. Warn if exceeded.
- **MLflow metric:** `fm_pk_no_boundary_meetings`

**FM 3 — Speaker relabeling invariance (`speaker_relabel_invariance`)**
- **What:** Template-based perturbation test. All `[SPEAKER_A]` tokens replaced with `[SPEAKER_X]` and `[SPEAKER_B]` with `[SPEAKER_Y]` in the test texts. Gold labels are unchanged.
- **Why:** Topic boundaries depend on *what* is said, not *who* is saying it. If Pk degrades significantly after relabeling, the model is over-relying on speaker identity tokens rather than content — a form of spurious correlation that would fail on meetings with unusual speaker labels.
- **Threshold:** Pk ≤ 0.30. Warn if exceeded.
- **MLflow metric:** `fm_pk_speaker_relabel_invariance`
- **Special tokens:** `[SPEAKER_X]` and `[SPEAKER_Y]` are added to the tokenizer vocabulary at training time so these perturbation tests run without UNK tokens.

---

### Model Card

Every training run — pass or fail — produces a `model_card.json` artifact logged to MLflow under `artifacts/model_card/`. It documents:

- Training data provenance (dataset version, feedback included, split strategy, leakage controls)
- Model hyperparameters and warm-start details
- Inference threshold and serving note (read from registry tag, not hardcoded)
- Aggregate test metrics
- Quality gate thresholds and pass/fail status
- Full fairness slice evaluation results
- Robustness failure mode test results and WARN flags
- Privacy handling (speaker PII abstraction, feedback data handling, data retention, consent)
- Accountability chain (training owner, serving owner, data owner, promotion process, rollback trigger, audit log reference)
- Known limitations

The model card is the primary transparency artifact for the system and covers the TRANSPARENCY, ACCOUNTABILITY, FAIRNESS, PRIVACY, and ROBUSTNESS safeguarding principles from the course.

---

### Metrics Logged to MLflow Per Run

| MLflow metric | Description |
|---|---|
| `test_f1` | Aggregate F1 on test set |
| `test_pk` | Aggregate Pk on test set |
| `test_window_diff` | Aggregate WindowDiff on test set |
| `test_precision` | Aggregate precision on test set |
| `test_recall` | Aggregate recall on test set |
| `best_threshold` | Val-set optimal probability threshold |
| `gates_passed` | 1 if all aggregate + fairness gates passed |
| `fairness_gate_passed` | 1 if all slice fairness gates passed |
| `slice_pk_<name>` | Per-slice Pk (6 slices) |
| `slice_f1_<name>` | Per-slice F1 (6 slices) |
| `fm_pk_very_short_lt5` | Failure mode: very short meetings |
| `fm_pk_no_boundary_meetings` | Failure mode: single-topic meetings |
| `fm_pk_speaker_relabel_invariance` | Failure mode: speaker token perturbation |

---

## Standalone Evaluation Scripts

### `offline_eval.py`

Re-evaluates any registered model version against the held-out test set without retraining. Runs the full test suite (aggregate metrics, slice evaluation, speaker relabeling invariance) and logs results to MLflow under the `offline-evaluation` experiment.

```bash
# Evaluate production model
docker compose exec retrain-watcher python /app/offline_eval.py

# Evaluate candidate before promoting
docker compose exec retrain-watcher python /app/offline_eval.py --model_alias candidate

# Compare candidate vs production
docker compose exec retrain-watcher python /app/offline_eval.py \
    --model_alias candidate --compare_to production

# Evaluate on a specific data dir
docker compose exec retrain-watcher python /app/offline_eval.py \
    --data_dir /mnt/block/roberta_stage1/v2
```

### `online_eval.py`

Queries `feedback_events` and `utterance_transitions` from Postgres to compute online quality metrics based on real user corrections. Logs results to MLflow under the `online-evaluation` experiment.

**Metrics computed:**

| Metric | Description | Alert threshold |
|---|---|---|
| `online_correction_rate` | Corrections / total predictions | > 0.15 |
| `online_meeting_correction_rate` | Meetings with ≥1 correction / total meetings | — |
| `online_fpr` | `merge_segments` events / predicted positives | > 0.10 |
| `online_fnr` | `split_segment` events / predicted negatives | > 0.10 |
| `corrections_per_meeting_v<N>` | Per model version correction rate | — |

```bash
# Evaluate last 30 days
docker compose exec retrain-watcher python /app/online_eval.py

# Evaluate last 7 days (post-deployment check)
docker compose exec retrain-watcher python /app/online_eval.py --days 7

# Evaluate a specific date range
docker compose exec retrain-watcher python /app/online_eval.py \
    --since 2026-04-01 --until 2026-04-20
```

**As a persistent service (runs every hour, logs to MLflow automatically):**
```bash
docker compose --profile monitoring up -d online-eval
docker compose logs online-eval -f
```
Results appear in MLflow under the `online-evaluation` experiment at `http://<IP>:5000`.

Run `online_eval.py` on demand or keep `online-eval` service running to track model quality over time. If `online_correction_rate > 0.15`, consider triggering a manual retrain or rolling back to `fallback`.
