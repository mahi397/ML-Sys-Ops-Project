BEGIN;

-- MLOps training / retraining history.
-- This log can capture both the initial training run and later retrains.
CREATE TABLE IF NOT EXISTS retrain_log (
    retrain_id BIGSERIAL PRIMARY KEY,
    run_type TEXT NOT NULL DEFAULT 'retrain' CHECK (
        run_type IN ('train', 'retrain')
    ),
    triggered_by_user_id TEXT REFERENCES users(user_id) ON DELETE SET NULL,
    dataset_version TEXT,
    model_version TEXT,
    corrections_used INTEGER CHECK (
        corrections_used IS NULL OR corrections_used >= 0
    ),
    high_watermark_event_id BIGINT NOT NULL DEFAULT 0 CHECK (
        high_watermark_event_id >= 0
    ),
    f1_score REAL,
    windowdiff REAL,
    pk_score REAL,
    passed_gates BOOLEAN,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    CHECK (finished_at IS NULL OR finished_at >= started_at)
);

-- Audit trail for training and retraining lifecycle events.
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id BIGSERIAL PRIMARY KEY,
    retrain_id BIGINT REFERENCES retrain_log(retrain_id) ON DELETE SET NULL,
    event_type TEXT NOT NULL,
    details JSONB,
    created_by_user_id TEXT REFERENCES users(user_id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_retrain_log_run_type_started_at
    ON retrain_log (run_type, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_retrain_log_finished_at
    ON retrain_log (finished_at DESC);

CREATE INDEX IF NOT EXISTS idx_retrain_log_triggered_by_user_id
    ON retrain_log (triggered_by_user_id);

CREATE INDEX IF NOT EXISTS idx_audit_log_event_type_created_at
    ON audit_log (event_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_log_retrain_id
    ON audit_log (retrain_id);

CREATE INDEX IF NOT EXISTS idx_audit_log_created_by_user_id
    ON audit_log (created_by_user_id);

COMMIT;
