BEGIN;

CREATE TABLE IF NOT EXISTS dataset_quality_reports (
    quality_report_id BIGSERIAL PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    report_scope TEXT NOT NULL CHECK (
        report_scope IN ('feedback_pool', 'retraining_snapshot', 'production_live')
    ),
    report_status TEXT NOT NULL CHECK (
        report_status IN ('passed', 'failed', 'warn', 'skipped')
    ),
    dataset_version TEXT,
    reference_dataset_name TEXT,
    reference_dataset_version TEXT,
    report_path TEXT,
    share_drifted_features REAL,
    drifted_feature_count INTEGER,
    total_feature_count INTEGER,
    window_started_at TIMESTAMPTZ,
    window_ended_at TIMESTAMPTZ,
    details_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dataset_quality_reports_scope_created_at
    ON dataset_quality_reports (report_scope, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_dataset_quality_reports_dataset_name_created_at
    ON dataset_quality_reports (dataset_name, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_dataset_quality_reports_status_created_at
    ON dataset_quality_reports (report_status, created_at DESC);

COMMIT;
