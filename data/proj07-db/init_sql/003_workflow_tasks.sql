BEGIN;

CREATE TABLE IF NOT EXISTS workflow_tasks (
    task_id BIGSERIAL PRIMARY KEY,
    task_type TEXT NOT NULL,
    meeting_id TEXT NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    artifact_version INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN (
            'pending',
            'running',
            'retry_scheduled',
            'succeeded',
            'failed_permanent',
            'cancelled'
        )
    ),
    payload_json JSONB,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    max_attempts INTEGER NOT NULL DEFAULT 8 CHECK (max_attempts > 0),
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    locked_by TEXT,
    locked_at TIMESTAMPTZ,
    heartbeat_at TIMESTAMPTZ,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (task_type, meeting_id, artifact_version)
);

CREATE TABLE IF NOT EXISTS workflow_task_attempts (
    attempt_id BIGSERIAL PRIMARY KEY,
    task_id BIGINT NOT NULL REFERENCES workflow_tasks(task_id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
    worker_id TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    outcome TEXT,
    error_summary TEXT,
    stderr_tail TEXT,
    duration_ms BIGINT,
    UNIQUE (task_id, attempt_number)
);

CREATE INDEX IF NOT EXISTS idx_workflow_tasks_status_next_attempt_at
    ON workflow_tasks (status, next_attempt_at);

CREATE INDEX IF NOT EXISTS idx_workflow_tasks_task_type_status_next_attempt_at
    ON workflow_tasks (task_type, status, next_attempt_at);

CREATE INDEX IF NOT EXISTS idx_workflow_tasks_meeting_id
    ON workflow_tasks (meeting_id);

CREATE INDEX IF NOT EXISTS idx_workflow_task_attempts_task_id
    ON workflow_task_attempts (task_id);

COMMIT;
