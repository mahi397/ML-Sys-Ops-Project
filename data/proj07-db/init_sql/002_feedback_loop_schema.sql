BEGIN;

ALTER TABLE topic_segments DROP CONSTRAINT IF EXISTS topic_segments_segment_type_check;
ALTER TABLE topic_segments
ADD CONSTRAINT topic_segments_segment_type_check
CHECK (segment_type IN ('gold', 'predicted', 'user_corrected'));

ALTER TABLE meeting_artifacts DROP CONSTRAINT IF EXISTS meeting_artifacts_artifact_type_check;
ALTER TABLE meeting_artifacts
ADD CONSTRAINT meeting_artifacts_artifact_type_check
CHECK (
    artifact_type IN (
        'raw_transcript',
        'parsed_transcript',
        'summary_json',
        'stage1_requests_jsonl',
        'stage1_requests_json',
        'stage1_model_utterances_json',
        'stage1_manifest_json',
        'stage1_responses_jsonl',
        'stage1_responses_json',
        'stage2_inputs_jsonl',
        'stage2_inputs_json',
        'reconstructed_segments_json',
        'stage2_responses_jsonl',
        'stage2_responses_json',
        'feedback_json',
        'other'
    )
);

CREATE TABLE IF NOT EXISTS segment_summaries (
    segment_summary_id BIGSERIAL PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    topic_segment_id BIGINT NOT NULL REFERENCES topic_segments(topic_segment_id) ON DELETE CASCADE,
    summary_id BIGINT NOT NULL REFERENCES summaries(summary_id) ON DELETE CASCADE,
    segment_index INTEGER NOT NULL,
    topic_label TEXT,
    summary_bullets JSONB NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('complete', 'draft')),
    model_name TEXT,
    model_version TEXT,
    prompt_version TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (summary_id, segment_index)
);

CREATE TABLE IF NOT EXISTS feedback_events (
    feedback_event_id BIGSERIAL PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    summary_id BIGINT REFERENCES summaries(summary_id) ON DELETE SET NULL,
    segment_summary_id BIGINT REFERENCES segment_summaries(segment_summary_id) ON DELETE SET NULL,
    event_type TEXT NOT NULL CHECK (
        event_type IN (
            'accept_summary',
            'edit_topic_label',
            'edit_summary_bullets',
            'merge_segments',
            'split_segment',
            'boundary_correction'
        )
    ),
    event_source TEXT NOT NULL CHECK (event_source IN ('emulated', 'user')),
    before_payload JSONB,
    after_payload JSONB,
    created_by_user_id TEXT REFERENCES users(user_id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dataset_versions (
    dataset_version_id BIGSERIAL PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    stage TEXT NOT NULL CHECK (stage IN ('stage1', 'stage2')),
    source_type TEXT NOT NULL CHECK (source_type IN ('production_feedback', 'ami', 'synthetic')),
    object_key TEXT NOT NULL,
    manifest_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_segment_summaries_meeting_id
    ON segment_summaries(meeting_id);

CREATE INDEX IF NOT EXISTS idx_segment_summaries_summary_id
    ON segment_summaries(summary_id);

CREATE INDEX IF NOT EXISTS idx_feedback_events_meeting_id
    ON feedback_events(meeting_id);

CREATE INDEX IF NOT EXISTS idx_feedback_events_summary_id
    ON feedback_events(summary_id);

CREATE INDEX IF NOT EXISTS idx_feedback_events_segment_summary_id
    ON feedback_events(segment_summary_id);

CREATE INDEX IF NOT EXISTS idx_dataset_versions_stage_name
    ON dataset_versions(stage, dataset_name);

COMMIT;