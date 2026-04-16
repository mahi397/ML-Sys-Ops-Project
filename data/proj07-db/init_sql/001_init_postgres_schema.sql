BEGIN;

-- Initial PostgreSQL schema for:
-- AMI raw/processed metadata
-- Jitsi raw meeting artifacts
-- Stage 1 topic-boundary detection inputs/outputs
-- Stage 2 topic-segmented LLM summarization inputs/outputs

CREATE TABLE users (
    user_id BIGSERIAL PRIMARY KEY,
    display_name TEXT NOT NULL,
    email TEXT UNIQUE
);

CREATE TABLE meetings (
    meeting_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL CHECK (source_type IN ('ami', 'jitsi')),
    source_name TEXT NOT NULL,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    raw_folder_prefix TEXT,
    dataset_version INTEGER,
    dataset_split TEXT CHECK (
        dataset_split IS NULL
        OR dataset_split IN ('train', 'val', 'test')
    )
);

CREATE TABLE meeting_participants (
    meeting_participant_id BIGSERIAL PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('host', 'participant')),
    can_view_summary BOOLEAN NOT NULL DEFAULT TRUE,
    can_edit_summary BOOLEAN NOT NULL DEFAULT FALSE,
    joined_at TIMESTAMPTZ,
    left_at TIMESTAMPTZ,
    UNIQUE (meeting_id, user_id),
    CHECK (can_view_summary OR NOT can_edit_summary)
);

CREATE TABLE meeting_speakers (
    meeting_speaker_id BIGSERIAL PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    user_id BIGINT REFERENCES users(user_id) ON DELETE SET NULL,
    speaker_label TEXT NOT NULL,
    display_name TEXT NOT NULL,
    role TEXT,
    UNIQUE (meeting_id, speaker_label)
);

CREATE TABLE meeting_artifacts (
    artifact_id BIGSERIAL PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    artifact_type TEXT NOT NULL,
    object_key TEXT NOT NULL,
    content_type TEXT,
    artifact_version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (meeting_id, artifact_type, artifact_version),
    CONSTRAINT meeting_artifacts_artifact_type_check CHECK (
        artifact_type IN (
            'raw_transcript',
            'parsed_transcript',
            'summary_json',
            'other'
        )
    )
);

CREATE TABLE utterances (
    utterance_id BIGSERIAL PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    meeting_speaker_id BIGINT NOT NULL REFERENCES meeting_speakers(meeting_speaker_id) ON DELETE RESTRICT,
    utterance_index INTEGER NOT NULL,
    start_time_sec DOUBLE PRECISION NOT NULL,
    end_time_sec DOUBLE PRECISION NOT NULL,
    raw_text TEXT NOT NULL,
    clean_text TEXT,
    source_segment_id TEXT,
    UNIQUE (meeting_id, utterance_index),
    CHECK (end_time_sec >= start_time_sec)
);

CREATE TABLE utterance_transitions (
    transition_id BIGSERIAL PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    left_utterance_id BIGINT NOT NULL REFERENCES utterances(utterance_id) ON DELETE CASCADE,
    right_utterance_id BIGINT NOT NULL REFERENCES utterances(utterance_id) ON DELETE CASCADE,
    transition_index INTEGER NOT NULL,
    gold_boundary_label BOOLEAN,
    pred_boundary_prob DOUBLE PRECISION CHECK (
        pred_boundary_prob IS NULL OR (pred_boundary_prob >= 0.0 AND pred_boundary_prob <= 1.0)
    ),
    pred_boundary_label BOOLEAN,
    UNIQUE (meeting_id, transition_index),
    CHECK (left_utterance_id <> right_utterance_id)
);

CREATE TABLE topic_segments (
    topic_segment_id BIGSERIAL PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    segment_type TEXT NOT NULL,
    segment_index INTEGER NOT NULL,
    start_utterance_id BIGINT NOT NULL REFERENCES utterances(utterance_id) ON DELETE RESTRICT,
    end_utterance_id BIGINT NOT NULL REFERENCES utterances(utterance_id) ON DELETE RESTRICT,
    start_time_sec DOUBLE PRECISION NOT NULL,
    end_time_sec DOUBLE PRECISION NOT NULL,
    topic_label TEXT,
    UNIQUE (meeting_id, segment_type, segment_index),
    CHECK (end_time_sec >= start_time_sec),
    CONSTRAINT topic_segments_segment_type_check CHECK (
        segment_type IN ('gold', 'predicted')
    )
);

CREATE TABLE summaries (
    summary_id BIGSERIAL PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    summary_type TEXT NOT NULL CHECK (
        summary_type IN ('ami_gold', 'llm_generated', 'user_edited')
    ),
    summary_object_key TEXT NOT NULL,
    created_by_user_id BIGINT REFERENCES users(user_id) ON DELETE SET NULL,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (meeting_id, summary_type, version)
);

CREATE INDEX idx_meeting_participants_meeting_id
    ON meeting_participants (meeting_id);

CREATE INDEX idx_meeting_participants_user_id
    ON meeting_participants (user_id);

CREATE INDEX idx_meeting_speakers_meeting_id
    ON meeting_speakers (meeting_id);

CREATE INDEX idx_meeting_artifacts_meeting_id
    ON meeting_artifacts (meeting_id);

CREATE INDEX idx_utterances_meeting_id
    ON utterances (meeting_id);

CREATE INDEX idx_utterances_meeting_speaker_id
    ON utterances (meeting_speaker_id);

CREATE INDEX idx_utterance_transitions_meeting_id
    ON utterance_transitions (meeting_id);

CREATE INDEX idx_topic_segments_meeting_id
    ON topic_segments (meeting_id);

CREATE INDEX idx_summaries_meeting_id
    ON summaries (meeting_id);

CREATE INDEX idx_meetings_dataset_version
    ON meetings(dataset_version);

COMMIT;