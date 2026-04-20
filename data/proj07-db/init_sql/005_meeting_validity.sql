BEGIN;

ALTER TABLE meetings
    ADD COLUMN IF NOT EXISTS is_valid BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_meetings_source_type_is_valid
    ON meetings(source_type, is_valid);

UPDATE meetings
SET is_valid = TRUE
WHERE source_type = 'ami';

COMMIT;
