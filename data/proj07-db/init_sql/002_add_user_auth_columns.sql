BEGIN;

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS password_salt TEXT,
    ADD COLUMN IF NOT EXISTS password_hash TEXT,
    ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'users_password_columns_pair_check'
    ) THEN
        ALTER TABLE users
            ADD CONSTRAINT users_password_columns_pair_check CHECK (
                (password_salt IS NULL AND password_hash IS NULL)
                OR (password_salt IS NOT NULL AND password_hash IS NOT NULL)
            );
    END IF;
END $$;

COMMIT;
