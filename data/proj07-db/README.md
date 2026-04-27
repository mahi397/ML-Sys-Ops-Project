# proj07-db

This folder is now source-only for the database layer.

It contains the schema and migration assets that the runnable stack under `../proj07-runtime/` mounts into Postgres:

- `init_sql/`
  - schema bootstrap and integration-era SQL additions

## When to use this folder

- Use `proj07-db/` when you need to inspect or edit the canonical schema/bootstrap SQL.
- Use `../proj07-runtime/` when you want to launch Postgres, Adminer, and the ingest/workflow services.

## Schema files

- `init_sql/001_init_postgres_schema.sql`
- `init_sql/002_feedback_loop_schema.sql`
- `init_sql/002_add_user_auth_columns.sql`
- `init_sql/003_workflow_tasks.sql`
- `init_sql/004_retrain_audit_logs.sql`
- `init_sql/005_meeting_validity.sql`

The SQL files under `init_sql/` are mounted by the root `docker-compose.yml`. On a fresh Postgres data volume they run automatically during container initialization, and `data/setup.sh` also re-applies the idempotent post-bootstrap migrations so existing volumes pick up schema additions such as `meetings.is_valid`.
