# proj07-db

This folder is now DB-only.

It contains the Postgres bootstrap assets for the final integrated workflow:

- `init_sql/`
  - schema bootstrap and integration-era SQL additions
- `docker-compose.yml`
  - starts Postgres plus Adminer only
- `.env.example`
  - DB-only environment defaults

## When to use this folder

- Use `proj07-db/` when you only need the database and schema bootstrap.
- Use `../proj07-services/` when you want the full ingest and workflow-service stack.
- Do not run both compose files at the same time; the full-service compose already includes Postgres plus Adminer.

## Start the DB only

```bash
cp .env.example .env
docker compose up -d
```

## Schema files

- `init_sql/001_init_postgres_schema.sql`
- `init_sql/002_feedback_loop_schema.sql`
- `init_sql/002_add_user_auth_columns.sql`
- `init_sql/003_workflow_tasks.sql`
- `init_sql/004_retrain_audit_logs.sql`

The SQL files under `init_sql/` are applied automatically by Postgres only when the database volume is initialized for the first time.
