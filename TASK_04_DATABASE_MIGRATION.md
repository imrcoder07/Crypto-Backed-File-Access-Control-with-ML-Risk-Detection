# TASK 04: Database Migration System

## Objective
Replace the fragile raw schema initialization flow with a proper, scalable migration tooling (Alembic) to enable safe schema tracking and a reliable upgrade workflow.

## Files Changed
- `requirements.txt`: Added `Flask-Migrate` (which pulls in Alembic and SQLAlchemy).
- `migrations/env.py`: Configured to dynamically load `DATABASE_URL` from `.env`.
- `migrations/versions/450b3a06e49e_initial_schema.py`: Created the initial baseline migration containing the raw SQL for all existing tables.
- `modules/db.py`: Removed the hardcoded `_SCHEMA_SQL` and `init_schema()`, replacing them with an automated call to `alembic upgrade head` in `init_db()`.
- `alembic.ini`: Auto-generated configuration for Alembic.

## Implementation Summary
1.  **Alembic Setup:** Instead of a full ORM refactoring using Flask-SQLAlchemy (which would add unnecessary complexity and risk to the current raw `psycopg2` pipeline), we set up a raw `Alembic` environment. This perfectly blends the need for structured migrations with the simplicity of the existing data layer.
2.  **Environment Integration:** We modified `env.py` to seamlessly read `DATABASE_URL`, transforming the connection string from `postgres://` to `postgresql://` as required by SQLAlchemy.
3.  **Baseline Migration:** The exact schema creation logic from `db.py` was moved into the initial migration's `upgrade()` method, and corresponding `DROP TABLE` statements were placed in `downgrade()`.
4.  **Startup Workflow:** Modified the application entrypoint `init_db()` to execute `alembic upgrade head` via a subprocess. This guarantees that whether the app is run locally or in Docker, the schema is always verified and updated automatically on boot.

## Migration Notes
- The initial migration perfectly mirrors the old `_SCHEMA_SQL`.
- Future schema changes (e.g., adding a column) should be generated using `alembic revision -m "description"` and then filling out the raw SQL in the generated `upgrade()` block.

## Verification Results
- Ran Alembic programmatically on startup, outputting `✅ Database migrations applied successfully.`
- Executed core pytest suite (`test_auth.py`, `test_db.py`); all 4 core tests pass flawlessly with the new initialization flow.

## Rollback Notes
- To reverse the schema setup completely: `alembic downgrade base`.
- If issues occur with the automated Alembic call, developers can manually comment out `run_migrations()` in `db.py` and run Alembic directly via CLI.

## Next Recommended Task
Move on to **TASK 05 — Async Queue & Redis** to resolve the ML latency bottleneck by implementing Celery for async processing.
