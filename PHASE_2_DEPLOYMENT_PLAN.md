# Phase 2 Deployment Plan (Render Focus)

This project is currently deployed on Render and connected to a custom domain. This requires a hard pivot to ensure true production readiness, as Render uses an ephemeral filesystem for web services. We must ensure the app doesn't accidentally save files locally, properly binds to Render's dynamic ports, and handles scheduled background tasks safely under Gunicorn.

## 1. Storage Abstraction Hardening (MinIO/S3)
Render uses an ephemeral filesystem. If the application restarts or sleeps, all local files are wiped.
- **Current Issue:** `modules/storage_utils.py` currently falls back to writing files to the local `uploads/` directory if S3/MinIO fails.
- **Fix:** We must enforce a strict "hard fail" if `FLASK_ENV=production`. If the S3 bucket is unreachable in production, the upload must fail rather than silently falling back to local disk and eventually losing the user's data.

## 2. Docker & Configuration Consistency
- **Current Issue:** The `Dockerfile` and `render.yaml` configurations hardcode ports or rely on defaults that don't always align with Render's `$PORT` environment variable injection.
- **Fix:** Update `render.yaml` to ensure `startCommand` binds to `$PORT` (`gunicorn app:app --bind 0.0.0.0:$PORT`). Update the `Dockerfile` to mirror this consistency (`CMD gunicorn app:app --bind 0.0.0.0:${PORT:-8000}`).

## 3. Operational Stability (Background Tasks)
- **Current Issue:** `app.py` automatically starts a background thread `cleanup_worker()` as soon as it is imported. Since Gunicorn spawns worker processes by importing `app.py`, this means multiple cleanup threads could spawn simultaneously in production, causing database locking issues or race conditions.
- **Fix:** The background task scheduler must be guarded so it doesn't spawn recklessly in a multi-worker production environment.

## Execution
If approved, I will implement these changes directly into the project files and generate an artifact upon completion.
