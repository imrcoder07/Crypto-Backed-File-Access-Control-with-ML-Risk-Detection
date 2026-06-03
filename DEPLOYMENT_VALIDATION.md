# Deployment Validation Guide

Follow these steps precisely before and immediately after pushing the new codebase to the live Render environment.

## Pre-Deployment Checklist (Local Verification)

1. **Verify Local Docker Consistency**
   Run the stack locally with a custom port to simulate Render's injection:
   ```bash
   export PORT=8080
   docker build -t crypto-app .
   docker run -p 8080:8080 -e PORT=8080 -e FLASK_ENV=production crypto-app
   ```
   *Expected Result:* Gunicorn starts and binds to `0.0.0.0:8080`.

2. **Verify Storage Hard-Fail Mechanism**
   While running the production local container (above), attempt an upload without passing MinIO credentials.
   *Expected Result:* The upload fails with a `RuntimeError` and a 500 status code, and no file is silently written to your local disk.

3. **Verify Hybrid Celery Worker Startup**
   Run the stack locally simulating the production environment with the async ML flag enabled:
   ```bash
   export PORT=8080
   export USE_ASYNC_ML=true
   export REDIS_URL=redis://localhost:6379
   docker build -t crypto-app .
   docker run -p 8080:8080 -e PORT=8080 -e USE_ASYNC_ML=true -e REDIS_URL=redis://localhost:6379 -e FLASK_ENV=production crypto-app
   ```
   *Expected Result:* 
   - The startup logs output: `==> Starting Celery background worker (solo pool)...`
   - Celery boots successfully without dependency errors.
   - The startup logs output: `==> Starting Gunicorn...`
   - Gunicorn starts and binds to port 8080.

## Live Deployment Checklist (Render)

1. **Environment Variables**
   Ensure the following variables are active in the Render Dashboard before triggering the manual deploy:
   - `FLASK_ENV` = `production`
   - `MINIO_ENDPOINT`
   - `MINIO_ACCESS_KEY`
   - `MINIO_SECRET_KEY`
   - `MINIO_BUCKET_NAME`
   - `USE_ASYNC_ML` = `true` (toggles the asynchronous ML scanning engine)
   - `REDIS_URL` = `<your-upstash-redis-url>` (namespace isolated Redis connection string)

2. **Trigger Manual Deploy**
   Push the code and monitor the Build Logs on Render.
   *Expected Result:* Dependencies install successfully and the Start Command successfully outputs Gunicorn binding to `0.0.0.0:<dynamic_port>`.

3. **Storage Integration Test**
   Navigate to the live application:
   - Login / Register.
   - Upload a test file.
   - Check the console network requests.
   *Expected Result:* The upload returns HTTP 202 Accepted. The UI displays the loading scan status, polls `/api/request_status/<request_id>` every 2 seconds, and transitions to successful completion status once Celery finishes.

4. **Background Task & Celery Verification**
   Check the Render Application Logs immediately after startup and task execution.
   *Expected Result:* 
   - Logs show: `==> Starting Celery background worker (solo pool)...`
   - Celery reports: `celery@<hostname> ready`
   - Task logs show: `Background Job: Starting ML analysis for request req_...` and `Background Job: ML Analysis complete for request req_...`

