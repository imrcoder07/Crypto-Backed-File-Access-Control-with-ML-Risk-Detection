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

## Live Deployment Checklist (Render)

1. **Environment Variables**
   Ensure the following variables are active in the Render Dashboard before triggering the manual deploy:
   - `FLASK_ENV` = `production`
   - `MINIO_ENDPOINT`
   - `MINIO_ACCESS_KEY`
   - `MINIO_SECRET_KEY`
   - `MINIO_BUCKET_NAME`

2. **Trigger Manual Deploy**
   Push the code and monitor the Build Logs on Render.
   *Expected Result:* Dependencies install successfully and the Start Command successfully outputs Gunicorn binding to `0.0.0.0:<dynamic_port>`.

3. **Storage Integration Test**
   Navigate to the live application:
   - Login / Register.
   - Upload a test file.
   *Expected Result:* The file uploads successfully.
   - Verify the object appears inside your S3 / MinIO storage bucket.

4. **Background Task Verification**
   Check the Render Application Logs immediately after startup.
   *Expected Result:* You should see the log output: `⏳ Background cleanup thread disabled to ensure operational stability.` No automated cleanup runs should trigger on the web instances.
