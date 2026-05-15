# Safe Live Deployment Sequence

Follow this exact sequence to deploy the Phase 2 upgraded infrastructure to the live Render environment.

## 1. Render Dashboard Preparations
**Ensure Environment Variables are Correct:**
1. Navigate to the Render Dashboard -> `Environment`.
2. Ensure the following variables exist and are correct:
   - `FLASK_ENV` = `production`
   - `MINIO_ENDPOINT` = `<your_production_s3_url>`
   - `MINIO_ACCESS_KEY` = `<your_key>`
   - `MINIO_SECRET_KEY` = `<your_secret>`
   - `MINIO_BUCKET_NAME` = `<your_bucket>`
   - `SECRET_KEY` = `<a_very_secure_random_string>`
3. Ensure no `PORT` variable is hardcoded (Render will inject this natively).

## 2. Git Push Execution
Execute a standard push to your production branch:
```bash
git add .
git commit -m "chore: Phase 2 infrastructure hardening (ProxyFix, Logging, Strict Storage)"
git push origin main
```

## 3. Render Deployment Observation
1. Navigate to the Render Dashboard -> `Events` or `Logs`.
2. Monitor the build process. Ensure `pip install -r requirements.txt` succeeds.
3. Monitor the startup command. You must observe the following sequence:
   - `[INFO] 🚀 Application starting up...`
   - `[INFO] ⚙️ FLASK_ENV: production`
   - `[INFO] 🔌 Port Binding: <dynamic_port>`
   - `[INFO] 🛡️ ProxyFix Middleware: Active`
   - `[INFO] ⏳ Background cleanup thread disabled`
   - Gunicorn binding to `0.0.0.0:<dynamic_port>`

## 4. Final Handoff
Once the health checks pass and Render routes traffic, immediately proceed to the steps outlined in `POST_DEPLOYMENT_SMOKE_TESTS.md`.
