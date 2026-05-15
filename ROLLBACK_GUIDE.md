# Rollback Guide: Phase 2 Render Hardening

If the live deployment fails or encounters unforeseen infrastructure issues, follow this procedure to safely revert to the previous operational state without data corruption.

## 1. Zero-Downtime Code Reversion
Because the database schemas were already addressed in Phase 1, the codebase modifications in this step only impact **application-level logic** (storage behavior, thread management, Docker execution). 

To roll back, you can execute a clean git revert:
```bash
git revert HEAD
git push origin main
```
Render will automatically detect the push and redeploy the older stable codebase.

## 2. Re-enabling Local Storage Fallback
If S3 integration is failing and you absolutely must use the ephemeral local storage (knowing files will be lost on the next deploy), you can temporarily override the safety hard-fail without reverting the code.

In the Render Dashboard:
1. Navigate to the `Environment` tab.
2. Change `FLASK_ENV` from `production` to `development`.
3. Save changes (this triggers an automatic restart).

*Warning: This will re-enable the local filesystem fallback and the auto-spawning background worker threads. Do this only as an emergency bypass.*

## 3. Configuration Reversions
If you only need to rollback the Render port configuration:
1. Revert `render.yaml` `startCommand` back to `gunicorn app:app`.
2. Revert `Dockerfile` `CMD` back to `["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]`.
