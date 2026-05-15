# Incident Rollback Procedure

If the live deployment fails or encounters unforeseen infrastructure issues during the smoke tests, follow this procedure to safely revert to the previous operational state without data corruption.

## Rollback Triggers
You should initiate a rollback if you observe any of the following during post-deployment verification:
1. Application fails to boot (Render marks deployment as Failed).
2. Unrecoverable 500 errors occurring for all users upon load.
3. ProxyFix causes infinite redirect loops (due to load balancer misconfigurations).

## Execution Steps

### Option A: Zero-Downtime Code Reversion (Git)
Because the database schemas were already addressed in Phase 1, the codebase modifications in this step only impact **application-level logic**. 

Execute a clean git revert:
```bash
git revert HEAD
git push origin main
```
Render will automatically detect the push and redeploy the older stable codebase.

### Option B: Render Rollback (Dashboard)
Render allows you to roll back to a previously successful deploy instantly.
1. Navigate to the Render Dashboard -> `Deploys`.
2. Locate the previous successful deploy in the list.
3. Click `Rollback to this deploy`.

### Option C: Re-enabling Local Storage Fallback (Emergency Bypass)
If S3 integration is failing and you absolutely must use the ephemeral local storage (knowing files will be lost on the next deploy), you can temporarily override the safety hard-fail without reverting the code.

In the Render Dashboard:
1. Navigate to the `Environment` tab.
2. Change `FLASK_ENV` from `production` to `development`.
3. Save changes (this triggers an automatic restart).

*Warning: This will re-enable the local filesystem fallback and the auto-spawning background worker threads. Do this only as an emergency bypass.*
