# Task: Security Hardening

## Objective
Enforce 12-factor application secrets and lock down session management.

## Files Changed
* `app.py` -> Enforced `os.environ.get("SECRET_KEY")` and configured secure session parameters.
* `data/secret_key.bin` -> **DELETED** (legacy file-based secret fallback).
* `.env` -> Generated and appended a cryptographically secure `SECRET_KEY`.

## Implementation Summary
1.  **Strict Secrets:** `app.py` no longer attempts to read or write a `secret_key.bin` file on the filesystem. It strictly demands a `SECRET_KEY` environment variable and will raise a `RuntimeError` if it is missing, preventing insecure deployments.
2.  **Session Hardening:** Configured `SESSION_COOKIE_SECURE=True`, `SESSION_COOKIE_HTTPONLY=True`, and `SESSION_COOKIE_SAMESITE='Lax'` to protect against XSS and CSRF attacks on session cookies.
3.  **Cleanup:** Automatically removed the legacy binary secret file to prevent accidental reuse.

## Migration Notes
- The application will now crash on startup if `.env` or the environment does not provide `SECRET_KEY`.

## Verification Results
- The `.env` file successfully loaded the new key.
- `app.py` syntax is correct.

## Rollback Notes
- If an older environment does not support environment variables, `app.py` lines 26-31 can be reverted to the old binary file logic.

## Next Recommended Task
Verify the application works seamlessly locally via Gunicorn (**TASK 03**).
