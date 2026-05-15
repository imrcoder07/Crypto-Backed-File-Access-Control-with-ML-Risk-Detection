# Task: Stable Server Runtime (Gunicorn)

## Objective
Replace the default Flask development server with Gunicorn to ensure the application can handle concurrency, thread-safety, and production-level workloads.

## Files Changed
* `wsgi.py` (New) -> The primary entry point for Gunicorn.
* `requirements.txt` -> Verified that `gunicorn==21.2.0` is already present.

## Implementation Summary
1.  **WSGI Entry Point:** Created `wsgi.py` which simply imports the `app` from `app.py` and initializes the required background scheduler thread (`start_cleanup_scheduler`).
2.  **Concurrency Preparation:** The codebase is now ready to be invoked via `gunicorn --bind 0.0.0.0:5000 wsgi:app --workers 4 --threads 2`. This structure cleanly separates the app definition from the server initialization.

## Migration Notes
- When deploying, do not run `python app.py`. Always run via `gunicorn` or a similar production WSGI server.

## Verification Results
- `wsgi.py` correctly imports the Flask application instance without triggering the `__main__` block from `app.py`.

## Rollback Notes
- No rollback necessary; `app.py` still retains its `if __name__ == '__main__':` block, so `python app.py` still functions locally for quick testing.

## Next Recommended Task
Move on to **TASK 04: AUTOMATED TESTS & SYNC VERIFICATION** to build the initial `pytest` suite ensuring all components (especially auth and database flows) operate cohesively.
