# Future Background Worker Migration Plan

**Purpose:**
To outline the necessary steps for reactivating and scaling the background tasks (currently disabled in production to prevent Gunicorn concurrency issues) safely on Render's infrastructure.

**Current State:**
In `app.py`, the `cleanup_worker` scheduler is conditionally disabled if `FLASK_ENV == "production"`. This was necessary because deploying the web server behind multiple Gunicorn workers meant multiple cleanup threads would race each other to delete database records, creating severe locking issues.

**Future Execution Paths:**

### Option 1: Render Background Worker (Simplest)
Render offers a dedicated "Background Worker" service type that runs continuously without receiving external web traffic.
1. Create a new "Background Worker" service in the Render Dashboard connected to this exact same repository.
2. Set the Environment Variables for this worker to be identical to the web service, but add: `RUN_BACKGROUND_TASKS=true`.
3. Set the Start Command for this specific worker to just run the scheduler instead of Gunicorn. (e.g., create a simple `worker.py` that just invokes `start_cleanup_scheduler()` and sleeps).
*Benefit: Requires zero new dependencies.*

### Option 2: APScheduler Integration
If a dedicated worker instance is too expensive, you can integrate `APScheduler` (Advanced Python Scheduler) combined with a database lock (or Redis) to ensure only one web worker executes the cleanup task.
1. `pip install APScheduler`
2. Implement a lock mechanism (e.g., using a Postgres advisory lock) so that when the scheduled job triggers, only the worker that successfully acquires the lock executes the `db.cleanup_old_requests()` logic.

### Option 3: Celery + Redis (Most Scalable)
When the application scales to handle hundreds of concurrent ML modeling requests, offloading both the ML models and the background cleanup tasks to a Celery queue backed by Redis will be mandatory.
1. Spin up a managed Redis instance.
2. Refactor `cleanup_worker` into a `@celery.task`.
3. Use Celery Beat to trigger the cleanup task on a schedule.
4. Deploy a Render Background Worker explicitly running `celery -A app.celery worker` and `celery -A app.celery beat`.
