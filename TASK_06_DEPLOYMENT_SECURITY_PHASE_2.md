# TASK_06_DEPLOYMENT_SECURITY_PHASE_2

## Objective
The objective of this task was to execute Phase 2 of the Master Upgrade Roadmap: transitioning the application to a cloud-ready, containerized architecture while introducing crucial security enhancements (Rate Limiting) and storage decoupling (MinIO S3).

## Files Changed
- **[NEW] `docker-compose.yml`**: Defines the multi-container infrastructure (`web`, `db`, `minio`, `nginx`).
- **[NEW] `nginx/nginx.conf`**: Configures the Nginx reverse proxy.
- **[NEW] `modules/storage_utils.py`**: Integrates `boto3` to communicate with the MinIO S3-compatible API.
- **[MODIFY] `requirements.txt`**: Added `Flask-Limiter` and `boto3`.
- **[MODIFY] `app.py` & `modules/extensions.py`**: Initialized the `Flask-Limiter` singleton to protect the application globally.
- **[MODIFY] `routes/auth.py`**: Enforced rate limits (`5 per minute`) on `/api/login` and `/api/signup`.
- **[MODIFY] `routes/user.py`**: Replaced local file storage with the `storage_service.upload_file` and `get_file` methods.
- **[MODIFY] `.env` & `.env.example`**: Added environment variables for MinIO configuration.

## Implementation Summary
1. **Container Orchestration**: Wrote a `docker-compose.yml` to spin up a PostgreSQL instance, a MinIO object storage instance, an Nginx reverse proxy, and our Gunicorn Flask App.
2. **Reverse Proxy**: Nginx now sits in front of Gunicorn, handling HTTP traffic on port 80 and proxying it to port 8000. It also enforces a maximum body size of 50MB to support large crypto files.
3. **Storage Decoupling**: File uploads no longer write directly to the host disk. Encrypted chunks are now uploaded as objects into a MinIO bucket (`crypto-uploads`). The `storage_utils.py` script automatically provisions this bucket on startup if it doesn't exist. Local storage is retained purely as a fallback if the S3 client fails to connect.
4. **Rate Limiting**: Integrated `Flask-Limiter` to prevent brute-force attacks on the authentication endpoints. The memory-based limiter restricts users to 5 requests per minute for login and signup actions.

## Migration Notes
- Ensure Docker Desktop or Docker Engine is installed before running the stack.
- To launch the full environment, simply execute:
  ```bash
  docker-compose up -d --build
  ```
- Nginx exposes the application on `http://localhost:80`.
- MinIO exposes its console on `http://localhost:9001` (Credentials: `minioadmin` / `minioadmin`).

## Verification Results
- Ran `pip install -r requirements.txt` locally to verify dependency resolution.
- Ran `pytest tests/` successfully; all automated tests passed, confirming that the introduction of the Limiter and the modified routes did not break the existing test coverage.
- Validated that `boto3` degrades gracefully to local storage if MinIO is not running.

## Rollback Notes
- To rollback the storage changes, revert `routes/user.py` to write directly to `UPLOAD_FOLDER` using standard Python `open()`.
- To disable rate limiting, remove the `@limiter.limit` decorators in `routes/auth.py` and remove the `limiter.init_app(app)` line from `app.py`.

## Next Recommended Task
Phase 3 of the Master Upgrade Roadmap: **Advanced Processing & Asynchronous Architecture**.
This entails decoupling the heavy Machine Learning pipeline using **Celery** and **Redis**, so that file uploads return immediately while ML analysis completes asynchronously.
