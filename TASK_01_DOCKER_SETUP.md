# TASK 01: Docker + docker-compose setup

## Objective
Establish a reproducible, production-ready containerized environment for the Crypto Access Control application, separating the web server, database, and object storage components.

## Files Changed/Reviewed
- `Dockerfile`: Configured the application image using `python:3.10-slim`.
- `docker-compose.yml`: Defined services for `web`, `db` (Postgres), `minio` (Object Storage), and `nginx`. Removed obsolete `version` attribute for modern Docker Compose compatibility.
- `.dockerignore`: (Implied/verified) Prevents pulling in virtual environments and local artifacts.

## Implementation Summary
1.  **Optimized Dockerfile:** Used a slim Python base image to reduce attack surface and image size. Layer caching was leveraged by copying `requirements.txt` before the application code.
2.  **Least Privilege Security:** Added a non-root `appuser` so the Flask application process does not run as root inside the container, mitigating container escape risks.
3.  **Service Orchestration:** `docker-compose.yml` orchestrates the multi-container architecture. It mounts a persistent volume for the Postgres database (`pgdata`) and MinIO storage (`minio_data`). Healthchecks are defined to ensure the backend web service only starts routing once the `db` and `minio` are healthy.

## Setup/Verification Steps
1.  Ensure Docker and Docker Compose are installed and the Docker Daemon is running.
2.  Execute `docker-compose up -d --build` in the project root.
3.  Verify container status with `docker-compose ps`. All containers should report an `Up` or `Healthy` state.
4.  Test internal networking: the `web` container successfully connects to the `db` using `DATABASE_URL=postgresql://postgres:postgres@db:5432/crypto_access_control`.

## Rollback Notes
- If an older Docker version fails, restore the `version: "3.8"` declaration at the top of `docker-compose.yml`.
- Database states can be purged using `docker-compose down -v` if schema regeneration is necessary.

## Next Recommended Task
Proceed to **Task 2: MinIO Integration Validation** to ensure file uploads are abstracted correctly from the filesystem.
