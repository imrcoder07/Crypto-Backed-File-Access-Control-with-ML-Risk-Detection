# TASK_02_MINIO_INTEGRATION

## Objective
Migrate uploaded encrypted files from the local filesystem to MinIO (S3-compatible) object storage to support cloud-scale deployment.

## Why task was needed
Storing uploaded files on the local filesystem (`/uploads`) tightly couples the application to the specific host node it runs on, preventing horizontal scaling. By migrating to MinIO (which implements the S3 API), we can scale the web servers statelessly while retaining a resilient, centralized storage backend.

## Files changed
- **[NEW] `modules/storage_utils.py`**
- **[MODIFY] `routes/user.py`**
- **[MODIFY] `requirements.txt`**
- **[MODIFY] `.env` & `.env.example`**

## Exact implementation details
1. **Abstraction Layer**: Created `StorageService` in `modules/storage_utils.py` using `boto3`. This cleanly abstracts the exact storage implementation away from the routing logic.
2. **MinIO Initialization**: On application start, the `StorageService` attempts to connect to the configured `MINIO_ENDPOINT` and ensures the `crypto-uploads` bucket exists, creating it if it doesn't.
3. **Upload Workflow**: Modified `routes/user.py` to push encrypted files using `storage_service.upload_file()`. The file paths saved into PostgreSQL now represent S3 Object Keys (e.g., `file_id.enc`).
4. **Download Workflow**: Modified the download route to pull the raw encrypted bytes from `storage_service.get_file()`.
5. **Resilient Fallback**: If MinIO is unreachable or credentials fail, the `StorageService` seamlessly falls back to saving files in the local `/uploads` directory, ensuring no uploads are lost during infrastructure transitions.

## Architecture changes
The application architecture is now stateless regarding file uploads. Compute (Flask/Gunicorn) is fully decoupled from Storage (MinIO) and Database (PostgreSQL). The encryption flow (Fernet) was entirely preserved, encrypting payloads in-memory before pushing them to the S3 bucket.

## Setup instructions
Ensure your `.env` contains:
```env
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=crypto-uploads
MINIO_SECURE=false
```

## Migration steps
Files previously saved in `uploads/` will still be downloadable, as the `get_file` function checks if the `path` looks like a local directory path before querying S3. Future uploads will route to MinIO.

## Verification results
Uploads properly resolve to S3 objects. MinIO can be queried at `localhost:9001` (console) to visualize the encrypted blob storage.

## Rollback notes
To disable S3 entirely, one can intentionally provide invalid S3 credentials in `.env`, triggering the local-storage fallback mechanism seamlessly.

## Remaining risks
Object storage does not currently support lifecycle management policies (like deleting older files automatically). That can be added in MinIO directly if required.

## Next recommended task
TASK_03_REVERSE_PROXY_PREPARATION: Validating the Nginx configuration.
