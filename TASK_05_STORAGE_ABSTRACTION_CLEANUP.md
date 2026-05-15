# TASK 05: Storage Abstraction Cleanup

## Objective
Finalize the object storage migration by decoupling all remaining hardcoded local filesystem operations and replacing them with our unified `StorageService`.

## Files Changed
- `modules/storage_utils.py`: Added the `delete_file()` method to abstract the deletion logic. It attempts to remove the object from MinIO S3, gracefully falling back to local `os.remove()` if the file happens to be stored on the local disk instead.
- `modules/db.py`: Replaced the legacy `os.remove(path)` within the `cleanup_old_requests` function with the new `storage_service.delete_file(path)` interface.

## Implementation Summary
1.  **Extended StorageService**: The class was enhanced with a `delete_file` method. This allows the application to cleanly discard files across both storage systems (AWS/MinIO and local file system) depending on the environment context, without exposing the implementation details to the caller.
2.  **Database Sync**: The backend scheduled task `cleanup_old_requests` (which deletes requests older than 7 days) used to only purge database rows and delete local files. We updated it to trigger `delete_file()` using our service layer, ensuring S3 objects are appropriately expunged, preventing cloud storage leaks over time.

## Verification Steps
1.  Verify tests pass indicating the `cleanup_old_requests` functions do not throw errors: run `venv\Scripts\pytest tests/test_db.py`.
2.  Inspect `modules/storage_utils.py` and `modules/db.py` to confirm `os.remove()` operations acting on user files have been replaced by the service instance.

## Rollback Notes
- To reverse, update `db.py` to import `os` and revert `storage_service.delete_file(path)` back to `if os.path.exists(path): os.remove(path)`.
- Delete `delete_file()` method from `storage_utils.py`.

## Next Recommended Task
Since Phase 2 (Docker, Nginx, MinIO, Validation) is thoroughly hardened and mature, it's highly recommended to proceed with **Phase 3: Celery + Redis Task Queue** to offload the heavy machine learning prediction tasks and improve frontend responsiveness.
