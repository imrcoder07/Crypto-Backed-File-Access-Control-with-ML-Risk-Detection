# Task: PostgreSQL Migration

## Objective
Eradicate the fragmented state of the application by fully migrating `users.json` and the in-memory `blockchain_ledger` (now renamed to `audit_ledger`) into the robust PostgreSQL backend.

## Files Changed
* `data/users.json` -> Backed up to `data/users_backup.json`
* `modules/db.py` -> Added schemas and helpers for `users` and `audit_ledger`.
* `migrate_users.py` (New) -> Script created to migrate existing users from JSON to PostgreSQL.
* `modules/auth_utils.py` -> Refactored to drop JSON reading/writing entirely in favor of PostgreSQL operations. Dropped silent admin creation logic.
* `modules/blockchain_utils.py` -> Renamed to `modules/audit_utils.py`. The `Blockchain` class was renamed to `TamperEvidentLedger` and updated to read/write its state from PostgreSQL.
* `modules/extensions.py` -> Renamed `blockchain` to `audit_ledger`.
* `routes/auth.py`, `routes/user.py`, `routes/admin.py`, `routes/main.py` -> Updated to use `db.get_user`, `db.get_all_users`, and `audit_ledger`.

## Implementation Summary
1.  **Schema Updates:** The PostgreSQL schema was expanded with a `users` table and an `audit_ledger` table.
2.  **Data Migration:** A standalone script (`migrate_users.py`) was executed to copy the local `users.json` state securely into the database.
3.  **Route Refactoring:** All instances of the global `USER_DB` dictionary in memory were eliminated. Every route now talks to PostgreSQL for user lookup, login, registration, and system metrics.
4.  **Audit Ledger Modernization:** The in-memory audit log was converted to `TamperEvidentLedger`. Events are still processed asynchronously by a daemon thread, but blocks are written to PostgreSQL upon creation. It gracefully builds a Genesis Block if the table is empty.

## Migration Notes
- We used `dotenv` in the migration script to pick up `DATABASE_URL`.
- The `blockchain_ledger` was dropped and recreated as `audit_ledger`. The previous blockchain ledger contained only volatile dummy data from the current runtime session.
- Two users (`admin` and `admin_01`) were successfully migrated from the JSON file.

## Verification Results
- All migration queries passed without error.
- App routes were checked to verify they now import the correct classes and use the correct function calls.

## Rollback Notes
- If the PostgreSQL migration needs to be rolled back, the `data/users_backup.json` is preserved on disk.
- Routes can be rolled back using standard git reversion (if tracked).

## Next Recommended Task
Move on to **TASK 02: SECURITY HARDENING** to enforce `SECRET_KEY` generation directly from environment variables (avoiding the `.bin` fallback) and locking down session cookies.
