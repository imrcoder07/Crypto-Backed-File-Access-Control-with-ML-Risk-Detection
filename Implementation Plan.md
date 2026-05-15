# Implementation Plan: Phase 1 (Elite Foundation)

This plan covers the immediate execution of Phase 1 from the Master Upgrade Roadmap. The focus is strictly on data integrity, security hardening, and stable runtime execution.

## Proposed Changes

---
### 1. PostgreSQL Migration (Eradicating Fragmented State)

Currently, the application runs a split-brain state: files/requests are in Postgres, but users are in `users.json`, and the blockchain audit log is an in-memory list.

#### [MODIFY] `modules/db.py`
- Add schema definitions for `users` and `blockchain_ledger` tables.
- Add helper methods: `get_user`, `create_user`, `get_all_users`.
- Add helper methods: `append_block`, `get_recent_blocks`, `get_chain_length`.

#### [MODIFY] `modules/auth_utils.py`
- Completely remove `USER_DB`, `load_user_db`, and `save_user_db` referencing the local JSON file.
- Update `check_password` and `hash_password` usage to interact directly with `db.py`'s `get_user` method.
- Seed a default Admin user directly into Postgres if the `users` table is empty.

#### [MODIFY] `modules/blockchain_utils.py`
- Modify the `Blockchain` class to load the chain from Postgres on initialization.
- Modify the `_append_block` worker to persist new blocks to Postgres instead of just appending to a Python list.

#### [MODIFY] Routes (`routes/auth.py`, `routes/user.py`, `routes/admin.py`, `routes/main.py`)
- Refactor any imports and usages of `USER_DB` to perform standard database queries using `modules/db.py`.

---
### 2. Security Hardening (12-Factor App Secrets & Sessions)

The app currently generates a `secret.key` file fallback. This is an anti-pattern for secure production deployments.

#### [MODIFY] `app.py`
- **Secret Management:** Remove the local file fallback for `SECRET_KEY`. Enforce `os.environ.get("SECRET_KEY")` and explicitly raise a `RuntimeError` if missing to enforce 12-factor compliance.
- **Session Security:** Configure Flask sessions to use `SESSION_COOKIE_SECURE = True`, `SESSION_COOKIE_HTTPONLY = True`, and `SESSION_COOKIE_SAMESITE = 'Lax'`.

---
### 3. Stable Server Runtime (Gunicorn)

The default Flask development server is not built to handle concurrent requests securely or efficiently.

#### [NEW] `wsgi.py`
- Create a standard WSGI entry point for Gunicorn.

#### [MODIFY] `requirements.txt`
- Add `gunicorn` to the dependencies list.

---
### 4. Automated Tests (Foundation)

To prevent regressions, we will establish a baseline testing suite.

#### [NEW] `tests/test_auth.py`
- Write tests using `pytest` to verify signup and login flows against the new PostgreSQL user schema.

---

## Verification Plan

### Automated Tests
- Run `pytest` to ensure authentication flows pass with the new database structure.
- Run a compilation check (`python -m py_compile`) on all modified routes and modules.

### Manual Verification
- Start the server using Gunicorn (`gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app`).
- Verify that registering a new user creates a record in the Postgres database, not `users.json`.
- Verify that performing an action (like logging in) appends a block to the `blockchain_ledger` Postgres table.
- Verify that restarting the server **does not** wipe out the user accounts or the audit ledger.

> [!WARNING]
> This refactoring will break the existing `users.json` state. Any users currently saved in the local JSON file will be lost, as the system will rely entirely on the PostgreSQL database going forward. The default `admin` account will be auto-regenerated if missing. Is this acceptable?
