"""
modules/db.py — PostgreSQL persistence layer for Crypto Access Control App.

Replaces the in-memory PersistentStorage class with a thread-safe connection
pool backed by PostgreSQL (psycopg2, no ORM).

Environment variable required:
    DATABASE_URL=postgresql://user:pass@host:5432/dbname
"""
import os
import json
import datetime
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

# ── Connection pool ──────────────────────────────────────────────────────────

DATABASE_URL: str = os.environ.get("DATABASE_URL", "")

_pool: ThreadedConnectionPool | None = None


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL is not set. Add it to your .env file."
            )
        _pool = ThreadedConnectionPool(minconn=1, maxconn=10, dsn=DATABASE_URL)
    return _pool


@contextmanager
def get_db():
    """Yield a pooled connection; commit on success, rollback on error."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


import subprocess
import sys

def run_migrations():
    """Run Alembic migrations to ensure the schema is up to date."""
    print("⏳ Running database migrations...")
    try:
        subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ Database migrations applied successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Migration failed: {e.stderr}")
        raise

def init_db() -> None:
    """Initialize the database by running migrations."""
    run_migrations()

# ── User helpers ──────────────────────────────────────────────────────────────

def get_user(username: str) -> dict | None:
    sql = "SELECT * FROM users WHERE username = %s;"
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (username,))
            row = cur.fetchone()
    return dict(row) if row else None


def create_user(username: str, password_hash: str, role: str, email: str, department: str) -> None:
    sql = """
        INSERT INTO users (username, password, role, email, department)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (username) DO UPDATE
        SET password = EXCLUDED.password,
            role = EXCLUDED.role,
            email = EXCLUDED.email,
            department = EXCLUDED.department;
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (username, password_hash, role, email, department))


def get_all_users() -> list:
    sql = "SELECT * FROM users ORDER BY created_at ASC;"
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def update_user_password(username: str, new_password_hash: str) -> bool:
    """Update a user's password hash. Returns True if the user existed and was updated."""
    sql = "UPDATE users SET password = %s WHERE username = %s RETURNING username;"
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (new_password_hash, username))
            return cur.fetchone() is not None


def promote_user_role(username: str, new_role: str = "Admin") -> bool:
    """Change a user's role. Returns True if the user existed and was updated."""
    allowed_roles = {"Admin", "User"}
    if new_role not in allowed_roles:
        raise ValueError(f"Invalid role '{new_role}'. Must be one of: {allowed_roles}")
    sql = "UPDATE users SET role = %s WHERE username = %s RETURNING username;"
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (new_role, username))
            return cur.fetchone() is not None


# ── Audit Ledger helpers ────────────────────────────────────────────────────────

def append_block(index: int, timestamp: str, data: str, proof: str, previous_hash: str, block_hash: str) -> None:
    sql = """
        INSERT INTO audit_ledger (index, timestamp, data, proof, previous_hash, block_hash)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (index) DO NOTHING;
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (index, timestamp, data, proof, previous_hash, block_hash))

def get_chain() -> list:
    sql = "SELECT * FROM audit_ledger ORDER BY index ASC;"
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    
    # Format timestamps back to strings for compatibility
    for row in rows:
        if isinstance(row.get('timestamp'), datetime.datetime):
            row['timestamp'] = row['timestamp'].isoformat()
    return [dict(r) for r in rows]

def get_chain_length() -> int:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM audit_ledger;")
            return cur.fetchone()[0]

def get_recent_blocks(count: int = 50) -> list:
    sql = "SELECT * FROM audit_ledger ORDER BY index DESC LIMIT %s;"
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (count,))
            rows = cur.fetchall()
    
    # The above returns newest first. We reverse to return oldest to newest in the recent batch.
    rows.reverse()
    
    for row in rows:
        if isinstance(row.get('timestamp'), datetime.datetime):
            row['timestamp'] = row['timestamp'].isoformat()
    return [dict(r) for r in rows]



# ── File helpers ──────────────────────────────────────────────────────────────

def save_file(file_id: str, filename: str, safe_filename: str, path: str,
              salt: str, owner: str, file_size: int, file_size_mb: float,
              file_hash: str = None, requires_password: bool = True) -> None:
    sql = """
        INSERT INTO files
            (file_id, filename, safe_filename, path, salt, owner,
             file_size, file_size_mb, file_hash, requires_password)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (file_id) DO NOTHING;
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (file_id, filename, safe_filename, path, salt,
                              owner, file_size, file_size_mb, file_hash, requires_password))


def create_file_record(file_id: str, owner: str, filename: str,
                       path: str, file_size: int, file_hash: str = None) -> None:
    """Backward-compatible wrapper used by the upload route."""
    file_size_mb = round(file_size / (1024 * 1024), 4) if file_size else 0.0
    save_file(
        file_id=file_id,
        filename=filename,
        safe_filename=filename,
        path=path,
        salt='',
        owner=owner,
        file_size=file_size,
        file_size_mb=file_size_mb,
        file_hash=file_hash,
        requires_password=True,
    )


def get_file_by_name_and_owner(filename: str, owner: str) -> dict | None:
    sql = "SELECT * FROM files WHERE filename = %s AND owner = %s ORDER BY uploaded_at DESC LIMIT 1;"
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (filename, owner))
            row = cur.fetchone()
    return dict(row) if row else None


def get_file(file_id: str) -> dict | None:
    sql = "SELECT * FROM files WHERE file_id = %s;"
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (file_id,))
            row = cur.fetchone()
    return dict(row) if row else None


def file_count() -> int:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM files;")
            return cur.fetchone()[0]


def password_protected_count() -> int:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM files WHERE requires_password = TRUE;")
            return cur.fetchone()[0]


def get_user_files(username: str) -> list:
    sql = """
        SELECT f.*, r.status, r.request_id 
        FROM files f 
        LEFT JOIN requests r ON f.file_id = r.file_id 
        WHERE f.owner = %s 
        ORDER BY f.uploaded_at DESC;
    """
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (username,))
            rows = cur.fetchall()

    files = []
    for row in rows:
        item = dict(row)
        uploaded_at = item.get('uploaded_at')
        item['upload_time'] = (
            uploaded_at.isoformat()
            if isinstance(uploaded_at, datetime.datetime)
            else uploaded_at
        )
        files.append(item)
    return files



# ── Request helpers ───────────────────────────────────────────────────────────

def save_request(request_id: str, file_id: str, filename: str, username: str,
                 user_role: str, ml_verdict: str, ml_details: dict,
                 file_size: int, file_size_mb: float,
                 requires_password: bool = True, status: str = "pending") -> None:
    sql = """
        INSERT INTO requests
            (request_id, file_id, filename, username, user_role,
             ml_verdict, ml_details, file_size, file_size_mb,
             requires_password, password_provided, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE, %s)
        ON CONFLICT (request_id) DO NOTHING;
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                request_id, file_id, filename, username, user_role,
                ml_verdict, json.dumps(ml_details),
                file_size, file_size_mb, requires_password, status
            ))


def create_request(request_id: str, username: str, file_id: str, filename: str,
                   user_role: str, ml_details: dict, status: str = "pending") -> None:
    """Backward-compatible wrapper used by the upload route."""
    file_info = get_file(file_id) or {}
    file_size = file_info.get('file_size') or 0
    file_size_mb = file_info.get('file_size_mb') or 0.0
    verdict = (
        ml_details.get('verdict')
        or ml_details.get('risk_level')
        or ml_details.get('classification')
        or 'review'
    )
    save_request(
        request_id=request_id,
        file_id=file_id,
        filename=filename,
        username=username,
        user_role=user_role,
        ml_verdict=str(verdict),
        ml_details=ml_details,
        file_size=file_size,
        file_size_mb=file_size_mb,
        requires_password=True,
        status=status
    )


def update_request_ml_results(request_id: str, ml_verdict: str, ml_details: dict, new_status: str = "pending") -> bool:
    """Update an existing request with ML risk analysis results and change its status."""
    sql = """
        UPDATE requests
        SET ml_verdict = %s,
            ml_details = %s,
            status = %s
        WHERE request_id = %s
        RETURNING request_id;
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (ml_verdict, json.dumps(ml_details), new_status, request_id))
            return cur.fetchone() is not None


def update_request_status_only(request_id: str, new_status: str) -> bool:
    """Update only the status column of a request."""
    sql = "UPDATE requests SET status = %s WHERE request_id = %s RETURNING request_id;"
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (new_status, request_id))
            return cur.fetchone() is not None



def get_request(request_id: str) -> dict | None:
    sql = "SELECT * FROM requests WHERE request_id = %s;"
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (request_id,))
            row = cur.fetchone()
    return _deserialize_request(dict(row)) if row else None


def get_pending_requests() -> list:
    sql = "SELECT * FROM requests WHERE status = 'pending' ORDER BY upload_time DESC;"
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    return [_deserialize_request(dict(r)) for r in rows]


def get_approved_requests() -> list:
    sql = ("SELECT * FROM requests WHERE status = 'approved' "
           "ORDER BY approved_at DESC NULLS LAST;")
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    return [_deserialize_request(dict(r)) for r in rows]


def get_user_requests(username: str) -> list:
    sql = "SELECT * FROM requests WHERE username = %s ORDER BY upload_time DESC;"
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (username,))
            rows = cur.fetchall()
    return [_deserialize_request(dict(r)) for r in rows]


def get_requests_by_user(username: str) -> list:
    """Backward-compatible alias for route code."""
    return get_user_requests(username)


def get_approved_request_by_file(file_id: str) -> dict | None:
    sql = "SELECT * FROM requests WHERE file_id = %s AND status = 'approved';"
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (file_id,))
            row = cur.fetchone()
    return _deserialize_request(dict(row)) if row else None


def get_approved_request_for_user(file_id: str, username: str) -> dict | None:
    sql = ("SELECT * FROM requests "
           "WHERE file_id = %s AND username = %s AND status = 'approved';")
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (file_id, username))
            row = cur.fetchone()
    return _deserialize_request(dict(row)) if row else None


def approve_request(request_id: str, admin_username: str,
                    admin_notes: str, approved_at: str) -> dict:
    sql = """
        UPDATE requests
        SET status       = 'approved',
            admin_action = 'Approved',
            admin_notes  = %s,
            approved_by  = %s,
            approved_at  = %s
        WHERE request_id = %s
        RETURNING *;
    """
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (admin_notes, admin_username, approved_at, request_id))
            row = cur.fetchone()
    return _deserialize_request(dict(row))


def reject_request(request_id: str, admin_username: str,
                   admin_notes: str, rejected_at: str) -> dict:
    sql = """
        UPDATE requests
        SET status       = 'rejected',
            admin_action = 'Rejected',
            admin_notes  = %s,
            rejected_by  = %s,
            rejected_at  = %s
        WHERE request_id = %s
        RETURNING *;
    """
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (admin_notes, admin_username, rejected_at, request_id))
            row = cur.fetchone()
    return _deserialize_request(dict(row))


def pending_count() -> int:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM requests WHERE status = 'pending';")
            return cur.fetchone()[0]


def approved_count() -> int:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM requests WHERE status = 'approved';")
            return cur.fetchone()[0]


def database_status() -> dict:
    """Return a lightweight PostgreSQL health summary for the UI."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), version();")
            database_name, version = cur.fetchone()

    return {
        "status": "connected",
        "engine": "PostgreSQL",
        "database": database_name,
        "version": version.split()[1] if version else "",
    }


def all_requests_ml_details() -> list:
    """Return ml_details dicts for all requests (for stats aggregation)."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT ml_details FROM requests;")
            rows = cur.fetchall()
    result = []
    for (ml_str,) in rows:
        try:
            result.append(json.loads(ml_str) if ml_str else {})
        except Exception:
            result.append({})
    return result


# ── Activity log ──────────────────────────────────────────────────────────────

def log_activity(username: str, activity: str, details: str = '') -> None:
    insert_sql = """
        INSERT INTO user_activity_log (username, activity, details)
        VALUES (%s, %s, %s);
    """
    prune_sql = """
        DELETE FROM user_activity_log
        WHERE id IN (
            SELECT id FROM user_activity_log
            WHERE username = %s
            ORDER BY ts DESC
            OFFSET 100
        );
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(insert_sql, (username, activity, details or ''))
            cur.execute(prune_sql, (username,))


def get_user_activity(username: str, count: int = 20) -> list:
    sql = """
        SELECT activity, details, ts
        FROM user_activity_log
        WHERE username = %s
        ORDER BY ts DESC
        LIMIT %s;
    """
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (username, count))
            rows = cur.fetchall()
    return [dict(r) for r in rows]


# ── File access log ───────────────────────────────────────────────────────────

def log_file_access(file_id: str, username: str,
                    action: str, success: bool = True) -> None:
    sql = """
        INSERT INTO file_access_log (file_id, username, action, success)
        VALUES (%s, %s, %s, %s);
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (file_id, username, action, success))


def get_file_access_count(file_id: str) -> int:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM file_access_log WHERE file_id = %s;",
                (file_id,))
            return cur.fetchone()[0]


def get_last_file_access_time(file_id: str) -> str | None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ts FROM file_access_log WHERE file_id = %s "
                "ORDER BY ts DESC LIMIT 1;",
                (file_id,))
            row = cur.fetchone()
    return str(row[0]) if row else None


# ── Maintenance ───────────────────────────────────────────────────────────────

def cleanup_old_requests() -> int:
    """Delete pending requests older than 7 days and their encrypted files on disk.
    Approved/rejected requests are never auto-deleted."""
    select_sql = """
        SELECT r.request_id, r.file_id, f.path
        FROM requests r
        LEFT JOIN files f ON r.file_id = f.file_id
        WHERE r.status = 'pending'
          AND r.upload_time < NOW() - INTERVAL '7 days';
    """
    from modules.storage_utils import storage_service
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(select_sql)
            expired = cur.fetchall()
            removed = 0
            for req_id, file_id, path in expired:
                if path:
                    storage_service.delete_file(path)
                cur.execute("DELETE FROM requests WHERE request_id = %s;", (req_id,))
                if file_id:
                    cur.execute("DELETE FROM files WHERE file_id = %s;", (file_id,))
                removed += 1
    return removed


# ── Internal helpers ──────────────────────────────────────────────────────────

def _deserialize_request(row: dict) -> dict:
    """Parse ml_details JSON string; stringify psycopg2 datetime objects."""
    if isinstance(row.get('ml_details'), str):
        try:
            row['ml_details'] = json.loads(row['ml_details'])
        except Exception:
            row['ml_details'] = {}
    elif row.get('ml_details') is None:
        row['ml_details'] = {}

    for key in ('upload_time', 'approved_at', 'rejected_at'):
        if isinstance(row.get(key), datetime.datetime):
            row[key] = row[key].isoformat()

    return row
