import hashlib
import json
import datetime
import threading
import re
import queue as _queue_module

from modules.db import append_block, get_chain_length, get_recent_blocks

class TamperEvidentLedger:
    """Tamper-evident audit log with cryptographic hash chaining backed by PostgreSQL.

    Every ``add_event()`` call is O(1) — it enqueues the event string and
    returns immediately.  A single daemon thread drains the queue and appends
    blocks to PostgreSQL, so no Flask request thread is ever blocked waiting for mining.

    The old proof-of-work loop is replaced with an O(1) deterministic
    SHA-256 proof that still ties each block to the previous one.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._event_queue: _queue_module.Queue = _queue_module.Queue()
        
        # Ensure genesis block exists
        if get_chain_length() == 0:
            self._create_genesis()
            
        self._start_worker()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fast_proof(previous_hash: str, data: str) -> str:
        """O(1) deterministic proof: SHA-256(prev_hash + data)[:16].
        Replaces the unbounded proof-of-work loop.
        """
        return hashlib.sha256(f"{previous_hash}:{data}".encode()).hexdigest()[:16]

    @staticmethod
    def _build_block(index: int, data: str, proof: str, previous_hash: str) -> dict:
        """Construct a block dict and embed its own SHA-256 hash."""
        payload = {
            'index': index,
            'timestamp': str(datetime.datetime.now()),
            'data': data,
            'proof': proof,
            'previous_hash': previous_hash,
        }
        payload['block_hash'] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        return payload

    def _create_genesis(self) -> None:
        genesis_data = 'Genesis Block'
        proof = self._fast_proof('0', genesis_data)
        block = self._build_block(
            index=1, data=genesis_data, proof=proof, previous_hash='0'
        )
        append_block(
            block['index'], block['timestamp'], block['data'], 
            block['proof'], block['previous_hash'], block['block_hash']
        )
        print("✅ Audit Ledger genesis block created in PostgreSQL.")

    def _append_block(self, data: str) -> None:
        """Write a new block to the chain. Only called from the worker thread."""
        with self._lock:
            recent = get_recent_blocks(1)
            if recent:
                previous_block = recent[-1]
                index = previous_block['index'] + 1
                previous_hash = previous_block['block_hash']
            else:
                index = 1
                previous_hash = '0'
                
            proof = self._fast_proof(previous_hash, data)
            block = self._build_block(
                index=index,
                data=data,
                proof=proof,
                previous_hash=previous_hash,
            )
            
            append_block(
                block['index'], block['timestamp'], block['data'], 
                block['proof'], block['previous_hash'], block['block_hash']
            )
            print(f"[AUDIT] Audit Ledger Block #{block['index']} persisted to DB: {data[:100]}")

    def _worker(self) -> None:
        """Daemon thread: drains the event queue and persists blocks."""
        while True:
            data = self._event_queue.get()
            try:
                self._append_block(data)
            except Exception as exc:
                print(f"Audit Ledger worker error: {exc}")
            finally:
                self._event_queue.task_done()

    def _start_worker(self) -> None:
        t = threading.Thread(
            target=self._worker, daemon=True, name='audit-ledger-worker'
        )
        t.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_event(self, event_data: str) -> None:
        """Non-blocking: enqueue the event and return immediately (O(1))."""
        self._event_queue.put(str(event_data))

    def hash(self, block: dict) -> str:
        """SHA-256 of a block (used for hash chaining)."""
        return hashlib.sha256(
            json.dumps(block, sort_keys=True).encode()
        ).hexdigest()

    @property
    def chain_length(self) -> int:
        """Database snapshot of current chain length."""
        return get_chain_length()

    def get_recent_events(self, count: int = 10) -> list:
        """Database query of the most-recent events."""
        return get_recent_blocks(count)

    def get_chain_snapshot(self, count: int = 50) -> list:
        """Database query of the most-recent chain blocks."""
        return get_recent_blocks(count)

    def get_user_events(self, username: str, count: int = 20) -> list:
        """Scan of recent blocks mentioning *username*."""
        recent = get_recent_blocks(count * 5) # Scan a wider window
        return [b['data'] for b in recent if username in b['data']][:count]


# ==============================================================================
# Phase 3: Structured Audit Log Event Registry & Centralized Helper
# ==============================================================================

CANONICAL_EVENTS = {
    "ACCOUNT_CREATED": {"required": ["username", "details"]},
    "LOGIN_SUCCESS": {"required": ["username", "details"]},
    "LOGIN_FAILED": {"required": ["username", "details"]},
    "LOGOUT": {"required": ["username", "details"]},
    "FILE_UPLOAD": {"required": ["username", "file_id", "filename", "details"]},
    "FILE_ENCRYPT": {"required": ["username", "file_id", "filename", "details"]},
    "ACCESS_REQUEST_CREATED": {"required": ["username", "request_id", "file_id", "filename", "details"]},
    "ML_ANALYSIS_STARTED": {"required": ["request_id", "details"]},
    "ML_ANALYSIS_COMPLETED": {"required": ["request_id", "details"]},
    "ACCESS_REQUEST_APPROVED": {"required": ["admin", "request_id", "details"]},
    "ACCESS_REQUEST_REJECTED": {"required": ["admin", "request_id", "details"]},
    "FILE_DECRYPT": {"required": ["username", "file_id", "filename", "details"]},
    "FILE_DOWNLOAD": {"required": ["username", "file_id", "filename", "details"]},
    "ADMIN_FILE_DOWNLOAD": {"required": ["admin", "file_id", "filename", "details"]},
    "FILE_DELETED": {"required": ["admin", "request_id", "filename", "details"]},
    "STORAGE_INCONSISTENCY": {"required": ["admin", "request_id", "filename", "details"]},
    "ADMIN_USER_ACTIVITY_VIEW": {"required": ["admin", "username", "details"]},
}

SEVERITY_MAP = {
    "LOGIN_FAILED": "CRITICAL",
    "STORAGE_INCONSISTENCY": "CRITICAL",
    "FILE_DELETED": "WARNING",
    "ACCESS_REQUEST_REJECTED": "WARNING",
    "ACCOUNT_CREATED": "INFO",
    "LOGIN_SUCCESS": "INFO",
    "LOGOUT": "INFO",
    "FILE_UPLOAD": "INFO",
    "FILE_ENCRYPT": "INFO",
    "ACCESS_REQUEST_CREATED": "INFO",
    "ML_ANALYSIS_STARTED": "INFO",
    "ML_ANALYSIS_COMPLETED": "INFO",
    "ACCESS_REQUEST_APPROVED": "INFO",
    "FILE_DECRYPT": "INFO",
    "FILE_DOWNLOAD": "INFO",
    "ADMIN_FILE_DOWNLOAD": "INFO",
    "ADMIN_USER_ACTIVITY_VIEW": "INFO",
}

LEGACY_EVENT_PATTERNS = [
    (re.compile(r"logged in", re.I), "LOGIN_SUCCESS"),
    (re.compile(r"logged out", re.I), "LOGOUT"),
    (re.compile(r"uploaded file", re.I), "FILE_UPLOAD"),
    (re.compile(r"securely downloaded file", re.I), "FILE_DOWNLOAD"),
    (re.compile(r"downloaded encrypted blob", re.I), "ADMIN_FILE_DOWNLOAD"),
    (re.compile(r"approved access request", re.I), "ACCESS_REQUEST_APPROVED"),
    (re.compile(r"rejected access request", re.I), "ACCESS_REQUEST_REJECTED"),
    (re.compile(r"new user registered", re.I), "ACCOUNT_CREATED"),
]


def _contains_sensitive_data(val) -> bool:
    """Helper to detect potentially sensitive credentials, keys or tokens in audit values."""
    if isinstance(val, dict):
        for k, v in val.items():
            k_str = str(k).lower()
            if any(s in k_str for s in ["password", "token", "secret", "cookie", "jwt", "key", "fernet"]):
                return True
            if _contains_sensitive_data(v):
                return True
    elif isinstance(val, list):
        for item in val:
            if _contains_sensitive_data(item):
                return True
    elif val is not None:
        val_str = str(val).lower()
        if any(s in val_str for s in ["password=", "token=", "jwt=", "bearer ", "cookie="]):
            return True
    return False

def log_event(
    action: str,
    username: str | None = None,
    admin: str | None = None,
    request_id: str | None = None,
    file_id: str | None = None,
    filename: str | None = None,
    details: str | None = None,
    metadata: dict | None = None
) -> bool:
    """Centralized, registry-validated audit logging.
    
    Generates UTC ISO-8601 timestamps, validates inputs, prunes None/empty values,
    and enqueues standard structured JSON entries to the blockchain ledger.
    """
    import inspect
    import logging
    
    # 1. Generate UTC timestamp in ISO-8601 format (e.g. 2026-07-13T18:42:31Z)
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    timestamp = now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    missing_fields = []
    invalid_fields = []
    
    # 2. Validate action existence in canonical registry
    if action not in CANONICAL_EVENTS:
        invalid_fields.append("action (non-canonical)")
    else:
        # Check all required fields are present and not None/empty
        required_fields = CANONICAL_EVENTS[action]["required"]
        args = {
            "action": action,
            "username": username,
            "admin": admin,
            "request_id": request_id,
            "file_id": file_id,
            "filename": filename,
            "details": details,
        }
        for field in required_fields:
            if args.get(field) is None or str(args.get(field)).strip() == "":
                missing_fields.append(field)

    # 3. Check for any sensitive information leakage
    for k, v in [("username", username), ("admin", admin), ("details", details), ("metadata", metadata)]:
        if v is not None and _contains_sensitive_data(v):
            invalid_fields.append(f"{k} (contains sensitive/credential patterns)")

    # 4. Handle validation failures safely: return False and log a warning
    if missing_fields or invalid_fields:
        caller_frame = inspect.currentframe().f_back
        caller_module = caller_frame.f_globals.get('__name__') if caller_frame else 'unknown'
        
        logging.getLogger("security_audit").warning(
            "Security Audit Validation Failure: Action=%s | Missing=%s | Invalid=%s | Caller=%s | Timestamp=%s",
            action, missing_fields, invalid_fields, caller_module, timestamp
        )
        return False

    # 5. Build structured payload (omitting None and empty metadata values)
    payload = {
        "action": action,
        "details": details,
        "timestamp": timestamp,
    }
    if username is not None:
        payload["username"] = username
    if admin is not None:
        payload["admin"] = admin
    if request_id is not None:
        payload["request_id"] = request_id
    if file_id is not None:
        payload["file_id"] = file_id
    if filename is not None:
        payload["filename"] = filename
    if metadata is not None and len(metadata) > 0:
        payload["metadata"] = metadata

    # 6. Serialize to JSON and forward to the blockchain ledger queue
    try:
        from modules.extensions import audit_ledger
        event_str = json.dumps(payload, sort_keys=True)
        audit_ledger.add_event(event_str)
        return True
    except Exception as exc:
        logging.getLogger("security_audit").error(
            "Failed to enqueue audit event %s to ledger: %s", action, exc, exc_info=True
        )
        return False

