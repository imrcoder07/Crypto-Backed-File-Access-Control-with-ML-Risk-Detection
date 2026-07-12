import hashlib
import json
import datetime
import threading
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
