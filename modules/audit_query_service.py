import json
import datetime
import re
import logging
from modules.audit_utils import SEVERITY_MAP, LEGACY_EVENT_PATTERNS

logger = logging.getLogger(__name__)

# Configurable max records constant
AUDIT_QUERY_MAX_RECORDS = 10000

class AuditQueryFilter:
    """Standardized filter object for audit queries."""
    def __init__(
        self,
        query=None,
        username=None,
        admin=None,
        action=None,
        severity=None,
        start_date=None,
        end_date=None,
        limit=50,
        offset=0
    ):
        self.query = self._sanitize_search_query(query)
        self.username = username.strip() if username and str(username).strip() else None
        self.admin = admin.strip() if admin and str(admin).strip() else None
        self.action = action.strip() if action and str(action).strip() else None
        self.severity = severity.strip().upper() if severity and str(severity).strip() else None
        self.start_date = start_date.strip() if start_date and str(start_date).strip() else None
        self.end_date = end_date.strip() if end_date and str(end_date).strip() else None
        
        try:
            self.limit = max(1, min(int(limit or 50), 100))
        except (ValueError, TypeError):
            self.limit = 50
            
        try:
            self.offset = max(0, int(offset or 0))
        except (ValueError, TypeError):
            self.offset = 0

    def _sanitize_search_query(self, q):
        if not q:
            return None
        q = str(q).strip()
        # Collapse multiple spaces to a single space
        q = re.sub(r'\s+', ' ', q)
        return q if q else None

    def to_dict(self):
        """Helper to return applied filters to client for state sync."""
        res = {}
        if self.query:
            res["query"] = self.query
        if self.username:
            res["username"] = self.username
        if self.admin:
            res["admin"] = self.admin
        if self.action:
            res["action"] = self.action
        if self.severity:
            res["severity"] = self.severity
        if self.start_date:
            res["start_date"] = self.start_date
        if self.end_date:
            res["end_date"] = self.end_date
        return res

def parse_utc_timestamp(ts_val) -> datetime.datetime | None:
    """Safely parse various datetime/string formats into a UTC-aware datetime object."""
    if isinstance(ts_val, datetime.datetime):
        if ts_val.tzinfo is None:
            return ts_val.replace(tzinfo=datetime.timezone.utc)
        return ts_val.astimezone(datetime.timezone.utc)
    
    ts_str = str(ts_val).strip()
    # Try parsing common formats
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.datetime.strptime(ts_str, fmt)
            return dt.replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
            
    # ISO-8601 parser fallback for fractional seconds
    try:
        # standard ISO format support: e.g. 2026-07-13T19:00:00.123456+00:00
        dt = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.astimezone(datetime.timezone.utc)
    except ValueError:
        return None

def normalize_record(block: dict) -> dict:
    """Normalization layer mapping both legacy plaintext and structured JSON to a single schema."""
    data_raw = block.get('data') or ''
    index = block.get('index') or 1
    
    # 1. Default schema fields
    action = "UNKNOWN_EVENT"
    username = None
    admin = None
    request_id = None
    file_id = None
    filename = None
    details = data_raw
    metadata = None
    
    # Check block timestamp from DB
    block_ts = block.get('timestamp')
    timestamp_dt = parse_utc_timestamp(block_ts)
    
    # Format to standard ISO-8601 UTC
    if timestamp_dt:
        timestamp = timestamp_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    else:
        timestamp = str(block_ts)
        
    is_genesis = (index == 1 or data_raw == "Genesis Block")
    
    # 2. Try parsing structured JSON
    is_structured = False
    if not is_genesis:
        try:
            parsed = json.loads(data_raw)
            if isinstance(parsed, dict) and "action" in parsed:
                is_structured = True
                action = parsed.get("action")
                username = parsed.get("username")
                admin = parsed.get("admin")
                request_id = parsed.get("request_id")
                file_id = parsed.get("file_id")
                filename = parsed.get("filename")
                details = parsed.get("details") or details
                metadata = parsed.get("metadata")
                # Prefer timestamp embedded in the event
                embed_ts = parsed.get("timestamp")
                if embed_ts:
                    embed_dt = parse_utc_timestamp(embed_ts)
                    if embed_dt:
                        timestamp = embed_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        except Exception:
            pass

    # 3. Fallback for legacy plaintext mapping
    if not is_structured:
        if is_genesis:
            action = "SYSTEM_INIT"
        else:
            # Match regex patterns to find action
            for regex, act_name in LEGACY_EVENT_PATTERNS:
                if regex.search(data_raw):
                    action = act_name
                    break
            
            # Extract legacy actors
            user_match = re.search(r"User\s+'([^']+)'", data_raw, re.I)
            admin_match = re.search(r"Admin\s+'([^']+)'", data_raw, re.I)
            reg_match = re.search(r"New user registered:\s*([^\s]+)", data_raw, re.I)
            
            if user_match:
                username = user_match.group(1)
            elif reg_match:
                username = reg_match.group(1)
                
            if admin_match:
                admin = admin_match.group(1)

    # 4. Deriving severity on demand (Never stored directly in ledger blocks)
    severity = SEVERITY_MAP.get(action, "INFO")
    
    return {
        "action": action,
        "username": username,
        "admin": admin,
        "request_id": request_id,
        "file_id": file_id,
        "filename": filename,
        "details": details,
        "timestamp": timestamp,
        "severity": severity,
        "metadata": metadata
    }

def match_filter(rec: dict, filter_obj: AuditQueryFilter) -> bool:
    """Generic query filter matching engine."""
    # 1. Action filter
    if filter_obj.action and rec["action"] != filter_obj.action:
        return False
        
    # 2. Username filter
    if filter_obj.username and rec["username"] != filter_obj.username:
        return False
        
    # 3. Admin filter
    if filter_obj.admin and rec["admin"] != filter_obj.admin:
        return False
        
    # 4. Severity filter
    if filter_obj.severity and rec["severity"] != filter_obj.severity:
        return False
        
    # 5. Date filtering (ISO-8601 comparison boundaries interpreted in UTC)
    rec_time = parse_utc_timestamp(rec["timestamp"])
    if rec_time:
        if filter_obj.start_date:
            # If only date supplied, beginning of UTC day
            start_val = filter_obj.start_date
            if len(start_val) == 10:
                start_val += "T00:00:00Z"
            start_time = parse_utc_timestamp(start_val)
            if start_time and rec_time < start_time:
                return False
                
        if filter_obj.end_date:
            # If only date supplied, end of UTC day
            end_val = filter_obj.end_date
            if len(end_val) == 10:
                end_val += "T23:59:59Z"
            end_time = parse_utc_timestamp(end_val)
            if end_time and rec_time > end_time:
                return False

    # 6. Case-insensitive query search across indexed fields only
    if filter_obj.query:
        search_term = filter_obj.query.lower()
        searchable_fields = [
            rec.get("details"),
            rec.get("username"),
            rec.get("admin"),
            rec.get("filename"),
            rec.get("request_id")
        ]
        found = False
        for field in searchable_fields:
            if field and search_term in str(field).lower():
                found = True
                break
        if not found:
            return False
            
    return True

class AuditQueryService:
    """Decoupled query processing and investigation engine."""
    @staticmethod
    def query_audit_logs(filter_obj: AuditQueryFilter) -> dict:
        import time
        from flask import current_app
        from modules.extensions import audit_ledger
        
        # 1. Fetch configurable limit dynamically from Flask application context
        max_records = 10000
        try:
            if current_app:
                max_records = current_app.config.get("AUDIT_QUERY_MAX_RECORDS", 10000)
        except RuntimeError:
            pass # Outside Flask context (e.g. CLI scripts or basic unit testing)
            
        t_start = time.perf_counter()
        
        # Retrieve raw ledger snapshots
        blocks = audit_ledger.get_chain_snapshot(count=max_records)
        raw_count = len(blocks)
        
        # 2. Normalization
        normalized = []
        for b in blocks:
            try:
                normalized.append(normalize_record(b))
            except Exception as e:
                logger.error(f"Failed to normalize block #{b.get('index')}: {e}", exc_info=True)
        
        normalized_count = len(normalized)
                
        # 3. Filtering
        filtered = []
        for rec in normalized:
            if match_filter(rec, filter_obj):
                filtered.append(rec)
        
        # Reverse list to ensure newest-first (new to old) order
        filtered.reverse()
                
        filtered_count = len(filtered)
                
        # 4. Sorting & Pagination
        total = filtered_count
        sliced = filtered[filter_obj.offset : filter_obj.offset + filter_obj.limit]
        
        t_duration = time.perf_counter() - t_start
        
        # Performance logging instrumentation (Phase 5 Refinement)
        logger.info(
            "Audit Search Query Executed: duration=%.4fs | max_records=%d | raw_blocks=%d | "
            "normalized=%d | filtered=%d | returned=%d | offset=%d | limit=%d",
            t_duration, max_records, raw_count, normalized_count, filtered_count, len(sliced),
            filter_obj.offset, filter_obj.limit
        )
        
        return {
            "activities": sliced,
            "pagination": {
                "limit": filter_obj.limit,
                "offset": filter_obj.offset,
                "total": total,
                "has_more": (filter_obj.offset + len(sliced)) < total
            },
            "filters": filter_obj.to_dict()
        }
