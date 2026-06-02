import os
from modules.audit_utils import TamperEvidentLedger
from modules.ml_analyzer import MLRiskAnalyzer
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Initialize singletons for the application
audit_ledger = TamperEvidentLedger()
ml_analyzer = MLRiskAnalyzer()

# Rate Limiter storage:
#   - Production (Phase 3+): set REDIS_URL env var → counters persist across restarts
#   - Development / no Redis: falls back to in-process memory (resets on restart)
_limiter_storage_uri = os.environ.get("REDIS_URL", "memory://")

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=_limiter_storage_uri,
)

