from modules.audit_utils import TamperEvidentLedger
from modules.ml_analyzer import MLRiskAnalyzer
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Initialize singletons for the application
audit_ledger = TamperEvidentLedger()
ml_analyzer = MLRiskAnalyzer()

# Initialize Rate Limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)
