# Structured Logging & Diagnostics Strategy

**Purpose:**
To provide a production-safe, deterministic, and easily searchable logging infrastructure for Render's log aggregation interface.

**Production Risk Addressed:**
Previously, the application utilized naked `print()` statements for debugging. Standard prints are often buffered asynchronously, lack critical context (timestamps, severity levels, module names), and fail to securely capture Python exception tracebacks, making production incident response nearly impossible.

**Implementation Details:**
- **Centralized Configuration:** Replaced `print()` with Python's built-in `logging` module in `app.py`. Configured using `logging.basicConfig()` with `level=logging.INFO` and a structured format: `%(asctime)s [%(levelname)s] %(name)s - %(message)s`.
- **Exception Tracebacks:** Replaced bare `except Exception as e: print(...)` calls in `storage_utils.py` and `app.py` with `logger.error(..., exc_info=True)`. This ensures complete stack traces are piped to Render without crashing the runtime.
- **Startup Diagnostics:** Injected a strict diagnostic block upon application boot. It safely logs `FLASK_ENV`, injected `$PORT`, ProxyFix status, and Scheduler Activation status. *No secrets, tokens, or credentials are included in this output.*

**Operational Verification Steps:**
1. During deployment, monitor the Render "Logs" tab.
2. Look for the `[INFO]` sequence beginning with `🚀 Application starting up...`.
3. Verify that the startup variables accurately reflect the Render dashboard settings.
