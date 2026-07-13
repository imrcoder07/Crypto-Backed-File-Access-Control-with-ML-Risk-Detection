import os
import threading
import time
import logging
from flask import Flask, Response, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv

# Initialize dotenv
load_dotenv()

# Configure Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import the core modules
from modules import db

# Import Blueprints
from routes.auth import auth_bp
from routes.main import main_bp
from routes.user import user_bp
from routes.admin import admin_bp

app = Flask(__name__)

# Apply ProxyFix middleware to handle Render's reverse proxy correctly
# x_proto=1 ensures HTTPS requests are recognized as secure
# x_host=1 ensures Host headers are respected
app.wsgi_app = ProxyFix(
    app.wsgi_app,
    x_proto=1,
    x_host=1
)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static', 'images'),
        'favicon.ico',
        mimetype='image/x-icon',
        max_age=604800  # cache for 7 days
    )

# Security Hardening: Enforce 12-Factor App secrets and secure sessions
secret_key = os.environ.get("SECRET_KEY")
if not secret_key:
    raise RuntimeError("SECRET_KEY is not set. Add it to your .env file for production security.")
app.secret_key = secret_key

# Session Security Config
app.config.update(
    SESSION_COOKIE_SECURE=True,     # Requires HTTPS
    SESSION_COOKIE_HTTPONLY=True,   # Prevents client-side JS from reading the cookie
    SESSION_COOKIE_SAMESITE='Lax'   # CSRF protection
)

# Initialize PostgreSQL Database schema
with app.app_context():
    db.init_db()

# Initialize Rate Limiter
from modules.extensions import limiter
limiter.init_app(app)

# Register Blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(main_bp)
app.register_blueprint(user_bp)
app.register_blueprint(admin_bp)

# Background Cleanup Task
def cleanup_worker():
    while True:
        try:
            removed = db.cleanup_old_requests()
            if removed > 0:
                logger.info(f"🧹 Cleanup: Removed {removed} expired pending requests.")
        except Exception as e:
            logger.error(f"Cleanup task error: {e}", exc_info=True)
        time.sleep(3600)  # run once an hour

def start_cleanup_scheduler():
    thread = threading.Thread(target=cleanup_worker, daemon=True)
    thread.start()

# Startup Diagnostics
flask_env = os.environ.get("FLASK_ENV", "development")
port_binding = os.environ.get("PORT", 5000)
run_bg_tasks = os.environ.get("RUN_BACKGROUND_TASKS") == "true"

logger.info("🚀 Application starting up...")
logger.info(f"⚙️ FLASK_ENV: {flask_env}")
logger.info(f"🔌 Port Binding: {port_binding}")
logger.info("🛡️ ProxyFix Middleware: Active (x_proto=1, x_host=1)")

# In production under Gunicorn, multiple workers importing app.py would spawn multiple 
# overlapping background threads, causing database locking risks and race conditions.
# We disable automatic scheduler startup in production to prepare for future migration 
# to a dedicated worker service (like Celery) or a single cron-based execution.
if flask_env == "production" and not run_bg_tasks:
    logger.info("⏳ Background cleanup thread disabled to ensure operational stability.")
else:
    logger.info("⏳ Starting background cleanup scheduler.")
    start_cleanup_scheduler()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
