import os
import logging
import bcrypt
from modules.db import get_user, create_user, get_all_users

logger = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt (cost factor 12)."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12)).decode('utf-8')


def check_password(hashed_password: str, user_password: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(user_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception:
        return False


def bootstrap_admin_from_env() -> bool:
    """Create the admin account from environment variables on first deployment.

    Reads ADMIN_USERNAME and ADMIN_PASSWORD from the environment.
    If both are set and no account exists with that username, the account
    is created with role='Admin'.

    Returns True if an account was created, False if skipped (already exists
    or env vars not set).

    Designed to be called from entrypoint.sh before Gunicorn starts.
    Safe to call on every deploy — idempotent via ON CONFLICT DO UPDATE in db.
    """
    logger.info("Admin bootstrap: Started.")
    try:
        username = os.environ.get("ADMIN_USERNAME", "").strip()
        password = os.environ.get("ADMIN_PASSWORD", "").strip()

        if not username or not password:
            logger.info("Admin bootstrap: Finished. Skipped because ADMIN_USERNAME or ADMIN_PASSWORD not set.")
            return False

        logger.info(f"Admin bootstrap: ADMIN_USERNAME '{username}' detected.")

        if len(password) < 12:
            logger.error(
                "Admin bootstrap: Aborted. ADMIN_PASSWORD must be at least 12 characters. "
                "Set a stronger password in your Render environment variables."
            )
            return False

        logger.info(f"Admin bootstrap: Checking if admin user '{username}' exists in database...")
        existing = get_user(username)
        if existing:
            if existing.get('role') == 'Admin':
                logger.info(f"Admin bootstrap: Finished. Account '{username}' already exists as Admin. Skipping.")
            else:
                logger.warning(
                    f"Admin bootstrap: Finished. Account '{username}' exists with role '{existing['role']}'. "
                    "Not overwriting role. Use 'python manage_admin.py promote-user' to promote."
                )
            return False

        logger.info(f"Admin bootstrap: Creating admin user '{username}'...")
        create_user(
            username=username,
            password_hash=hash_password(password),
            role="Admin",
            email=f"{username}@admin.internal",
            department="Security",
        )
        logger.info(f"Admin bootstrap: Finished. Account '{username}' created with role=Admin successfully.")
        return True
    except Exception as e:
        logger.exception("Admin bootstrap: Failed with an unexpected exception.")
        return False
