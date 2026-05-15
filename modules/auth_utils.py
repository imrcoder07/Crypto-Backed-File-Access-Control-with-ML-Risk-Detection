import bcrypt
from modules.db import get_user, create_user, get_all_users

def hash_password(password: str) -> str:
    """Hash password with bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(hashed_password: str, user_password: str) -> bool:
    """Verify password against bcrypt hash"""
    try:
        return bcrypt.checkpw(user_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception:
        return False

def seed_default_admin():
    """Seed the default admin account if no users exist in the database."""
    users = get_all_users()
    if not users:
        print("Seeding default admin user...")
        create_user(
            username="admin",
            password_hash=hash_password("admin123"),
            role="Admin",
            email="admin@secure.corp",
            department="Security"
        )

