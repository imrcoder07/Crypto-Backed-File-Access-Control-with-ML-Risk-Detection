#!/usr/bin/env python3
"""
manage_admin.py — Admin account management CLI for Crypto Access Control.

Usage:
    python manage_admin.py create-admin
    python manage_admin.py reset-password <username>
    python manage_admin.py promote-user <username>
    python manage_admin.py demote-user <username>
    python manage_admin.py list-admins
    python manage_admin.py verify <username>

All commands require DATABASE_URL to be set in the environment (or .env file).
No credentials are hardcoded. Passwords are always read interactively via
getpass — they are never echoed to the terminal or stored in shell history.

Run from the project root directory:
    python manage_admin.py <command>

On Render: use the Shell tab in the Dashboard to run this script.
"""

import sys
import os
import logging
import getpass

# ── Bootstrap: load .env and configure logging ──────────────────────────────
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# ── Validate DATABASE_URL before importing DB modules ───────────────────────
if not os.environ.get("DATABASE_URL"):
    print("ERROR: DATABASE_URL is not set. Add it to your .env file or export it.", file=sys.stderr)
    sys.exit(1)

from modules.db import get_user, get_all_users, create_user, update_user_password, promote_user_role
from modules.auth_utils import hash_password, bootstrap_admin_from_env


# ── Helpers ──────────────────────────────────────────────────────────────────

def _prompt_new_password(min_length: int = 12) -> str:
    """Prompt for a new password twice, enforce minimum length."""
    while True:
        pw1 = getpass.getpass("Enter new password (min 12 chars): ")
        if len(pw1) < min_length:
            print(f"Password too short. Minimum {min_length} characters required.")
            continue
        pw2 = getpass.getpass("Confirm new password: ")
        if pw1 != pw2:
            print("Passwords do not match. Try again.")
            continue
        return pw1


def _get_username_arg(args: list, command: str) -> str:
    if len(args) < 3:
        print(f"Usage: python manage_admin.py {command} <username>", file=sys.stderr)
        sys.exit(1)
    return args[2].strip()


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_create_admin():
    """Interactively create a new Admin account."""
    print("\n=== Create Admin Account ===")
    username = input("Username: ").strip()

    if not username or len(username) < 3:
        print("ERROR: Username must be at least 3 characters.", file=sys.stderr)
        sys.exit(1)

    existing = get_user(username)
    if existing:
        print(f"ERROR: User '{username}' already exists with role '{existing['role']}'.")
        print("Use 'promote-user' to make an existing user an Admin.")
        sys.exit(1)

    password = _prompt_new_password()
    email = input("Email (press Enter to skip): ").strip() or f"{username}@admin.internal"
    department = input("Department (press Enter to use 'Security'): ").strip() or "Security"

    create_user(
        username=username,
        password_hash=hash_password(password),
        role="Admin",
        email=email,
        department=department,
    )
    print(f"\n✅ Admin account '{username}' created successfully.")


def cmd_reset_password(args: list):
    """Reset password for any existing user."""
    username = _get_username_arg(args, "reset-password")
    print(f"\n=== Reset Password for '{username}' ===")

    user = get_user(username)
    if not user:
        print(f"ERROR: User '{username}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found user: {username} (role={user['role']})")
    password = _prompt_new_password()

    updated = update_user_password(username, hash_password(password))
    if updated:
        print(f"\n✅ Password for '{username}' updated successfully.")
    else:
        print(f"\nERROR: Password update failed for '{username}'.", file=sys.stderr)
        sys.exit(1)


def cmd_promote_user(args: list):
    """Promote an existing User to Admin role."""
    username = _get_username_arg(args, "promote-user")
    print(f"\n=== Promote '{username}' to Admin ===")

    user = get_user(username)
    if not user:
        print(f"ERROR: User '{username}' not found.", file=sys.stderr)
        sys.exit(1)

    if user['role'] == 'Admin':
        print(f"'{username}' is already an Admin. Nothing to do.")
        return

    confirm = input(f"Promote '{username}' from '{user['role']}' to 'Admin'? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return

    updated = promote_user_role(username, "Admin")
    if updated:
        print(f"\n✅ '{username}' is now an Admin.")
    else:
        print(f"\nERROR: Promotion failed.", file=sys.stderr)
        sys.exit(1)


def cmd_demote_user(args: list):
    """Demote an Admin back to User role."""
    username = _get_username_arg(args, "demote-user")
    print(f"\n=== Demote '{username}' to User ===")

    user = get_user(username)
    if not user:
        print(f"ERROR: User '{username}' not found.", file=sys.stderr)
        sys.exit(1)

    if user['role'] != 'Admin':
        print(f"'{username}' is already a '{user['role']}'. Nothing to do.")
        return

    # Safety check: ensure at least one other admin will remain
    all_admins = [u for u in get_all_users() if u['role'] == 'Admin']
    if len(all_admins) <= 1:
        print("ERROR: Cannot demote the last Admin. Create another Admin first.", file=sys.stderr)
        sys.exit(1)

    confirm = input(f"Demote '{username}' from 'Admin' to 'User'? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return

    updated = promote_user_role(username, "User")
    if updated:
        print(f"\n✅ '{username}' is now a regular User.")
    else:
        print(f"\nERROR: Demotion failed.", file=sys.stderr)
        sys.exit(1)


def cmd_list_admins():
    """List all Admin accounts in the database."""
    print("\n=== Admin Accounts ===")
    users = get_all_users()
    admins = [u for u in users if u.get('role') == 'Admin']

    if not admins:
        print("No admin accounts found.")
        return

    print(f"{'Username':<20} {'Email':<35} {'Created'}")
    print("-" * 75)
    for u in admins:
        created = str(u.get('created_at', 'N/A'))[:19]
        print(f"{u['username']:<20} {u.get('email', 'N/A'):<35} {created}")
    print(f"\nTotal admins: {len(admins)}")


def cmd_verify(args: list):
    """Verify a user exists and display their role."""
    username = _get_username_arg(args, "verify")
    user = get_user(username)
    if not user:
        print(f"User '{username}' does NOT exist.")
        sys.exit(1)
    print(f"\nUser found:")
    print(f"  Username  : {user['username']}")
    print(f"  Role      : {user['role']}")
    print(f"  Email     : {user.get('email', 'N/A')}")
    print(f"  Department: {user.get('department', 'N/A')}")
    print(f"  Created   : {str(user.get('created_at', 'N/A'))[:19]}")


def cmd_bootstrap():
    """Run the environment-variable bootstrap (same as entrypoint.sh calls)."""
    print("\n=== Admin Bootstrap from Environment Variables ===")
    print("Reading ADMIN_USERNAME and ADMIN_PASSWORD from environment...")
    created = bootstrap_admin_from_env()
    if created:
        print("✅ Admin account created from environment variables.")
    else:
        print("ℹ️  Bootstrap skipped (see logs above for reason).")


# ── Main dispatcher ───────────────────────────────────────────────────────────

COMMANDS = {
    "create-admin":   cmd_create_admin,
    "reset-password": cmd_reset_password,
    "promote-user":   cmd_promote_user,
    "demote-user":    cmd_demote_user,
    "list-admins":    cmd_list_admins,
    "verify":         cmd_verify,
    "bootstrap":      cmd_bootstrap,
}

HELP_TEXT = """
Crypto Access Control — Admin Management CLI

Commands:
  create-admin              Create a new Admin account interactively
  reset-password <user>     Reset any user's password
  promote-user <user>       Elevate a User to Admin role
  demote-user <user>        Demote an Admin back to User role
  list-admins               List all Admin accounts
  verify <user>             Check if a user exists and display their role
  bootstrap                 Run env-var bootstrap (same as entrypoint.sh)

Examples:
  python manage_admin.py create-admin
  python manage_admin.py reset-password alice
  python manage_admin.py promote-user bob
  python manage_admin.py list-admins
  python manage_admin.py verify admin

Requirements:
  DATABASE_URL must be set (via .env file or shell environment).
  Never set credentials in source code or commit them to git.
"""

def main():
    args = sys.argv
    if len(args) < 2 or args[1] in ("-h", "--help", "help"):
        print(HELP_TEXT)
        sys.exit(0)

    command = args[1]
    if command not in COMMANDS:
        print(f"ERROR: Unknown command '{command}'.", file=sys.stderr)
        print(f"Available commands: {', '.join(COMMANDS.keys())}", file=sys.stderr)
        sys.exit(1)

    func = COMMANDS[command]
    # Commands that need args receive the full args list
    if command in ("reset-password", "promote-user", "demote-user", "verify"):
        func(args)
    else:
        func()


if __name__ == "__main__":
    main()
