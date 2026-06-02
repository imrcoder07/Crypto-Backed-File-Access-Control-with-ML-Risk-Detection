#!/bin/sh
set -e

echo "==> Running database migrations..."
alembic upgrade head
echo "==> Migrations complete."

echo "==> Running admin bootstrap (no-op if ADMIN_USERNAME not set)..."
python -c "
import os, sys, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
from dotenv import load_dotenv
load_dotenv()
from modules.auth_utils import bootstrap_admin_from_env
created = bootstrap_admin_from_env()
if created:
    print('Admin account created successfully.')
"
echo "==> Bootstrap complete. Starting Gunicorn..."

exec gunicorn app:app --bind 0.0.0.0:${PORT:-8000}
