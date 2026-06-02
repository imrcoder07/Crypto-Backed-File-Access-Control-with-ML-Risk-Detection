#!/bin/sh
set -e

echo "==> Running database migrations..."
alembic upgrade head
echo "==> Migrations complete."

echo "==> Running admin bootstrap (no-op if ADMIN_USERNAME not set)..."
python -u -c "
import os, sys, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
from dotenv import load_dotenv
load_dotenv()
from modules.auth_utils import bootstrap_admin_from_env
try:
    created = bootstrap_admin_from_env()
    if created:
        print('Admin account created successfully.')
except Exception as e:
    logging.exception('Unexpected error during admin bootstrap execution:')
    sys.exit(1)
" || echo "⚠️ Admin bootstrap encountered an error or timed out, but continuing startup sequence..."

echo "==> Bootstrap complete. Starting Gunicorn..."

exec gunicorn app:app --bind 0.0.0.0:${PORT:-8000}
