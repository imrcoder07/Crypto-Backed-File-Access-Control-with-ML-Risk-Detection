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
echo "==> Bootstrap complete."

if [ "$USE_ASYNC_ML" = "true" ]; then
    echo "==> Starting Celery background worker (solo pool)..."
    celery -A worker worker --pool=solo --loglevel=info &
else
    echo "==> Async ML disabled (USE_ASYNC_ML is not set to true). Skipping Celery worker."
fi

echo "==> Starting Gunicorn..."
exec gunicorn app:app --bind 0.0.0.0:${PORT:-8000}

