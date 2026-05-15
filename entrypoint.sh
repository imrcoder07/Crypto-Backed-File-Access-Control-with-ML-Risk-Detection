#!/bin/sh
set -e

echo "==> Running database migrations..."
alembic upgrade head
echo "==> Migrations complete. Starting Gunicorn..."

exec gunicorn app:app --bind 0.0.0.0:${PORT:-8000}
