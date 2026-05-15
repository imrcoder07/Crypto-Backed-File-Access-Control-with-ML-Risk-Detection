"""initial schema

Revision ID: 450b3a06e49e
Revises: 
Create Date: 2026-05-13 19:59:57.841842

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '450b3a06e49e'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("""
    CREATE TABLE IF NOT EXISTS files (
        file_id           TEXT PRIMARY KEY,
        filename          TEXT NOT NULL,
        safe_filename     TEXT,
        path              TEXT NOT NULL,
        salt              TEXT NOT NULL,
        owner             TEXT NOT NULL,
        uploaded_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        file_size         BIGINT,
        file_size_mb      FLOAT,
        requires_password BOOLEAN DEFAULT TRUE
    );

    CREATE TABLE IF NOT EXISTS requests (
        request_id        TEXT PRIMARY KEY,
        file_id           TEXT REFERENCES files(file_id) ON DELETE CASCADE,
        filename          TEXT NOT NULL,
        username          TEXT NOT NULL,
        user_role         TEXT,
        upload_time       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        status            TEXT NOT NULL DEFAULT 'pending',
        ml_verdict        TEXT,
        ml_details        TEXT,
        admin_action      TEXT DEFAULT 'Waiting for Review',
        admin_notes       TEXT DEFAULT '',
        requires_password BOOLEAN DEFAULT TRUE,
        password_provided BOOLEAN DEFAULT TRUE,
        file_size         BIGINT,
        file_size_mb      FLOAT,
        approved_by       TEXT,
        approved_at       TIMESTAMPTZ,
        rejected_by       TEXT,
        rejected_at       TIMESTAMPTZ
    );

    CREATE TABLE IF NOT EXISTS user_activity_log (
        id       BIGSERIAL PRIMARY KEY,
        username TEXT NOT NULL,
        activity TEXT NOT NULL,
        details  TEXT DEFAULT '',
        ts       TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_ual_username ON user_activity_log (username);

    CREATE TABLE IF NOT EXISTS file_access_log (
        id       BIGSERIAL PRIMARY KEY,
        file_id  TEXT,
        username TEXT NOT NULL,
        action   TEXT NOT NULL,
        success  BOOLEAN NOT NULL DEFAULT TRUE,
        ts       TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_fal_file_id ON file_access_log (file_id);

    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'User',
        email TEXT,
        department TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS audit_ledger (
        index BIGSERIAL PRIMARY KEY,
        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        data TEXT NOT NULL,
        proof TEXT NOT NULL,
        previous_hash TEXT NOT NULL,
        block_hash TEXT NOT NULL
    );
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("""
    DROP TABLE IF EXISTS audit_ledger CASCADE;
    DROP TABLE IF EXISTS users CASCADE;
    DROP TABLE IF EXISTS file_access_log CASCADE;
    DROP TABLE IF EXISTS user_activity_log CASCADE;
    DROP TABLE IF EXISTS requests CASCADE;
    DROP TABLE IF EXISTS files CASCADE;
    """)
