"""add file deletion columns

Revision ID: 670a3a06e49e
Revises: 560a3a06e49e
Create Date: 2026-07-12 17:30:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '670a3a06e49e'
down_revision: Union[str, Sequence[str], None] = '560a3a06e49e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.execute("ALTER TABLE requests ADD COLUMN IF NOT EXISTS file_deleted BOOLEAN DEFAULT FALSE;")
    op.execute("ALTER TABLE requests ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP;")
    op.execute("ALTER TABLE requests ADD COLUMN IF NOT EXISTS deleted_by VARCHAR(120);")
    op.execute("ALTER TABLE requests ADD COLUMN IF NOT EXISTS storage_status VARCHAR(50) DEFAULT 'Stored';")

def downgrade() -> None:
    op.execute("ALTER TABLE requests DROP COLUMN IF EXISTS file_deleted;")
    op.execute("ALTER TABLE requests DROP COLUMN IF EXISTS deleted_at;")
    op.execute("ALTER TABLE requests DROP COLUMN IF EXISTS deleted_by;")
    op.execute("ALTER TABLE requests DROP COLUMN IF EXISTS storage_status;")
