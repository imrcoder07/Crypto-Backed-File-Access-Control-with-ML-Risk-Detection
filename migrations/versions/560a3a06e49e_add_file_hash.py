"""add file hash column

Revision ID: 560a3a06e49e
Revises: 450b3a06e49e
Create Date: 2026-07-05 12:00:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '560a3a06e49e'
down_revision: Union[str, Sequence[str], None] = '450b3a06e49e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Add file_hash column to files table
    op.execute("ALTER TABLE files ADD COLUMN file_hash VARCHAR(64);")

def downgrade() -> None:
    # Remove file_hash column from files table
    op.execute("ALTER TABLE files DROP COLUMN file_hash;")
