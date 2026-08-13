"""Add optional organization and contact fields to users.

Revision ID: b7c2a11d9e41
Revises: 922e7a387820
"""

from alembic import op
import sqlalchemy as sa


revision = "b7c2a11d9e41"
down_revision = "922e7a387820"
branch_labels = None
depends_on = None


def _column_names() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    return {column["name"] for column in inspector.get_columns("user")}


def upgrade():
    columns = _column_names()
    if "organization" not in columns:
        op.add_column("user", sa.Column("organization", sa.String(), nullable=True))
    if "contact" not in columns:
        op.add_column("user", sa.Column("contact", sa.String(), nullable=True))


def downgrade():
    columns = _column_names()
    if "contact" in columns:
        op.drop_column("user", "contact")
    if "organization" in columns:
        op.drop_column("user", "organization")
