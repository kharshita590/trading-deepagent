"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-11
"""

from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "analysis_runs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("query", sa.Text(), nullable=True),
        sa.Column("parameters", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "analysis_results",
        sa.Column("run_id", sa.String(), sa.ForeignKey("analysis_runs.id"), primary_key=True),
        sa.Column("recommendations", sa.JSON(), nullable=False),
        sa.Column("full_result", sa.JSON(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("analysis_results")
    op.drop_table("analysis_runs")
