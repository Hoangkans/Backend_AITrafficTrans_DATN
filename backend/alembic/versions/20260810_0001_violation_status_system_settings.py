"""add violation status and persistent system settings

Revision ID: 20260810_0001
Revises:
Create Date: 2026-08-10
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260810_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "violations",
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
    )
    op.create_index("ix_violations_status", "violations", ["status"])
    op.execute("UPDATE violations SET status = CASE WHEN is_confirmed THEN 'verified' ELSE 'pending' END")
    op.alter_column("violations", "status", server_default=None)

    op.create_table(
        "system_settings",
        sa.Column("id", sa.String(length=50), nullable=False),
        sa.Column("app_name", sa.String(length=255), nullable=False),
        sa.Column("detection_threshold", sa.Float(), nullable=False),
        sa.Column("max_cameras", sa.Integer(), nullable=False),
        sa.Column("retention_days", sa.Integer(), nullable=False),
        sa.Column("notification_sound_enabled", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("email_alerts_enabled", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("daily_report_time", sa.String(length=5), nullable=False, server_default="23:59"),
        sa.Column("two_factor_enabled", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("system_settings")
    op.drop_index("ix_violations_status", table_name="violations")
    op.drop_column("violations", "status")
