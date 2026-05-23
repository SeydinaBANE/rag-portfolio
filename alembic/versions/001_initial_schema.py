"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-05-23
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("source_uri", sa.Text, nullable=False, unique=True),
        sa.Column("doc_type", sa.String(20), nullable=False),
        sa.Column("title", sa.Text),
        sa.Column(
            "ingested_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("chunk_count", sa.Integer, default=0),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
    )

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(1024)),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("token_count", sa.Integer),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.execute(
        "CREATE INDEX chunks_embedding_idx ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )

    op.create_table(
        "query_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("answer", sa.Text, nullable=False),
        sa.Column("retrieved_chunk_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True))),
        sa.Column("retrieval_scores", postgresql.ARRAY(sa.Float)),
        sa.Column("latency_ms", sa.Integer),
        sa.Column("model_name", sa.String(100)),
        sa.Column("prompt_version", sa.String(20)),
        sa.Column("mlflow_run_id", sa.String(64)),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_table(
        "feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "query_log_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("query_logs.id"),
            nullable=False,
        ),
        sa.Column("rating", sa.SmallInteger, nullable=False),
        sa.Column("comment", sa.Text),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_table(
        "eval_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("mlflow_run_id", sa.String(64)),
        sa.Column("prompt_version", sa.String(20)),
        sa.Column("model_name", sa.String(100)),
        sa.Column("avg_relevancy", sa.Float),
        sa.Column("avg_faithfulness", sa.Float),
        sa.Column("avg_context_prec", sa.Float),
        sa.Column("avg_latency_ms", sa.Float),
        sa.Column("num_questions", sa.Integer),
        sa.Column(
            "ran_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("eval_runs")
    op.drop_table("feedback")
    op.drop_table("query_logs")
    op.drop_index("chunks_embedding_idx", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.execute("DROP EXTENSION IF EXISTS vector")
