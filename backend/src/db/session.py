"""
src/db/session.py
Database engine and session factory.
Supports both PostgreSQL (Docker) and SQLite (local dev).
"""
import os
import logging
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from src.db.models import Base

logger = logging.getLogger("medsight")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./medsight.db"  # default to SQLite for local dev
)

# SQLite needs check_same_thread=False
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

# An in-memory SQLite DB is per-connection by default, so every new connection would
# see an empty database. StaticPool shares one connection (used by the test-suite).
_engine_kwargs = {"poolclass": StaticPool} if DATABASE_URL in ("sqlite://", "sqlite:///:memory:") else {}
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args, **_engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """
    Create any tables that don't exist yet, then reconcile columns on
    tables that already exist on disk.

    Base.metadata.create_all() alone only creates whole tables that are
    missing — it does nothing for a table that already exists but is
    missing a column a model gained later (e.g. AgentRun.caller_fingerprint
    added after some local .db files already existed). That used to only
    surface as `OperationalError: table X has no column named Y` the first
    time a code path touching the new column actually ran — confusing to
    debug, and with no migration tool wired up (alembic is in
    requirements.txt but unused), the only fix was deleting the .db file.

    _add_missing_columns() below is a lightweight, ADD-COLUMN-ONLY
    auto-migration: it can only add a new NULLABLE column, never drop,
    rename, retype, or backfill one — so it can't lose data. It's a
    stand-in for real migrations, not a replacement for them; anything
    beyond "a model gained a nullable column" (renames, type changes,
    NOT NULL backfills) still needs Alembic set up properly.
    """
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


def _add_missing_columns():
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue  # brand-new table — create_all() just built it correctly
            existing_columns = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing_columns:
                    continue
                if not column.nullable and column.default is None and column.server_default is None:
                    # Can't safely auto-add a NOT NULL column with no
                    # default — existing rows would have nothing to put
                    # there. This needs a real migration (with an explicit
                    # backfill), not a silent guess here.
                    logger.warning(
                        "Schema drift: %s.%s is a new NOT NULL column with no "
                        "default — skipping auto-add. Add it via a real "
                        "migration (or give it a default) instead.",
                        table.name, column.name,
                    )
                    continue
                col_type = column.type.compile(dialect=engine.dialect)
                logger.warning(
                    "Schema drift: auto-adding missing column %s.%s (%s). "
                    "This is a dev convenience, not a substitute for real "
                    "migrations — see create_tables()'s docstring.",
                    table.name, column.name, col_type,
                )
                conn.execute(text(
                    f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {col_type}'
                ))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()