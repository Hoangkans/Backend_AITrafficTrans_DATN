from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from app.core.config import settings

import socket
import sqlite3
import json
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, UUID

try:
    sqlite3.register_adapter(list, json.dumps)
    sqlite3.register_adapter(dict, json.dumps)
except Exception:
    pass

# Support PostgreSQL specific types in SQLite fallback for local development/testing
@compiles(JSONB, "sqlite")
def compile_jsonb_sqlite(element, compiler, **kw):
    return "JSON"

@compiles(ARRAY, "sqlite")
def compile_array_sqlite(element, compiler, **kw):
    return "JSON"

@compiles(UUID, "sqlite")
def compile_uuid_sqlite(element, compiler, **kw):
    return "CHAR(36)"

db_url = settings.DATABASE_URL

# Auto-fix '@db:' hostname when running outside Docker container
if "@db:" in db_url:
    try:
        socket.gethostbyname("db")
    except socket.gaierror:
        # Running outside docker: 'db' container host cannot be resolved.
        # Fall back to local SQLite database file for local development outside Docker
        db_url = "sqlite+aiosqlite:///./traffic_monitoring.db"

engine_args = {"echo": False, "future": True}
if "sqlite" in db_url:
    engine_args["connect_args"] = {"check_same_thread": False}

# Create database engine
engine = create_async_engine(db_url, **engine_args)



# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

# Base class for SQLAlchemy Models
Base = declarative_base()

# FastAPI Dependency for Database Sessions
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
