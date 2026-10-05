"""Optional Supabase / SQLite. The live board does not need a database."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.pool import NullPool

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _ensure_query_param(url: str, key: str, value: str) -> str:
    if f"{key}=" in url:
        return url
    return url + ("&" if "?" in url else "?") + f"{key}={value}"


def _normalize_database_url(url: str) -> str:
    url = url.strip().strip('"').strip("'")
    if url.startswith("postgres://"):
        url = "postgresql+psycopg2://" + url[len("postgres://") :]
    elif url.startswith("postgresql://") and "+psycopg2" not in url and "+psycopg://" not in url:
        url = "postgresql+psycopg2://" + url[len("postgresql://") :]
    if url.startswith("postgresql"):
        url = _ensure_query_param(url, "sslmode", "require")
        url = _ensure_query_param(url, "gssencmode", "disable")
    return url


def configured_database_url() -> str | None:
    raw = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
    if not raw or not raw.strip():
        return None
    return _normalize_database_url(raw)


@lru_cache(maxsize=1)
def get_engine() -> Engine | None:
    url = configured_database_url()
    if not url:
        return None
    try:
        engine = create_engine(
            url,
            pool_pre_ping=True,
            poolclass=NullPool,
            connect_args={"connect_timeout": 10},
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception:
        return None


def is_postgres() -> bool:
    engine = get_engine()
    return bool(engine and engine.dialect.name in {"postgresql", "postgres"})


def database_label() -> str:
    engine = get_engine()
    if engine is None:
        return "none"
    return "supabase" if is_postgres() else engine.dialect.name
