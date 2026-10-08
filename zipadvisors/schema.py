"""Postgres helpers: schema, shared quote cache, and rate limits."""

from __future__ import annotations

import json
import time
from pathlib import Path

from sqlalchemy import text

from zipadvisors.db import get_engine

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "supabase" / "schema.sql"
_memory_limits: dict[str, tuple[int, float]] = {}


def split_sql(sql: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    in_dollar = False
    for line in sql.splitlines(True):
        if line.lstrip().startswith("--") and not in_dollar:
            continue
        count = line.count("$$")
        if count:
            in_dollar = (in_dollar + count) % 2 == 1
        buf.append(line)
        if not in_dollar and line.strip().endswith(";"):
            statement = "".join(buf).strip()
            if statement:
                parts.append(statement)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def ensure_schema() -> None:
    engine = get_engine()
    if engine is None or not SCHEMA_PATH.exists():
        return
    statements = split_sql(SCHEMA_PATH.read_text(encoding="utf-8"))
    raw = engine.raw_connection()
    try:
        raw.autocommit = True
        cursor = raw.cursor()
        for statement in statements:
            try:
                cursor.execute(statement)
            except Exception:
                raw.rollback() if not raw.autocommit else None
                continue
        cursor.close()
    finally:
        raw.close()


def cache_get_json(key: str):
    engine = get_engine()
    if engine is None:
        return None
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT payload FROM quote_cache "
                    "WHERE cache_key = :key AND expires_at > now()"
                ),
                {"key": key},
            ).first()
    except Exception:
        return None
    if row is None:
        return None
    payload = row[0]
    if isinstance(payload, str):
        return json.loads(payload)
    return payload


def cache_set_json(key: str, payload: dict, ttl_seconds: int) -> None:
    engine = get_engine()
    if engine is None:
        return
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO quote_cache (cache_key, payload, expires_at)
                    VALUES (:key, CAST(:payload AS jsonb), now() + make_interval(secs => :ttl))
                    ON CONFLICT (cache_key) DO UPDATE
                    SET payload = EXCLUDED.payload, expires_at = EXCLUDED.expires_at
                    """
                ),
                {"key": key, "payload": json.dumps(payload), "ttl": int(ttl_seconds)},
            )
    except Exception:
        return


def rate_limited(bucket: str, limit: int, window_seconds: int) -> bool:
    """True when this bucket is over the limit (this hit included)."""
    engine = get_engine()
    if engine is None:
        return _memory_limited(bucket, limit, window_seconds)
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    INSERT INTO rate_limits (bucket, hits, window_start)
                    VALUES (:bucket, 1, now())
                    ON CONFLICT (bucket) DO UPDATE SET
                      hits = CASE
                        WHEN rate_limits.window_start < now() - make_interval(secs => :window)
                        THEN 1 ELSE rate_limits.hits + 1 END,
                      window_start = CASE
                        WHEN rate_limits.window_start < now() - make_interval(secs => :window)
                        THEN now() ELSE rate_limits.window_start END
                    RETURNING hits
                    """
                ),
                {"bucket": bucket, "window": int(window_seconds)},
            ).first()
        hits = int(row[0]) if row else 1
        return hits > limit
    except Exception:
        return _memory_limited(bucket, limit, window_seconds)


def _memory_limited(bucket: str, limit: int, window_seconds: int) -> bool:
    now = time.time()
    hits, start = _memory_limits.get(bucket, (0, now))
    if now - start >= window_seconds:
        hits, start = 0, now
    hits += 1
    _memory_limits[bucket] = (hits, start)
    return hits > limit
