"""Database access for ZipAdvisors.

Primary store is a single `ticks` table on Supabase (Postgres). Set
DATABASE_URL to the project's connection URI. If that is unset, the bundled
SQLite file is used so local clones still run.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.pool import NullPool

load_dotenv()

ROOT = Path(__file__).resolve().parent
DEFAULT_SQLITE = ROOT / "data" / "markets.db"
TICKS_TABLE = "ticks"


def _sqlite_url(path: Path) -> str:
    return "sqlite:///" + path.resolve().as_posix()


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
        # Windows libpq otherwise spends ~30s trying GSS/SSPI before TLS.
        url = _ensure_query_param(url, "sslmode", "require")
        url = _ensure_query_param(url, "gssencmode", "disable")
    return url


def database_url() -> str:
    explicit = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
    if explicit:
        return _normalize_database_url(explicit)
    return _sqlite_url(DEFAULT_SQLITE)


def is_postgres() -> bool:
    engine = get_engine()
    return engine.dialect.name in {"postgresql", "postgres"}


def _sqlite_engine() -> Engine:
    DEFAULT_SQLITE.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        _sqlite_url(DEFAULT_SQLITE),
        pool_pre_ping=True,
        connect_args={"check_same_thread": False, "timeout": 30},
    )

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=8000")
        cursor.close()

    return engine


def _with_pg_params(url: str) -> str:
    url = _ensure_query_param(url, "sslmode", "require")
    return _ensure_query_param(url, "gssencmode", "disable")


def _candidate_urls(url: str) -> list[str]:
    urls = [_with_pg_params(url)]
    try:
        parsed = make_url(url)
    except Exception:
        return urls
    host = parsed.host or ""
    if host.endswith(".pooler.supabase.com") and parsed.port == 6543:
        urls.append(_with_pg_params(parsed.set(port=5432).render_as_string(hide_password=False)))
    if host.startswith("db.") and host.endswith(".supabase.co"):
        ref = host.split(".")[1]
        for pool_host, port in (
            ("aws-0-us-east-2.pooler.supabase.com", 6543),
            ("aws-0-us-east-2.pooler.supabase.com", 5432),
            ("aws-0-us-east-1.pooler.supabase.com", 6543),
            ("aws-0-us-east-1.pooler.supabase.com", 5432),
            ("aws-1-us-east-1.pooler.supabase.com", 6543),
        ):
            candidate = parsed.set(host=pool_host, port=port, username=f"postgres.{ref}")
            urls.append(_with_pg_params(candidate.render_as_string(hide_password=False)))
    return list(dict.fromkeys(urls))


def _connect_postgres(url: str) -> Engine:
    last_error: Exception | None = None
    for candidate in _candidate_urls(url):
        engine = create_engine(
            candidate,
            pool_pre_ping=True,
            poolclass=NullPool,
            connect_args={"connect_timeout": 10},
        )
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine
        except Exception as exc:
            last_error = exc
            engine.dispose()
    assert last_error is not None
    raise last_error


def _safe_engine_url(engine: Engine) -> str:
    return engine.url.render_as_string(hide_password=True)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    url = database_url()
    if url.startswith("sqlite"):
        print(f"DATABASE_URL unset. Using local SQLite ({DEFAULT_SQLITE.name}).")
        return _sqlite_engine()
    try:
        engine = _connect_postgres(url)
    except Exception as exc:
        raise RuntimeError(
            "DATABASE_URL is set but Supabase Postgres is unreachable "
            f"({exc.__class__.__name__}: {exc}). "
            "Use the Transaction or Session pooler URI from "
            "Project Settings → Database → Connect (not db.*.supabase.co)."
        ) from exc
    print(f"Using Supabase Postgres ({_safe_engine_url(engine)})")
    return engine


def event_key(venue: str, event_id: str) -> str:
    prefix = "K_" if venue == "kalshi" else "P_"
    return f"{prefix}{event_id}"


def split_event_key(event_key_name: str) -> tuple[str, str]:
    if event_key_name.startswith("K_"):
        return "kalshi", event_key_name[2:]
    if event_key_name.startswith("P_"):
        return "polymarket", event_key_name[2:]
    raise ValueError(f"Invalid event key: {event_key_name!r}")


def paired_table_name(event_key_name: str) -> str:
    venue, event_id = split_event_key(event_key_name)
    other = "polymarket" if venue == "kalshi" else "kalshi"
    return event_key(other, event_id)


def ticks_ddl() -> str:
    if is_postgres():
        return f"""
        CREATE TABLE IF NOT EXISTS {TICKS_TABLE} (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            venue TEXT NOT NULL CHECK (venue IN ('kalshi', 'polymarket')),
            event_id TEXT NOT NULL,
            market_name TEXT NOT NULL,
            yes_price DOUBLE PRECISION NOT NULL,
            no_price DOUBLE PRECISION NOT NULL,
            trading_volume DOUBLE PRECISION,
            ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_ticks_lookup
            ON {TICKS_TABLE} (venue, event_id, market_name, ts);
        """
    return f"""
        CREATE TABLE IF NOT EXISTS {TICKS_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            venue TEXT NOT NULL,
            event_id TEXT NOT NULL,
            market_name TEXT NOT NULL,
            yes_price REAL NOT NULL,
            no_price REAL NOT NULL,
            trading_volume REAL,
            ts TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ticks_lookup
            ON {TICKS_TABLE} (venue, event_id, market_name, ts);
        """


def ensure_schema() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        for statement in ticks_ddl().split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))


def ticks_ready() -> bool:
    try:
        return TICKS_TABLE in inspect(get_engine()).get_table_names()
    except Exception:
        return False


def list_tables() -> list[str]:
    return list_event_keys("K_") + list_event_keys("P_")


def list_event_keys(prefix: str) -> list[str]:
    venue = "kalshi" if prefix == "K_" else "polymarket"
    if not ticks_ready():
        return []
    query = text(
        f"SELECT DISTINCT event_id FROM {TICKS_TABLE} "
        "WHERE venue = :venue ORDER BY event_id"
    )
    df = pd.read_sql(query, get_engine(), params={"venue": venue})
    return [event_key(venue, eid) for eid in df["event_id"].dropna().astype(str).tolist()]


def list_market_names(event_key_name: str) -> list[str]:
    venue, event_id = split_event_key(event_key_name)
    query = text(
        f"SELECT DISTINCT market_name FROM {TICKS_TABLE} "
        "WHERE venue = :venue AND event_id = :event_id AND market_name IS NOT NULL "
        "ORDER BY market_name"
    )
    df = pd.read_sql(query, get_engine(), params={"venue": venue, "event_id": event_id})
    return df["market_name"].dropna().astype(str).tolist()


def _to_utc_naive(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, format="ISO8601")
    return parsed.dt.tz_convert("UTC")


def fetch_market_frame(event_key_name: str, market_name: str, limit: int | None = None) -> pd.DataFrame:
    venue, event_id = split_event_key(event_key_name)
    sql = (
        f"SELECT yes_price, no_price, trading_volume, ts AS timestamp FROM {TICKS_TABLE} "
        "WHERE venue = :venue AND event_id = :event_id AND market_name = :market_name "
        "ORDER BY ts"
    )
    params: dict = {"venue": venue, "event_id": event_id, "market_name": market_name}
    if limit is not None:
        sql += " LIMIT :limit"
        params["limit"] = int(limit)
    df = pd.read_sql(text(sql), get_engine(), params=params)
    if df.empty:
        return df
    df["timestamp"] = _to_utc_naive(df["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def fetch_latest_volume(event_key_name: str, market_name: str) -> float | None:
    venue, event_id = split_event_key(event_key_name)
    query = text(
        f"SELECT trading_volume FROM {TICKS_TABLE} "
        "WHERE venue = :venue AND event_id = :event_id AND market_name = :market_name "
        "AND trading_volume IS NOT NULL ORDER BY ts DESC LIMIT 1"
    )
    df = pd.read_sql(
        query,
        get_engine(),
        params={"venue": venue, "event_id": event_id, "market_name": market_name},
    )
    if df.empty:
        return None
    try:
        return float(df.iloc[0]["trading_volume"])
    except (TypeError, ValueError):
        return None


def _normalize_venue(venue: str) -> str:
    return "kalshi" if venue.lower().startswith("k") else "polymarket"


def _as_utc(ts: datetime | None) -> datetime:
    if ts is None:
        return datetime.now(timezone.utc)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def insert_tick(
    venue: str,
    event_id: str,
    market_name: str,
    yes_price: float,
    no_price: float,
    trading_volume: float | None = None,
    ts: datetime | None = None,
) -> None:
    insert_ticks(
        [
            {
                "venue": _normalize_venue(venue),
                "event_id": event_id,
                "market_name": market_name,
                "yes_price": yes_price,
                "no_price": no_price,
                "trading_volume": trading_volume,
                "ts": _as_utc(ts),
            }
        ]
    )


def insert_ticks(rows: list[dict]) -> int:
    if not rows:
        return 0
    payload = []
    for row in rows:
        payload.append(
            {
                "venue": _normalize_venue(str(row["venue"])),
                "event_id": row["event_id"],
                "market_name": row["market_name"],
                "yes_price": float(row["yes_price"]),
                "no_price": float(row["no_price"]),
                "trading_volume": row.get("trading_volume"),
                "ts": _as_utc(row.get("ts")),
            }
        )
    query = text(
        f"INSERT INTO {TICKS_TABLE} "
        "(venue, event_id, market_name, yes_price, no_price, trading_volume, ts) "
        "VALUES (:venue, :event_id, :market_name, :yes_price, :no_price, :trading_volume, :ts)"
    )
    with get_engine().begin() as conn:
        conn.execute(query, payload)
    return len(payload)


def delete_ticks(venue: str | None = None, event_id: str | None = None) -> int:
    ensure_schema()
    clauses = []
    params: dict = {}
    if venue:
        clauses.append("venue = :venue")
        params["venue"] = _normalize_venue(venue)
    if event_id:
        clauses.append("event_id = :event_id")
        params["event_id"] = event_id
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with get_engine().begin() as conn:
        result = conn.execute(text(f"DELETE FROM {TICKS_TABLE}{where}"), params)
        return int(result.rowcount or 0)


def tick_count(venue: str | None = None, event_id: str | None = None) -> int:
    if not ticks_ready():
        return 0
    clauses = []
    params: dict = {}
    if venue:
        clauses.append("venue = :venue")
        params["venue"] = _normalize_venue(venue)
    if event_id:
        clauses.append("event_id = :event_id")
        params["event_id"] = event_id
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with get_engine().connect() as conn:
        return int(conn.execute(text(f"SELECT COUNT(*) FROM {TICKS_TABLE}{where}"), params).scalar() or 0)
