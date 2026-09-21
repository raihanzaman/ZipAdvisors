"""Optional synthetic ticks. Live scrapers are the default data source."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import inspect, text

from db import DEFAULT_SQLITE, TICKS_TABLE, delete_ticks, ensure_schema, get_engine, is_postgres

N_STEPS = 720
BAR_SECONDS = 20
START = datetime(2026, 8, 1, 12, 0, 0)

EVENTS = {
    "nba_championship": {
        "oklahoma_city_thunder": 0.38,
        "boston_celtics": 0.27,
        "new_york_knicks": 0.18,
        "los_angeles_lakers": 0.12,
    },
    "fed_funds": {
        "rate_cut": 0.46,
        "rate_hold": 0.41,
        "rate_hike": 0.09,
    },
}


def _simulate_pair(n: int, start_p: float, rng: np.random.Generator, fast_venue: str = "kalshi"):
    true = np.empty(n)
    fast = np.empty(n)
    slow = np.empty(n)
    true[0] = start_p
    fast[0] = np.clip(start_p + rng.normal(0, 0.01), 0.06, 0.94)
    slow[0] = np.clip(start_p + rng.normal(0, 0.02), 0.06, 0.94)

    for t in range(1, n):
        shock = rng.normal(0, 0.01)
        if rng.random() < 0.05:
            shock += rng.choice([-1.0, 1.0]) * rng.uniform(0.035, 0.08)
        true[t] = np.clip(true[t - 1] + shock, 0.08, 0.92)
        fast[t] = np.clip(fast[t - 1] + 0.58 * (true[t] - fast[t - 1]) + rng.normal(0, 0.004), 0.05, 0.95)
        slow[t] = np.clip(slow[t - 1] + 0.16 * (fast[t] - slow[t - 1]) + rng.normal(0, 0.006), 0.05, 0.95)

    fast_no = np.clip(1.0 - fast + rng.normal(0.012, 0.006, n), 0.05, 0.95)
    slow_no = np.clip(1.0 - slow + rng.normal(0.018, 0.008, n), 0.05, 0.95)
    if fast_venue == "kalshi":
        return fast, fast_no, slow, slow_no
    return slow, slow_no, fast, fast_no


def _timestamps(n: int) -> list[datetime]:
    return [START + timedelta(seconds=BAR_SECONDS * i) for i in range(n)]


def build_ticks(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    stamps = _timestamps(N_STEPS)

    for event_id, markets in EVENTS.items():
        for market, start_p in markets.items():
            fast_venue = "kalshi" if sum(ord(c) for c in market) % 2 == 0 else "polymarket"
            k_yes, k_no, p_yes, p_no = _simulate_pair(N_STEPS, start_p, rng, fast_venue=fast_venue)
            vol = np.cumsum(np.abs(rng.normal(250, 80, N_STEPS))) + rng.uniform(5e5, 2e6)
            for i in range(N_STEPS):
                rows.append(
                    {
                        "venue": "kalshi",
                        "event_id": event_id,
                        "market_name": market,
                        "yes_price": round(float(k_yes[i]), 4),
                        "no_price": round(float(k_no[i]), 4),
                        "trading_volume": None,
                        "ts": stamps[i],
                    }
                )
                rows.append(
                    {
                        "venue": "polymarket",
                        "event_id": event_id,
                        "market_name": market,
                        "yes_price": round(float(p_yes[i]), 4),
                        "no_price": round(float(p_no[i]), 4),
                        "trading_volume": round(float(vol[i]), 2),
                        "ts": stamps[i],
                    }
                )
    return pd.DataFrame(rows)


def seed_database(force: bool = False, db_path: Path | None = None) -> str:
    if not is_postgres():
        path = Path(db_path) if db_path else DEFAULT_SQLITE
        path.parent.mkdir(parents=True, exist_ok=True)

    get_engine.cache_clear()
    ensure_schema()
    engine = get_engine()

    with engine.begin() as conn:
        inspector = inspect(engine)
        for old in inspector.get_table_names():
            if old.startswith("K_") or old.startswith("P_"):
                conn.execute(text(f'DROP TABLE IF EXISTS "{old}"'))
        existing = int(conn.execute(text(f"SELECT COUNT(*) FROM {TICKS_TABLE}")).scalar() or 0)
        if existing and not force:
            return f"{engine.url.render_as_string(hide_password=True)} ({existing} rows)"
        conn.execute(text(f"DELETE FROM {TICKS_TABLE}"))

    frame = build_ticks()
    frame.to_sql(
        TICKS_TABLE,
        engine,
        if_exists="append",
        index=False,
        chunksize=500,
        method="multi",
    )
    get_engine.cache_clear()
    return engine.url.render_as_string(hide_password=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Seed or clear ZipAdvisors ticks.")
    parser.add_argument("--force", action="store_true", help="Wipe ticks and reload synthetic demo data.")
    parser.add_argument("--clear", action="store_true", help="Delete all ticks. Do not insert demo rows.")
    args = parser.parse_args()
    ensure_schema()
    if args.clear:
        n = delete_ticks()
        print(f"Cleared {n} ticks from {get_engine().url.render_as_string(hide_password=True)}")
        return
    target = seed_database(force=args.force)
    engine = get_engine()
    names = inspect(engine).get_table_names()
    with engine.connect() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) FROM {TICKS_TABLE}")).scalar()
    print(f"Demo ticks ready ({n} rows) at {target}")
    print(f"Tables: {names}")


if __name__ == "__main__":
    main()
