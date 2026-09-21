"""Poll Polymarket Gamma + CLOB APIs and write ticks.

Default event is MLB World Series 2026. Selenium is not used.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

from db import delete_ticks, ensure_schema, insert_tick, insert_ticks, is_postgres, tick_count
from markets import (
    DEFAULT_EVENT_ID,
    POLY_CLOB,
    POLY_GAMMA,
    as_float,
    event_from_url,
    fetch_json,
    get_event,
    parse_json_field,
    polymarket_slug,
)

POLL_SECONDS = 300
HISTORY_DAYS = 14


def _utc_from_unix(ts: int) -> datetime:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc)


def list_polymarket_markets(event: dict) -> list[dict]:
    data = fetch_json(f"{POLY_GAMMA}/events?slug={event['polymarket_slug']}")
    payload = data[0] if isinstance(data, list) and data else data
    rows = []
    for market in payload.get("markets") or []:
        title = market.get("groupItemTitle") or ""
        slug = polymarket_slug(title)
        if not slug or slug not in event["markets"]:
            continue
        prices = parse_json_field(market.get("outcomePrices")) or []
        yes = as_float(prices[0] if len(prices) > 0 else None)
        no = as_float(prices[1] if len(prices) > 1 else None)
        if yes is None:
            continue
        if no is None:
            no = max(0.0, min(1.0, 1.0 - yes))
        tokens = parse_json_field(market.get("clobTokenIds")) or []
        yes_token = tokens[0] if tokens else None
        rows.append(
            {
                "slug": slug,
                "title": title,
                "yes": yes,
                "no": no,
                "volume": as_float(market.get("volume")),
                "token": yes_token,
            }
        )
    return rows


def backfill(event: dict) -> int:
    rows = []
    markets = list_polymarket_markets(event)
    print(f"Polymarket backfill: {len(markets)} markets, CLOB hourly history")
    for i, market in enumerate(markets, start=1):
        if not market["token"]:
            print(f"  skip {market['slug']}: no CLOB token")
            continue
        url = f"{POLY_CLOB}/prices-history?market={market['token']}&interval=1w&fidelity=60"
        try:
            data = fetch_json(url)
        except Exception as exc:
            print(f"  skip {market['slug']}: {exc}")
            continue
        history = data.get("history") or []
        for point in history:
            yes = as_float(point.get("p"))
            ts = point.get("t")
            if yes is None or ts is None:
                continue
            rows.append(
                {
                    "venue": "polymarket",
                    "event_id": event["id"],
                    "market_name": market["slug"],
                    "yes_price": yes,
                    "no_price": max(0.0, min(1.0, 1.0 - yes)),
                    "trading_volume": None,
                    "ts": _utc_from_unix(int(ts)),
                }
            )
        print(f"  [{i}/{len(markets)}] {market['slug']} points={len(history)}")
        time.sleep(0.12)
    delete_ticks(venue="polymarket", event_id=event["id"])
    n = insert_ticks(rows)
    print(f"Polymarket wrote {n} historical ticks")
    return n


def snapshot(event: dict) -> int:
    now = datetime.now(timezone.utc)
    count = 0
    for market in list_polymarket_markets(event):
        insert_tick(
            "polymarket",
            event["id"],
            market["slug"],
            market["yes"],
            market["no"],
            market["volume"],
            ts=now,
        )
        count += 1
        print(
            f"poly  {market['slug']:24} yes={market['yes']:.4f} "
            f"vol={market['volume'] if market['volume'] is not None else 'n/a'}"
        )
    return count


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket API poller")
    parser.add_argument("url", nargs="?", help="Polymarket event URL (optional)")
    parser.add_argument("iterations", nargs="?", type=int, default=0, help="0 = run until stopped")
    parser.add_argument("--event", default=DEFAULT_EVENT_ID)
    parser.add_argument("--interval", type=int, default=POLL_SECONDS)
    parser.add_argument("--skip-backfill", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.url:
        args.event = event_from_url(args.url)["id"]
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    event = get_event(args.event)
    ensure_schema()
    print(f"Polymarket scraper -> {event['label']} ({event['polymarket_slug']})")
    print(f"DB: {'Supabase Postgres' if is_postgres() else 'SQLite'}  ticks before: {tick_count()}")
    if not args.skip_backfill:
        backfill(event)
    snapshot(event)
    if args.once:
        print(f"Done. ticks={tick_count(venue='polymarket', event_id=event['id'])}")
        return
    iterations = args.iterations if args.iterations > 0 else 10**9
    n = 1
    while n < iterations:
        time.sleep(max(5, args.interval))
        try:
            snapshot(event)
        except Exception as exc:
            print(f"Polymarket snapshot failed: {exc}")
        n += 1
        print(f"iteration {n}/{'inf' if not args.iterations else iterations}")


if __name__ == "__main__":
    main()
