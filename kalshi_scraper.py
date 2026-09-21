"""Poll Kalshi's public Trade API and write ticks.

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
    KALSHI_API,
    as_float,
    event_from_url,
    fetch_json,
    get_event,
    kalshi_slug,
)

POLL_SECONDS = 300
HISTORY_DAYS = 14


def _utc_from_unix(ts: int) -> datetime:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc)


def list_kalshi_markets(event: dict) -> list[dict]:
    data = fetch_json(f"{KALSHI_API}/markets?event_ticker={event['kalshi_event']}&limit=200")
    rows = []
    for market in data.get("markets") or []:
        ticker = market.get("ticker") or ""
        slug = kalshi_slug(ticker)
        if not slug or slug not in event["markets"]:
            continue
        yes = as_float(market.get("last_price_dollars"))
        if yes is None:
            yes = as_float(market.get("yes_bid_dollars"))
        if yes is None:
            continue
        no = as_float(market.get("no_bid_dollars"))
        if no is None:
            no = max(0.0, min(1.0, 1.0 - yes))
        rows.append(
            {
                "ticker": ticker,
                "slug": slug,
                "yes": yes,
                "no": no,
                "volume": as_float(market.get("volume_fp")),
            }
        )
    return rows


def backfill(event: dict) -> int:
    end = int(time.time())
    start = end - HISTORY_DAYS * 24 * 3600
    series = event["kalshi_series"]
    rows = []
    markets = list_kalshi_markets(event)
    print(f"Kalshi backfill: {len(markets)} markets, {HISTORY_DAYS}d hourly candles")
    for i, market in enumerate(markets, start=1):
        url = (
            f"{KALSHI_API}/series/{series}/markets/{market['ticker']}/candlesticks"
            f"?start_ts={start}&end_ts={end}&period_interval=60"
        )
        try:
            data = fetch_json(url)
        except Exception as exc:
            print(f"  skip {market['ticker']}: {exc}")
            continue
        for candle in data.get("candlesticks") or []:
            price = candle.get("price") or {}
            yes = as_float(price.get("close_dollars"))
            if yes is None:
                continue
            rows.append(
                {
                    "venue": "kalshi",
                    "event_id": event["id"],
                    "market_name": market["slug"],
                    "yes_price": yes,
                    "no_price": max(0.0, min(1.0, 1.0 - yes)),
                    "trading_volume": as_float(candle.get("volume_fp")),
                    "ts": _utc_from_unix(candle["end_period_ts"]),
                }
            )
        print(f"  [{i}/{len(markets)}] {market['slug']} candles={len(data.get('candlesticks') or [])}")
        time.sleep(0.12)
    delete_ticks(venue="kalshi", event_id=event["id"])
    n = insert_ticks(rows)
    print(f"Kalshi wrote {n} historical ticks")
    return n


def snapshot(event: dict) -> int:
    now = datetime.now(timezone.utc)
    count = 0
    for market in list_kalshi_markets(event):
        insert_tick(
            "kalshi",
            event["id"],
            market["slug"],
            market["yes"],
            market["no"],
            market["volume"],
            ts=now,
        )
        count += 1
        print(
            f"kalshi {market['slug']:24} yes={market['yes']:.4f} "
            f"vol={market['volume'] if market['volume'] is not None else 'n/a'}"
        )
    return count


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kalshi API poller")
    parser.add_argument("url", nargs="?", help="Kalshi event URL (optional)")
    parser.add_argument("iterations", nargs="?", type=int, default=0, help="0 = run until stopped")
    parser.add_argument("--event", default=DEFAULT_EVENT_ID)
    parser.add_argument("--interval", type=int, default=POLL_SECONDS)
    parser.add_argument("--skip-backfill", action="store_true")
    parser.add_argument("--once", action="store_true", help="Backfill + one snapshot, then exit")
    args = parser.parse_args(argv)
    if args.url:
        args.event = event_from_url(args.url)["id"]
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    event = get_event(args.event)
    ensure_schema()
    print(f"Kalshi scraper -> {event['label']} ({event['kalshi_event']})")
    print(f"DB: {'Supabase Postgres' if is_postgres() else 'SQLite'}  ticks before: {tick_count()}")
    if not args.skip_backfill:
        backfill(event)
    snapshot(event)
    if args.once:
        print(f"Done. ticks={tick_count(venue='kalshi', event_id=event['id'])}")
        return
    iterations = args.iterations if args.iterations > 0 else 10**9
    n = 1
    while n < iterations:
        time.sleep(max(5, args.interval))
        try:
            snapshot(event)
        except Exception as exc:
            print(f"Kalshi snapshot failed: {exc}")
        n += 1
        print(f"iteration {n}/{'inf' if not args.iterations else iterations}")


if __name__ == "__main__":
    main()
