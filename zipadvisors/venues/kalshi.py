"""Kalshi public Trade API."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd

from zipadvisors.config import HISTORY_DAYS, KALSHI_API, kalshi_slug
from zipadvisors.http import as_float, as_prob, fetch_json


def _utc_from_unix(ts: int) -> datetime:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc)


def list_markets(event: dict) -> list[dict]:
    data = fetch_json(f"{KALSHI_API}/markets?event_ticker={event['kalshi_event']}&limit=200")
    rows = []
    for market in data.get("markets") or []:
        ticker = market.get("ticker") or ""
        by_ticker = event.get("kalshi_by_ticker") or {}
        slug = by_ticker.get(ticker) or kalshi_slug(ticker)
        if not slug or slug not in event["markets"]:
            continue
        last = as_prob(market.get("last_price_dollars")) or as_prob(market.get("last_price"))
        yes_bid = as_prob(market.get("yes_bid_dollars")) or as_prob(market.get("yes_bid"))
        yes_ask = as_prob(market.get("yes_ask_dollars")) or as_prob(market.get("yes_ask"))
        no_bid = as_prob(market.get("no_bid_dollars")) or as_prob(market.get("no_bid"))
        no_ask = as_prob(market.get("no_ask_dollars")) or as_prob(market.get("no_ask"))
        yes = last or yes_bid or yes_ask
        if yes is None:
            continue
        if yes_bid is None and last is not None:
            yes_bid = last
        if yes_ask is None and last is not None:
            yes_ask = last
        if no_bid is None:
            no_bid = as_prob(1.0 - (yes_ask if yes_ask is not None else yes))
        if no_ask is None:
            no_ask = as_prob(1.0 - (yes_bid if yes_bid is not None else yes))
        book = all(
            as_float(market.get(key)) is not None
            for key in ("yes_bid_dollars", "yes_ask_dollars")
        ) or all(as_float(market.get(key)) is not None for key in ("yes_bid", "yes_ask"))
        rows.append(
            {
                "venue": "kalshi",
                "slug": slug,
                "ticker": ticker,
                "yes_mid": yes,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "no_bid": no_bid,
                "no_ask": no_ask,
                "yes_ask_size": as_float(market.get("yes_ask_size_fp")),
                "yes_bid_size": as_float(market.get("yes_bid_size_fp")),
                "volume": as_float(market.get("volume_fp")) or as_float(market.get("volume")),
                "open_interest": as_float(market.get("open_interest_fp"))
                or as_float(market.get("open_interest")),
                "quality": "book" if book else "last",
            }
        )
    return rows


def history(event: dict, ticker: str) -> pd.DataFrame:
    end = int(time.time())
    start = end - HISTORY_DAYS * 24 * 3600
    url = (
        f"{KALSHI_API}/series/{event['kalshi_series']}/markets/{ticker}/candlesticks"
        f"?start_ts={start}&end_ts={end}&period_interval=60"
    )
    data = fetch_json(url)
    rows = []
    for candle in data.get("candlesticks") or []:
        price = candle.get("price") or {}
        yes = as_prob(price.get("close_dollars")) or as_prob(price.get("close"))
        ts = candle.get("end_period_ts") or candle.get("end_ts")
        if yes is None or ts is None:
            continue
        rows.append(
            {
                "timestamp": _utc_from_unix(int(ts)),
                "yes_price": yes,
                "no_price": max(0.0, min(1.0, 1.0 - yes)),
                "volume": as_float(candle.get("volume_fp")) or as_float(candle.get("volume")),
            }
        )
    return pd.DataFrame(rows)
