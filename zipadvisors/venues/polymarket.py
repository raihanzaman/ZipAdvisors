"""Polymarket Gamma + CLOB public APIs."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from zipadvisors.config import POLY_CLOB, POLY_GAMMA, polymarket_slug
from zipadvisors.http import as_float, as_prob, fetch_json, parse_json_field


def _utc_from_unix(ts: int) -> datetime:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc)


def list_markets(event: dict) -> list[dict]:
    data = fetch_json(f"{POLY_GAMMA}/events?slug={event['polymarket_slug']}")
    payload = data[0] if isinstance(data, list) and data else data
    rows = []
    for market in payload.get("markets") or []:
        title = market.get("groupItemTitle") or ""
        by_title = event.get("poly_by_title") or {}
        slug = by_title.get(title.strip().lower()) or polymarket_slug(title)
        if not slug or slug not in event["markets"]:
            continue
        prices = parse_json_field(market.get("outcomePrices")) or []
        yes = as_prob(prices[0] if len(prices) > 0 else None)
        no = as_prob(prices[1] if len(prices) > 1 else None)
        yes_bid = as_prob(market.get("bestBid"))
        yes_ask = as_prob(market.get("bestAsk"))
        if yes is None:
            yes = yes_bid or yes_ask
        if yes is None:
            continue
        if no is None:
            no = max(0.0, min(1.0, 1.0 - yes))
        if yes_bid is None:
            yes_bid = yes
        if yes_ask is None:
            yes_ask = yes
        no_bid = as_prob(1.0 - yes_ask)
        no_ask = as_prob(1.0 - yes_bid)
        tokens = parse_json_field(market.get("clobTokenIds")) or []
        book = as_float(market.get("bestBid")) is not None and as_float(market.get("bestAsk")) is not None
        rows.append(
            {
                "venue": "polymarket",
                "slug": slug,
                "title": title,
                "token": tokens[0] if tokens else None,
                "no_token": tokens[1] if len(tokens) > 1 else None,
                "yes_mid": yes,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "no_bid": no_bid,
                "no_ask": no_ask,
                "no_mid": no,
                "volume": as_float(market.get("volume")),
                "liquidity": as_float(market.get("liquidity")),
                "quality": "book" if book else "last",
            }
        )
    return rows


def history(token: str) -> pd.DataFrame:
    url = f"{POLY_CLOB}/prices-history?market={token}&interval=max&fidelity=60"
    try:
        data = fetch_json(url)
    except RuntimeError:
        data = fetch_json(f"{POLY_CLOB}/prices-history?market={token}&interval=1w&fidelity=60")
    rows = []
    for point in data.get("history") or []:
        yes = as_prob(point.get("p"))
        ts = point.get("t")
        if yes is None or ts is None:
            continue
        rows.append(
            {
                "timestamp": _utc_from_unix(int(ts)),
                "yes_price": yes,
                "no_price": max(0.0, min(1.0, 1.0 - yes)),
                "volume": None,
            }
        )
    return pd.DataFrame(rows)
