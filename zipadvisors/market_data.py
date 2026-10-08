"""Cross-venue quotes, history, and arb scan from live APIs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from zipadvisors.arb import attach_arb
from zipadvisors.config import (
    HISTORY_TTL_SECONDS,
    QUOTE_TTL_SECONDS,
    display_name,
    get_event,
    is_focus_market,
)
from zipadvisors.http import CACHE
from zipadvisors.schema import cache_get_json, cache_set_json
from zipadvisors.venues import kalshi, polymarket


def fetch_quotes(event_id: str | None = None) -> dict:
    event = get_event(event_id)
    cache_key = f"quotes:{event['id']}"
    cached = CACHE.get(cache_key)
    if cached is None:
        cached = cache_get_json(cache_key)
        if cached is not None:
            CACHE.set(cache_key, cached, QUOTE_TTL_SECONDS)
    if cached is not None:
        return cached

    with ThreadPoolExecutor(max_workers=2) as pool:
        k_fut = pool.submit(kalshi.list_markets, event)
        p_fut = pool.submit(polymarket.list_markets, event)
        k_rows = {row["slug"]: row for row in k_fut.result()}
        p_rows = {row["slug"]: row for row in p_fut.result()}

    markets = []
    for slug, label in event["markets"].items():
        k = k_rows.get(slug)
        p = p_rows.get(slug)
        if not k and not p:
            continue
        row = {
            "id": slug,
            "label": label,
            "focus": is_focus_market(event["id"], slug),
            "kalshi": k,
            "polymarket": p,
        }
        attach_arb(row)
        markets.append(row)

    markets.sort(key=lambda item: (-(item.get("net_edge") or -1), not item["focus"], item["label"]))
    payload = {
        "event_id": event["id"],
        "event_label": event["label"],
        "markets": markets,
        "source": "live_api",
    }
    CACHE.set(cache_key, payload, QUOTE_TTL_SECONDS)
    cache_set_json(cache_key, payload, QUOTE_TTL_SECONDS)
    return payload


def fetch_history(event_id: str, market_name: str) -> dict:
    event = get_event(event_id)
    if market_name not in event["markets"]:
        raise ValueError(f"Unknown contract {market_name!r}.")
    cache_key = f"history:{event['id']}:{market_name}"
    cached = CACHE.get(cache_key)
    if cached is not None:
        return cached

    quotes = fetch_quotes(event["id"])
    match = next((row for row in quotes["markets"] if row["id"] == market_name), None)
    if match is None:
        raise ValueError(f"No live quotes for {display_name(event['id'], market_name)}.")

    k_meta = match.get("kalshi") or {}
    p_meta = match.get("polymarket") or {}

    def _kalshi_frame() -> pd.DataFrame:
        ticker = k_meta.get("ticker")
        if not ticker:
            return pd.DataFrame()
        return kalshi.history(event, ticker)

    def _poly_frame() -> pd.DataFrame:
        token = p_meta.get("token")
        if not token:
            return pd.DataFrame()
        return polymarket.history(token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        k_fut = pool.submit(_kalshi_frame)
        p_fut = pool.submit(_poly_frame)
        k_df = k_fut.result()
        p_df = p_fut.result()

    payload = {"kalshi": k_df, "polymarket": p_df, "quote": match}
    return CACHE.set(cache_key, payload, HISTORY_TTL_SECONDS)
