"""Resolve a Kalshi event URL + Polymarket event URL into a paired board."""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import urlparse

from zipadvisors.config import (
    KALSHI_API,
    MLB_TEAMS,
    POLY_GAMMA,
    POLY_TITLE_ALIASES,
    register_event,
    slugify_label,
)
from zipadvisors.http import CACHE, fetch_json

RESOLVE_TTL = 30 * 60
_STOP = {
    "a",
    "an",
    "the",
    "to",
    "of",
    "and",
    "or",
    "will",
    "win",
    "wins",
    "winner",
    "be",
    "yes",
    "no",
    "for",
    "in",
    "on",
    "at",
}


def _normalize(text: str) -> str:
    value = (text or "").lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    tokens = [tok for tok in value.split() if tok and tok not in _STOP]
    return " ".join(tokens)


def parse_kalshi_event_ticker(url: str) -> str:
    parsed = urlparse((url or "").strip())
    host = (parsed.netloc or "").lower()
    if "kalshi.com" not in host:
        raise ValueError("Kalshi URL must be a kalshi.com event or market link.")
    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        raise ValueError("Could not read an event ticker from the Kalshi URL.")
    return parts[-1].upper()


def parse_polymarket_slug(url: str) -> str:
    parsed = urlparse((url or "").strip())
    host = (parsed.netloc or "").lower()
    if "polymarket.com" not in host:
        raise ValueError("Polymarket URL must be a polymarket.com event link.")
    parts = [p for p in parsed.path.split("/") if p]
    if "event" in parts:
        idx = parts.index("event")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if parts:
        return parts[-1]
    raise ValueError("Could not read an event slug from the Polymarket URL.")


def _kalshi_event(ticker: str) -> dict:
    try:
        data = fetch_json(f"{KALSHI_API}/events/{ticker}")
    except RuntimeError:
        market = fetch_json(f"{KALSHI_API}/markets/{ticker}")
        inner = market.get("market") or market
        event_ticker = inner.get("event_ticker")
        if not event_ticker:
            raise ValueError(f"Kalshi ticker {ticker} is not an event or market.") from None
        data = fetch_json(f"{KALSHI_API}/events/{event_ticker}")
    event = data.get("event") or data
    if not event.get("event_ticker"):
        raise ValueError(f"Kalshi event {ticker} was not found.")
    return event


def _kalshi_markets(event_ticker: str) -> list[dict]:
    rows: list[dict] = []
    cursor = None
    for _ in range(8):
        url = f"{KALSHI_API}/markets?event_ticker={event_ticker}&limit=200"
        if cursor:
            url += f"&cursor={cursor}"
        data = fetch_json(url)
        batch = data.get("markets") or []
        rows.extend(batch)
        cursor = data.get("cursor")
        if not cursor or not batch:
            break
    if not rows:
        data = fetch_json(f"{KALSHI_API}/markets?event_ticker={event_ticker}&limit=200")
        rows = data.get("markets") or []
    return rows


def _poly_event(slug: str) -> dict:
    data = fetch_json(f"{POLY_GAMMA}/events?slug={slug}")
    payload = data[0] if isinstance(data, list) and data else data
    if not payload or not payload.get("markets"):
        raise ValueError(f"Polymarket event {slug!r} was not found or has no markets.")
    return payload


def _canonical_norm(text: str) -> str:
    raw = (text or "").strip().lower()
    slug = POLY_TITLE_ALIASES.get(raw) or POLY_TITLE_ALIASES.get(_normalize(text))
    if slug:
        for team_slug, label in MLB_TEAMS.values():
            if team_slug == slug:
                return _normalize(label)
    return _normalize(text)


def _score(left: str, right: str) -> float:
    variants_a = {_normalize(left), _canonical_norm(left)}
    variants_b = {_normalize(right), _canonical_norm(right)}
    best = 0.0
    for a in variants_a:
        for b in variants_b:
            if not a or not b:
                continue
            if a == b:
                return 1.0
            if a in b or b in a:
                best = max(best, 0.93)
                continue
            sa, sb = set(a.split()), set(b.split())
            if sa and (sa <= sb or sb <= sa):
                best = max(best, 0.88)
                continue
            best = max(best, SequenceMatcher(None, a, b).ratio())
    return best


def _kalshi_label(market: dict) -> str:
    return (
        (market.get("yes_sub_title") or "").strip()
        or (market.get("title") or "").strip()
        or (market.get("ticker") or "").strip()
    )


def _poly_label(market: dict) -> str:
    return (
        (market.get("groupItemTitle") or "").strip()
        or (market.get("question") or "").strip()
        or (market.get("slug") or "").strip()
    )


def _pair_markets(kalshi_markets: list[dict], poly_markets: list[dict]) -> tuple[list[dict], list[str], list[str]]:
    if len(kalshi_markets) == 1 and len(poly_markets) == 1:
        k, p = kalshi_markets[0], poly_markets[0]
        label = _poly_label(p) or _kalshi_label(k) or "Yes"
        return (
            [
                {
                    "slug": slugify_label(label) or "yes",
                    "label": label,
                    "kalshi_ticker": k.get("ticker"),
                    "poly_title": _poly_label(p),
                    "score": 1.0,
                }
            ],
            [],
            [],
        )

    candidates: list[tuple[float, int, int]] = []
    for i, k in enumerate(kalshi_markets):
        for j, p in enumerate(poly_markets):
            if _normalize(_poly_label(p)) in {"other", "field"}:
                continue
            score = _score(_kalshi_label(k), _poly_label(p))
            if score >= 0.72:
                candidates.append((score, i, j))
    candidates.sort(reverse=True)

    used_k: set[int] = set()
    used_p: set[int] = set()
    pairs: list[dict] = []
    used_slugs: set[str] = set()
    for score, i, j in candidates:
        if i in used_k or j in used_p:
            continue
        used_k.add(i)
        used_p.add(j)
        k, p = kalshi_markets[i], poly_markets[j]
        label = _poly_label(p) or _kalshi_label(k)
        slug = slugify_label(label)
        base = slug or f"market_{len(pairs)}"
        slug = base
        n = 2
        while slug in used_slugs:
            slug = f"{base}_{n}"
            n += 1
        used_slugs.add(slug)
        pairs.append(
            {
                "slug": slug,
                "label": label,
                "kalshi_ticker": k.get("ticker"),
                "poly_title": _poly_label(p),
                "score": round(score, 3),
            }
        )

    unmatched_k = [_kalshi_label(kalshi_markets[i]) for i in range(len(kalshi_markets)) if i not in used_k]
    unmatched_p = [_poly_label(poly_markets[j]) for j in range(len(poly_markets)) if j not in used_p]
    return pairs, unmatched_k, unmatched_p


def resolve_pair(kalshi_url: str, polymarket_url: str) -> dict[str, Any]:
    k_url = (kalshi_url or "").strip()
    p_url = (polymarket_url or "").strip()
    if not k_url or not p_url:
        raise ValueError("Paste both a Kalshi event URL and a Polymarket event URL.")

    digest = hashlib.sha1(f"{k_url}|{p_url}".encode()).hexdigest()[:16]
    cache_key = f"resolve:v3:{digest}"
    cached = CACHE.get(cache_key)
    if cached is not None:
        register_event(cached)
        return cached

    k_ticker = parse_kalshi_event_ticker(k_url)
    p_slug = parse_polymarket_slug(p_url)
    k_event = _kalshi_event(k_ticker)
    event_ticker = k_event["event_ticker"]
    k_markets = _kalshi_markets(event_ticker)
    p_event = _poly_event(p_slug)
    p_markets = p_event.get("markets") or []
    pairs, unmatched_k, unmatched_p = _pair_markets(k_markets, p_markets)
    if not pairs:
        raise ValueError(
            "Those events loaded, but no outcomes had matching names. "
            "Use the same event on both venues (for example both World Series winner pages)."
        )

    event_id = "custom_" + digest
    label = p_event.get("title") or k_event.get("title") or event_ticker
    event = {
        "id": event_id,
        "label": label,
        "custom": True,
        "kalshi_event": event_ticker,
        "kalshi_series": k_event.get("series_ticker") or event_ticker.rsplit("-", 1)[0],
        "kalshi_url": k_url,
        "polymarket_slug": p_event.get("slug") or p_slug,
        "polymarket_url": p_url,
        "focus_markets": [row["slug"] for row in pairs[:6]],
        "markets": {row["slug"]: row["label"] for row in pairs},
        "kalshi_by_ticker": {row["kalshi_ticker"]: row["slug"] for row in pairs if row.get("kalshi_ticker")},
        "poly_by_title": {row["poly_title"].strip().lower(): row["slug"] for row in pairs if row.get("poly_title")},
        "unmatched_kalshi": unmatched_k,
        "unmatched_polymarket": unmatched_p,
        "pair_count": len(pairs),
    }
    CACHE.set(cache_key, event, RESOLVE_TTL)
    register_event(event)
    return event


def resolve_summary(event: dict) -> dict:
    return {
        "event_id": event["id"],
        "label": event["label"],
        "pair_count": event.get("pair_count") or len(event.get("markets") or {}),
        "markets": [{"id": slug, "label": label} for slug, label in event["markets"].items()],
        "unmatched_kalshi": event.get("unmatched_kalshi") or [],
        "unmatched_polymarket": event.get("unmatched_polymarket") or [],
        "kalshi_event": event.get("kalshi_event"),
        "polymarket_slug": event.get("polymarket_slug"),
    }
