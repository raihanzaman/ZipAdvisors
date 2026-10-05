"""Allowlisted events and canonical market slugs."""

from __future__ import annotations

import re
from typing import Any

USER_AGENT = "ZipAdvisors/2.0 (classroom project)"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
POLY_GAMMA = "https://gamma-api.polymarket.com"
POLY_CLOB = "https://clob.polymarket.com"

QUOTE_TTL_SECONDS = 45
HISTORY_TTL_SECONDS = 300
HISTORY_DAYS = 14

# Liquid names used when training the spread-convergence model.
MLB_FOCUS = (
    "los_angeles_dodgers",
    "milwaukee_brewers",
    "new_york_yankees",
    "tampa_bay_rays",
    "philadelphia_phillies",
    "boston_red_sox",
)

# Kalshi ticker suffix → (canonical slug, display label)
MLB_TEAMS: dict[str, tuple[str, str]] = {
    "ATH": ("athletics", "Athletics"),
    "ATL": ("atlanta_braves", "Atlanta Braves"),
    "AZ": ("arizona_diamondbacks", "Arizona Diamondbacks"),
    "BAL": ("baltimore_orioles", "Baltimore Orioles"),
    "BOS": ("boston_red_sox", "Boston Red Sox"),
    "CHC": ("chicago_cubs", "Chicago Cubs"),
    "CIN": ("cincinnati_reds", "Cincinnati Reds"),
    "CLE": ("cleveland_guardians", "Cleveland Guardians"),
    "COL": ("colorado_rockies", "Colorado Rockies"),
    "CWS": ("chicago_white_sox", "Chicago White Sox"),
    "DET": ("detroit_tigers", "Detroit Tigers"),
    "HOU": ("houston_astros", "Houston Astros"),
    "KC": ("kansas_city_royals", "Kansas City Royals"),
    "LAA": ("los_angeles_angels", "Los Angeles Angels"),
    "LAD": ("los_angeles_dodgers", "Los Angeles Dodgers"),
    "MIA": ("miami_marlins", "Miami Marlins"),
    "MIL": ("milwaukee_brewers", "Milwaukee Brewers"),
    "MIN": ("minnesota_twins", "Minnesota Twins"),
    "NYM": ("new_york_mets", "New York Mets"),
    "NYY": ("new_york_yankees", "New York Yankees"),
    "PHI": ("philadelphia_phillies", "Philadelphia Phillies"),
    "PIT": ("pittsburgh_pirates", "Pittsburgh Pirates"),
    "SD": ("san_diego_padres", "San Diego Padres"),
    "SEA": ("seattle_mariners", "Seattle Mariners"),
    "SF": ("san_francisco_giants", "San Francisco Giants"),
    "STL": ("st_louis_cardinals", "St. Louis Cardinals"),
    "TB": ("tampa_bay_rays", "Tampa Bay Rays"),
    "TEX": ("texas_rangers", "Texas Rangers"),
    "TOR": ("toronto_blue_jays", "Toronto Blue Jays"),
    "WSH": ("washington_nationals", "Washington Nationals"),
}

POLY_TITLE_ALIASES = {label.lower(): slug for slug, label in MLB_TEAMS.values()}
POLY_TITLE_ALIASES.update(
    {
        "athletics": "athletics",
        "oakland athletics": "athletics",
        "a's": "athletics",
        "st. louis cardinals": "st_louis_cardinals",
        "st louis cardinals": "st_louis_cardinals",
    }
)

TRACKED_EVENTS: dict[str, dict[str, Any]] = {
    "mlb_world_series_2026": {
        "id": "mlb_world_series_2026",
        "label": "MLB World Series Champion 2026",
        "kalshi_event": "KXMLB-26",
        "kalshi_series": "KXMLB",
        "kalshi_url": "https://kalshi.com/markets/kxmlb/world-series/kxmlb-26",
        "polymarket_slug": "mlb-world-series-champion-2026",
        "polymarket_url": "https://polymarket.com/event/mlb-world-series-champion-2026",
        "focus_markets": list(MLB_FOCUS),
        "markets": {slug: label for slug, label in MLB_TEAMS.values()},
    }
}

DEFAULT_EVENT_ID = "mlb_world_series_2026"

CUSTOM_EVENTS: dict[str, dict[str, Any]] = {}


def slugify_label(text: str) -> str:
    value = (text or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value[:80]


def register_event(event: dict[str, Any]) -> dict[str, Any]:
    CUSTOM_EVENTS[event["id"]] = event
    return event


def get_event(event_id: str | None = None) -> dict[str, Any]:
    key = event_id or DEFAULT_EVENT_ID
    event = TRACKED_EVENTS.get(key) or CUSTOM_EVENTS.get(key)
    if event is None:
        raise ValueError(f"Event {key!r} is not loaded. Paste both venue URLs and click Load pair.")
    return event


def event_catalog() -> list[dict[str, Any]]:
    rows = []
    for event in list(TRACKED_EVENTS.values()) + list(CUSTOM_EVENTS.values()):
        rows.append(
            {
                "id": event["id"],
                "label": event["label"],
                "custom": bool(event.get("custom")),
                "focus_markets": list(event.get("focus_markets") or []),
            }
        )
    return rows


def display_name(event_id: str, market_name: str) -> str:
    event = TRACKED_EVENTS.get(event_id) or CUSTOM_EVENTS.get(event_id) or {}
    return event.get("markets", {}).get(market_name, market_name.replace("_", " ").title())


def kalshi_slug(ticker: str) -> str | None:
    suffix = (ticker or "").rsplit("-", 1)[-1].upper()
    row = MLB_TEAMS.get(suffix)
    return row[0] if row else None


def polymarket_slug(title: str) -> str | None:
    if not title:
        return None
    return POLY_TITLE_ALIASES.get(title.strip().lower())


def is_focus_market(event_id: str, market_name: str) -> bool:
    event = TRACKED_EVENTS.get(event_id) or CUSTOM_EVENTS.get(event_id)
    if event is None:
        return False
    return market_name in set(event.get("focus_markets") or [])
