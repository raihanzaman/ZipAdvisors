"""Paired Kalshi / Polymarket events this app is allowed to chart and train on."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

USER_AGENT = "ZipAdvisors/1.0 (classroom project)"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
POLY_GAMMA = "https://gamma-api.polymarket.com"
POLY_CLOB = "https://clob.polymarket.com"

# Liquid contracts used for XGBoost. Price charts still cover every paired team.
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

POLY_TITLE_ALIASES = {label.lower(): slug for slug, label in (row for row in MLB_TEAMS.values())}
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


def fetch_json(url: str, timeout: int = 45) -> Any:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"HTTP {exc.code} for {url}: {detail}") from exc


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_json_field(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def get_event(event_id: str | None = None) -> dict[str, Any]:
    key = event_id or DEFAULT_EVENT_ID
    event = TRACKED_EVENTS.get(key)
    if event is None:
        raise ValueError(f"Event {key!r} is not in the allowlist.")
    return event


def event_from_url(url: str) -> dict[str, Any]:
    lowered = (url or "").strip().lower()
    for event in TRACKED_EVENTS.values():
        if event["kalshi_url"].lower() in lowered or event["kalshi_event"].lower() in lowered:
            return event
        if event["polymarket_url"].lower() in lowered or event["polymarket_slug"].lower() in lowered:
            return event
    raise ValueError(
        "URL is not a tracked event. Add it to TRACKED_EVENTS in markets.py "
        "or pass one of: "
        + ", ".join(e["kalshi_url"] for e in TRACKED_EVENTS.values())
    )


def event_catalog() -> list[dict[str, Any]]:
    rows = []
    for event in TRACKED_EVENTS.values():
        rows.append(
            {
                "id": event["id"],
                "label": event["label"],
                "kalshi_key": f"K_{event['id']}",
                "polymarket_key": f"P_{event['id']}",
                "focus_markets": list(event["focus_markets"]),
            }
        )
    return rows


def market_catalog(event_id: str | None = None) -> list[dict[str, Any]]:
    event = get_event(event_id)
    focus = set(event["focus_markets"])
    rows = [
        {"id": slug, "label": label, "xgb": slug in focus}
        for slug, label in event["markets"].items()
    ]
    rows.sort(key=lambda row: (not row["xgb"], row["label"]))
    return rows


def display_name(event_id: str, market_name: str) -> str:
    event = TRACKED_EVENTS.get(event_id) or {}
    return event.get("markets", {}).get(market_name, market_name.replace("_", " ").title())


def kalshi_slug(ticker: str) -> str | None:
    suffix = (ticker or "").rsplit("-", 1)[-1].upper()
    row = MLB_TEAMS.get(suffix)
    return row[0] if row else None


def polymarket_slug(title: str) -> str | None:
    if not title:
        return None
    return POLY_TITLE_ALIASES.get(title.strip().lower())


def is_xgb_market(event_id: str, market_name: str) -> bool:
    event = TRACKED_EVENTS.get(event_id)
    if event is None:
        return False
    return market_name in set(event["focus_markets"])
