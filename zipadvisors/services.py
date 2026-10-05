"""JSON payloads for the dashboard."""

from __future__ import annotations

import pandas as pd

from zipadvisors.config import display_name, get_event, is_focus_market
from zipadvisors.market_data import fetch_history, fetch_quotes
from zipadvisors.model import predict_convergence, predict_path


def _unix_series(df: pd.DataFrame, column: str) -> list[dict]:
    if df is None or df.empty or column not in df.columns:
        return []
    if isinstance(df.index, pd.DatetimeIndex):
        times = pd.to_datetime(df.index, utc=True)
        values = df[column].to_numpy()
    else:
        ordered = df.sort_values("timestamp")
        times = pd.to_datetime(ordered["timestamp"], utc=True)
        values = ordered[column].to_numpy()
    points = []
    last_t = None
    for ts, value in zip(times, values):
        if value is None or (isinstance(value, float) and value != value):
            continue
        t = int(pd.Timestamp(ts).timestamp())
        if last_t is not None and t <= last_t:
            points[-1] = {"time": t, "value": float(value)}
            last_t = t
            continue
        points.append({"time": t, "value": float(value)})
        last_t = t
    return points


def _choice_col(choice: str) -> str:
    return "no_price" if choice == "no" else "yes_price"


def _fmt_volume(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def board_payload(event_id: str | None = None) -> dict:
    data = fetch_quotes(event_id)
    tradable = [row for row in data["markets"] if row.get("tradable")]
    return {
        **data,
        "tradable_count": len(tradable),
        "market_count": len(data["markets"]),
    }


def series_payload(event_id: str, market_name: str, choice: str = "yes") -> dict:
    event = get_event(event_id)
    hist = fetch_history(event_id, market_name)
    column = _choice_col(choice)
    k_df = hist["kalshi"]
    p_df = hist["polymarket"]
    quote = hist["quote"]
    k_vol = None if not quote.get("kalshi") else quote["kalshi"].get("volume")
    p_vol = None if not quote.get("polymarket") else quote["polymarket"].get("volume")
    return {
        "event_id": event["id"],
        "event_label": event["label"],
        "market": market_name,
        "market_label": display_name(event_id, market_name),
        "choice": "no" if choice == "no" else "yes",
        "focus": is_focus_market(event_id, market_name),
        "quote": quote,
        "kalshi": _unix_series(k_df, column),
        "polymarket": _unix_series(p_df, column),
        "kalshi_points": 0 if k_df is None or k_df.empty else int(len(k_df)),
        "polymarket_points": 0 if p_df is None or p_df.empty else int(len(p_df)),
        "kalshi_volume_label": _fmt_volume(k_vol),
        "polymarket_volume_label": _fmt_volume(p_vol),
        "volume": p_vol if p_vol is not None else k_vol,
        "volume_label": _fmt_volume(p_vol if p_vol is not None else k_vol),
    }


def predict_payload(event_id: str, market_name: str, threshold: float | str = 0.08) -> dict:
    hist = fetch_history(event_id, market_name)
    k_df = hist["kalshi"]
    p_df = hist["polymarket"]
    if k_df is None or p_df is None or k_df.empty or p_df.empty:
        raise ValueError("Need overlapping Kalshi and Polymarket history for the spread model.")
    prediction = predict_convergence(k_df, p_df, threshold=float(threshold))
    path = predict_path(k_df, p_df)
    return {
        "prediction": prediction,
        "path": {
            "kalshi": _unix_series(path, "kalshi_yes_price"),
            "polymarket": _unix_series(path, "polymarket_yes_price"),
            "basis": _unix_series(path, "basis"),
            "prob_compress": _unix_series(path, "prob_compress"),
        },
        "quote": hist["quote"],
        "market_label": display_name(event_id, market_name),
    }
