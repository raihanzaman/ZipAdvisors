"""JSON payloads for the ZipAdvisors dashboard (no server-side chart HTML)."""

from __future__ import annotations

import pandas as pd

from db import fetch_latest_volume, fetch_market_frame
from markets import display_name, get_event, is_xgb_market
from model import predict_direction, predict_path

VOL_WINDOW = 12


def _unix_series(df: pd.DataFrame, column: str, time_from: str = "timestamp") -> list[dict]:
    if df is None or df.empty or column not in df.columns:
        return []
    if time_from == "index":
        times = pd.to_datetime(df.index, utc=True)
        values = df[column].to_numpy()
    else:
        ordered = df.sort_values("timestamp")
        times = pd.to_datetime(ordered["timestamp"], utc=True)
        values = ordered[column].to_numpy()
    points = []
    last_t = None
    for ts, value in zip(times, values):
        t = int(pd.Timestamp(ts).timestamp())
        if last_t is not None and t <= last_t:
            points[-1] = {"time": t, "value": float(value)}
            last_t = t
            continue
        points.append({"time": t, "value": float(value)})
        last_t = t
    return points


def _col(choice: str) -> str:
    return "no_price" if choice == "no" else "yes_price"


def _rolling_vol(df: pd.DataFrame, window: int = VOL_WINDOW) -> list[dict]:
    if df is None or df.empty:
        return []
    work = df.sort_values("timestamp").copy()
    work["vol"] = work["yes_price"].rolling(window=window, min_periods=max(3, window // 2)).std()
    return _unix_series(work.dropna(subset=["vol"]), "vol")


def series_payload(event_id: str, market_name: str, choice: str = "yes") -> dict:
    event = get_event(event_id)
    if market_name not in event["markets"]:
        raise ValueError(f"Unknown contract {market_name!r} for {event['label']}.")
    column = _col(choice)
    k_df = fetch_market_frame(f"K_{event_id}", market_name)
    p_df = fetch_market_frame(f"P_{event_id}", market_name)
    volume = fetch_latest_volume(f"P_{event_id}", market_name)
    if volume is None:
        volume = fetch_latest_volume(f"K_{event_id}", market_name)
    return {
        "event_id": event_id,
        "event_label": event["label"],
        "market": market_name,
        "market_label": display_name(event_id, market_name),
        "choice": "no" if choice == "no" else "yes",
        "xgb_eligible": is_xgb_market(event_id, market_name),
        "kalshi": _unix_series(k_df, column),
        "polymarket": _unix_series(p_df, column),
        "kalshi_vol": _rolling_vol(k_df),
        "polymarket_vol": _rolling_vol(p_df),
        "volume": volume,
        "kalshi_points": 0 if k_df.empty else int(len(k_df)),
        "polymarket_points": 0 if p_df.empty else int(len(p_df)),
    }


def predict_payload(
    event_id: str,
    market_name: str,
    target: str = "kalshi",
    threshold: float | str = 0.10,
) -> dict:
    event = get_event(event_id)
    if not is_xgb_market(event_id, market_name):
        allowed = ", ".join(display_name(event_id, slug) for slug in event["focus_markets"])
        raise ValueError(
            f"XGBoost is limited to a few liquid paired contracts: {allowed}."
        )
    target = "polymarket" if target == "polymarket" else "kalshi"
    k_df = fetch_market_frame(f"K_{event_id}", market_name)
    p_df = fetch_market_frame(f"P_{event_id}", market_name)
    if k_df.empty or p_df.empty:
        raise ValueError("Need overlapping Kalshi and Polymarket history to run XGBoost.")
    threshold_f = float(threshold)
    prediction = predict_direction(k_df, p_df, threshold=threshold_f, target=target)
    path = predict_path(k_df, p_df, target=target, threshold=threshold_f)
    return {
        "prediction": prediction,
        "path": {
            "kalshi": _unix_series(path, "kalshi_yes_price", time_from="index"),
            "polymarket": _unix_series(path, "polymarket_yes_price", time_from="index"),
            "prob_up": _unix_series(path, "prob_up", time_from="index"),
        },
        "volume": fetch_latest_volume(f"P_{event_id}", market_name),
    }
