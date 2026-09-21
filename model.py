"""XGBoost direction models for Kalshi and Polymarket.

Each booster predicts whether that venue's YES price rises over the next few
bars. Features are shared: lagged returns, spreads, momentum, and the
Kalshi−Polymarket basis (the main cross-venue signal).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
HORIZON = 5
MIN_MOVE = 0.0015
FEATURE_COLUMNS = [
    "basis",
    "lag_1_basis",
    "lag_2_basis",
    "kalshi_spread",
    "polymarket_spread",
    "delta_log_kalshi_yes",
    "delta_log_polymarket_yes",
    "delta_log_kalshi_no",
    "delta_log_polymarket_no",
    "kalshi_momentum_5",
    "polymarket_momentum_5",
    "kalshi_momentum_10",
    "polymarket_momentum_10",
    "kalshi_vol_8",
    "polymarket_vol_8",
    "lag_1_delta_log_kalshi_yes",
    "lag_2_delta_log_kalshi_yes",
    "lag_3_delta_log_kalshi_yes",
    "lag_1_delta_log_polymarket_yes",
    "lag_2_delta_log_polymarket_yes",
    "lag_3_delta_log_polymarket_yes",
]
TARGETS = ("polymarket", "kalshi")


def model_path(target: str) -> Path:
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    return MODELS_DIR / f"xgb_{target}.json"


def _log_return(series: pd.Series) -> pd.Series:
    clipped = np.clip(series.astype(float), 1e-6, 1.0)
    return np.log(clipped) - np.log(clipped.shift(1))


def _align_freq(start: pd.Timestamp, end: pd.Timestamp) -> str:
    span = (end - start).total_seconds()
    if span >= 3 * 86400:
        return "1h"
    if span >= 6 * 3600:
        return "15min"
    return "1min"


def align_pair(kalshi: pd.DataFrame, polymarket: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for df, prefix in ((kalshi, "kalshi"), (polymarket, "polymarket")):
        if df is None or df.empty:
            raise ValueError(f"{prefix} price history is empty.")
        piece = df.copy()
        piece["timestamp"] = pd.to_datetime(piece["timestamp"], utc=True)
        piece = piece.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        piece = piece.set_index("timestamp")[["yes_price", "no_price"]]
        piece = piece.rename(
            columns={
                "yes_price": f"{prefix}_yes_price",
                "no_price": f"{prefix}_no_price",
            }
        )
        frames.append(piece)

    start = max(frame.index.min() for frame in frames)
    end = min(frame.index.max() for frame in frames)
    if pd.isna(start) or pd.isna(end) or start >= end:
        raise ValueError("Kalshi and Polymarket histories do not overlap.")

    freq = _align_freq(start, end)
    bucketed = []
    for frame in frames:
        work = frame.loc[(frame.index >= start) & (frame.index <= end)].copy()
        work.index = work.index.floor(freq)
        work = work.groupby(level=0).last()
        bucketed.append(work)

    joined = bucketed[0].join(bucketed[1], how="outer").sort_index()
    joined = joined.interpolate(method="time").ffill().bfill()
    if joined.empty or joined.isna().all().any():
        raise ValueError("Kalshi and Polymarket histories do not overlap.")
    return joined.dropna()


def build_features(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    out = df.copy()
    out["delta_log_kalshi_yes"] = _log_return(out["kalshi_yes_price"])
    out["delta_log_kalshi_no"] = _log_return(out["kalshi_no_price"])
    out["delta_log_polymarket_yes"] = _log_return(out["polymarket_yes_price"])
    out["delta_log_polymarket_no"] = _log_return(out["polymarket_no_price"])
    out["kalshi_spread"] = out["kalshi_yes_price"] - out["kalshi_no_price"]
    out["polymarket_spread"] = out["polymarket_yes_price"] - out["polymarket_no_price"]
    out["basis"] = out["kalshi_yes_price"] - out["polymarket_yes_price"]
    out["lag_1_basis"] = out["basis"].shift(1)
    out["lag_2_basis"] = out["basis"].shift(2)
    out["kalshi_momentum_5"] = out["delta_log_kalshi_yes"].rolling(5).sum()
    out["polymarket_momentum_5"] = out["delta_log_polymarket_yes"].rolling(5).sum()
    out["kalshi_momentum_10"] = out["delta_log_kalshi_yes"].rolling(10).sum()
    out["polymarket_momentum_10"] = out["delta_log_polymarket_yes"].rolling(10).sum()
    out["kalshi_vol_8"] = out["delta_log_kalshi_yes"].rolling(8).std()
    out["polymarket_vol_8"] = out["delta_log_polymarket_yes"].rolling(8).std()

    for lag in (1, 2, 3):
        out[f"lag_{lag}_delta_log_kalshi_yes"] = out["delta_log_kalshi_yes"].shift(lag)
        out[f"lag_{lag}_delta_log_polymarket_yes"] = out["delta_log_polymarket_yes"].shift(lag)

    feature_block = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    feature_block.index = out.index

    if target:
        price_col = f"{target}_yes_price"
        future = np.log(np.clip(out[price_col].shift(-HORIZON), 1e-6, 1.0))
        now = np.log(np.clip(out[price_col], 1e-6, 1.0))
        move = future - now
        feature_block = feature_block.assign(move=move, target=(move > 0).astype("Int64"))
        feature_block = feature_block.iloc[:-HORIZON]
        feature_block = feature_block.dropna(subset=["target", "move"])
        large = feature_block["move"].abs() >= MIN_MOVE
        feature_block = feature_block.loc[large].copy()
        feature_block["target"] = feature_block["target"].astype(int)
    return feature_block


def _collect_training_frame(target: str) -> pd.DataFrame:
    from db import fetch_market_frame
    from markets import TRACKED_EVENTS

    rows = []
    for event in TRACKED_EVENTS.values():
        k_key = f"K_{event['id']}"
        p_key = f"P_{event['id']}"
        for market in event["focus_markets"]:
            try:
                aligned = align_pair(
                    fetch_market_frame(k_key, market),
                    fetch_market_frame(p_key, market),
                )
                featured = build_features(aligned, target=target)
            except (ValueError, KeyError):
                continue
            if not featured.empty:
                rows.append(featured)
    if not rows:
        raise RuntimeError(
            "No overlapping Kalshi/Polymarket history on allowlisted markets. "
            "Run the scrapers, then python train_model.py."
        )
    return pd.concat(rows, ignore_index=True)


def _finite(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    if np.isnan(number) or np.isinf(number):
        return None
    return number


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def train_one(target: str) -> dict:
    frame = _collect_training_frame(target)
    split = max(int(len(frame) * 0.8), 1)
    train_df = frame.iloc[:split]
    val_df = frame.iloc[split:]
    if val_df.empty:
        val_df = train_df.tail(max(len(train_df) // 5, 1))

    n_pos = int((train_df["target"] == 1).sum())
    n_neg = int((train_df["target"] == 0).sum())
    spw = (n_neg / n_pos) if n_pos else 1.0

    dtrain = xgb.DMatrix(train_df[FEATURE_COLUMNS], label=train_df["target"], feature_names=FEATURE_COLUMNS)
    dval = xgb.DMatrix(val_df[FEATURE_COLUMNS], label=val_df["target"], feature_names=FEATURE_COLUMNS)
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": 3,
        "eta": 0.06,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 4,
        "lambda": 1.0,
        "gamma": 0.0,
        "scale_pos_weight": spw,
        "seed": 7,
    }
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=35,
        verbose_eval=False,
    )

    val_prob = booster.predict(dval)
    val_pred = (val_prob >= 0.5).astype(int)
    y_true = val_df["target"].to_numpy()
    metrics = {
        "target": target,
        "rows": int(len(frame)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "val_accuracy": _finite((val_pred == y_true).mean()) if len(y_true) else None,
        "val_auc": _finite(_auc(y_true, val_prob)) if len(set(y_true)) == 2 else None,
        "best_iteration": int(booster.best_iteration) if booster.best_iteration is not None else None,
        "horizon_bars": HORIZON,
    }
    path = model_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(path)
    load_booster.cache_clear()
    return metrics


def train_model(model_path: Path | None = None) -> dict:
    """Train both venue models. `model_path` is ignored; kept for call-site compatibility."""
    metrics = {target: train_one(target) for target in TARGETS}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


@lru_cache(maxsize=4)
def load_booster(target: str = "polymarket") -> xgb.Booster:
    path = model_path(target)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Collect live ticks then run: python train_model.py"
        )
    booster = xgb.Booster()
    booster.load_model(path)
    return booster


def _call_from_prob(prob_up: float, threshold: float) -> str:
    confidence = abs(prob_up - 0.5)
    if confidence < threshold:
        return "neutral"
    return "up" if prob_up >= 0.5 else "down"


def predict_direction(
    kalshi: pd.DataFrame,
    polymarket: pd.DataFrame,
    threshold: float = 0.10,
    target: str = "polymarket",
) -> dict:
    aligned = align_pair(kalshi, polymarket)
    featured = build_features(aligned)
    if featured.empty:
        raise ValueError("Not enough history to compute model features.")

    latest = featured.iloc[[-1]][FEATURE_COLUMNS]
    dtest = xgb.DMatrix(latest, feature_names=FEATURE_COLUMNS)
    prob_up = float(load_booster(target).predict(dtest)[0])
    threshold = float(np.clip(threshold, 0.0, 0.5))
    direction = _call_from_prob(prob_up, threshold)
    latest_aligned = aligned.iloc[-1]

    labeled = build_features(aligned, target=target)
    recent_acc = None
    if len(labeled) >= 40:
        tail = labeled.tail(120)
        dtail = xgb.DMatrix(tail[FEATURE_COLUMNS], feature_names=FEATURE_COLUMNS)
        preds = (load_booster(target).predict(dtail) >= 0.5).astype(int)
        recent_acc = float((preds == tail["target"].to_numpy()).mean())

    return {
        "direction": direction,
        "target": target,
        "prob_up": prob_up,
        "prob_down": 1.0 - prob_up,
        "confidence": abs(prob_up - 0.5),
        "threshold": threshold,
        "basis": float(latest_aligned["kalshi_yes_price"] - latest_aligned["polymarket_yes_price"]),
        "kalshi_yes": float(latest_aligned["kalshi_yes_price"]),
        "polymarket_yes": float(latest_aligned["polymarket_yes_price"]),
        "bars_used": int(len(featured)),
        "recent_accuracy": recent_acc,
    }


def predict_path(
    kalshi: pd.DataFrame,
    polymarket: pd.DataFrame,
    target: str = "polymarket",
    threshold: float = 0.10,
) -> pd.DataFrame:
    """P(up) for every aligned bar, used by the XGBoost charts."""
    aligned = align_pair(kalshi, polymarket)
    featured = build_features(aligned)
    dtest = xgb.DMatrix(featured[FEATURE_COLUMNS], feature_names=FEATURE_COLUMNS)
    probs = load_booster(target).predict(dtest)
    out = aligned.loc[featured.index, ["kalshi_yes_price", "polymarket_yes_price"]].copy()
    out["prob_up"] = probs
    out["direction"] = [_call_from_prob(float(p), threshold) for p in probs]
    return out
