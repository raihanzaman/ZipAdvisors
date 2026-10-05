"""Spread-convergence model.

Direction-of-price XGBoost did not work: YES prices are close to a random
walk at hourly resolution. The Kalshi−Polymarket *basis* mean-reverts much
more often. This module forecasts whether |Kalshi YES − Poly YES| compresses
over the next few hourly bars, and which venue has been leading.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "xgb_convergence.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
HORIZON = 5
MIN_BASIS = 0.004
FEATURE_COLUMNS = [
    "basis",
    "abs_basis",
    "lag_1_basis",
    "lag_2_basis",
    "delta_basis",
    "kalshi_spread",
    "polymarket_spread",
    "delta_log_kalshi_yes",
    "delta_log_polymarket_yes",
    "kalshi_momentum_5",
    "polymarket_momentum_5",
    "kalshi_vol_8",
    "polymarket_vol_8",
    "lead_diff",
]


def _log_return(series: pd.Series) -> pd.Series:
    clipped = np.clip(series.astype(float), 1e-6, 1.0)
    return np.log(clipped) - np.log(clipped.shift(1))


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

    span = (end - start).total_seconds()
    freq = "1h" if span >= 6 * 3600 else "15min"
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


def build_features(df: pd.DataFrame, labeled: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["delta_log_kalshi_yes"] = _log_return(out["kalshi_yes_price"])
    out["delta_log_polymarket_yes"] = _log_return(out["polymarket_yes_price"])
    out["kalshi_spread"] = out["kalshi_yes_price"] - out["kalshi_no_price"]
    out["polymarket_spread"] = out["polymarket_yes_price"] - out["polymarket_no_price"]
    out["basis"] = out["kalshi_yes_price"] - out["polymarket_yes_price"]
    out["abs_basis"] = out["basis"].abs()
    out["lag_1_basis"] = out["basis"].shift(1)
    out["lag_2_basis"] = out["basis"].shift(2)
    out["delta_basis"] = out["basis"].diff()
    out["kalshi_momentum_5"] = out["delta_log_kalshi_yes"].rolling(5).sum()
    out["polymarket_momentum_5"] = out["delta_log_polymarket_yes"].rolling(5).sum()
    out["kalshi_vol_8"] = out["delta_log_kalshi_yes"].rolling(8).std()
    out["polymarket_vol_8"] = out["delta_log_polymarket_yes"].rolling(8).std()
    out["lead_diff"] = out["delta_log_kalshi_yes"].shift(1) - out["delta_log_polymarket_yes"].shift(1)

    block = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    block.index = out.index
    if not labeled:
        return block

    future_abs = out["abs_basis"].shift(-HORIZON)
    compressed = (future_abs < (out["abs_basis"] - 0.002)).astype("Int64")
    block = block.assign(target=compressed, future_abs=future_abs)
    block = block.iloc[:-HORIZON]
    block = block.loc[out["abs_basis"].iloc[:-HORIZON] >= MIN_BASIS].copy()
    block = block.dropna(subset=["target"])
    block["target"] = block["target"].astype(int)
    return block


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


def venue_leader(aligned: pd.DataFrame) -> dict:
    k = aligned["kalshi_yes_price"].astype(float).diff()
    p = aligned["polymarket_yes_price"].astype(float).diff()
    k_leads = k.shift(1).corr(p)
    p_leads = p.shift(1).corr(k)
    k_leads_f = None if pd.isna(k_leads) else float(k_leads)
    p_leads_f = None if pd.isna(p_leads) else float(p_leads)
    if k_leads_f is None and p_leads_f is None:
        leader = "unclear"
    elif (k_leads_f or 0) >= (p_leads_f or 0):
        leader = "kalshi"
    else:
        leader = "polymarket"
    return {
        "leader": leader,
        "kalshi_leads_corr": k_leads_f,
        "polymarket_leads_corr": p_leads_f,
    }


def empirical_forecast(aligned: pd.DataFrame) -> dict:
    """Always-on baseline: how often a gap this wide compresses in HORIZON bars."""
    basis = aligned["kalshi_yes_price"] - aligned["polymarket_yes_price"]
    abs_basis = basis.abs()
    current = float(abs_basis.iloc[-1])
    future = abs_basis.shift(-HORIZON)
    valid = abs_basis.iloc[:-HORIZON]
    fut = future.iloc[:-HORIZON]
    compressed = fut < (valid - 0.002)
    wide = valid >= max(MIN_BASIS, current * 0.75)
    overall = float(compressed.mean()) if len(compressed) else None
    conditional = float(compressed.loc[wide].mean()) if int(wide.sum()) else overall
    return {
        "current_abs_basis": current,
        "horizon_bars": HORIZON,
        "unconditional_compress_rate": overall,
        "similar_gap_compress_rate": conditional,
        "samples": int(wide.sum()),
        **venue_leader(aligned),
    }


@lru_cache(maxsize=1)
def load_booster() -> xgb.Booster | None:
    if not MODEL_PATH.exists():
        return None
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    return booster


def predict_convergence(kalshi: pd.DataFrame, polymarket: pd.DataFrame, threshold: float = 0.08) -> dict:
    aligned = align_pair(kalshi, polymarket)
    features = build_features(aligned)
    if features.empty:
        raise ValueError("Not enough overlapping history for the spread model.")
    empirical = empirical_forecast(aligned)
    latest = aligned.iloc[-1]
    threshold = float(np.clip(threshold, 0.0, 0.45))

    booster = load_booster()
    model_prob = None
    recent_acc = None
    if booster is not None:
        dtest = xgb.DMatrix(features.iloc[[-1]][FEATURE_COLUMNS], feature_names=FEATURE_COLUMNS)
        model_prob = float(booster.predict(dtest)[0])
        labeled = build_features(aligned, labeled=True)
        if len(labeled) >= 30:
            tail = labeled.tail(80)
            dtail = xgb.DMatrix(tail[FEATURE_COLUMNS], feature_names=FEATURE_COLUMNS)
            preds = (booster.predict(dtail) >= 0.5).astype(int)
            recent_acc = float((preds == tail["target"].to_numpy()).mean())

    prob = model_prob if model_prob is not None else empirical.get("similar_gap_compress_rate")
    if prob is None:
        direction = "neutral"
    elif abs(prob - 0.5) < threshold:
        direction = "neutral"
    else:
        direction = "compress" if prob >= 0.5 else "widen"

    source = "xgboost" if model_prob is not None else "empirical"
    return {
        "direction": direction,
        "source": source,
        "prob_compress": None if prob is None else float(prob),
        "confidence": None if prob is None else abs(float(prob) - 0.5),
        "threshold": threshold,
        "basis": float(latest["kalshi_yes_price"] - latest["polymarket_yes_price"]),
        "kalshi_yes": float(latest["kalshi_yes_price"]),
        "polymarket_yes": float(latest["polymarket_yes_price"]),
        "bars_used": int(len(features)),
        "recent_accuracy": recent_acc,
        "empirical": empirical,
    }


def predict_path(kalshi: pd.DataFrame, polymarket: pd.DataFrame) -> pd.DataFrame:
    aligned = align_pair(kalshi, polymarket)
    features = build_features(aligned)
    out = aligned.loc[features.index, ["kalshi_yes_price", "polymarket_yes_price"]].copy()
    out["basis"] = out["kalshi_yes_price"] - out["polymarket_yes_price"]
    booster = load_booster()
    if booster is None:
        abs_b = out["basis"].abs()
        future = abs_b.shift(-HORIZON)
        out["prob_compress"] = (future < (abs_b - 0.002)).astype(float).ffill().bfill()
    else:
        dtest = xgb.DMatrix(features[FEATURE_COLUMNS], feature_names=FEATURE_COLUMNS)
        out["prob_compress"] = booster.predict(dtest)
    return out


def _collect_training_frame() -> pd.DataFrame:
    from zipadvisors.config import TRACKED_EVENTS
    from zipadvisors.market_data import fetch_history

    rows = []
    for event in TRACKED_EVENTS.values():
        for market in event["focus_markets"]:
            try:
                hist = fetch_history(event["id"], market)
                featured = build_features(align_pair(hist["kalshi"], hist["polymarket"]), labeled=True)
            except (ValueError, KeyError, RuntimeError):
                continue
            if not featured.empty:
                rows.append(featured)
    if not rows:
        raise RuntimeError("No overlapping API history on focus markets. Try again in a moment.")
    return pd.concat(rows, ignore_index=True)


def train_model() -> dict:
    frame = _collect_training_frame()
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
    booster = xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "auc"],
            "max_depth": 3,
            "eta": 0.06,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 4,
            "lambda": 1.0,
            "scale_pos_weight": spw,
            "seed": 7,
        },
        dtrain,
        num_boost_round=280,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=35,
        verbose_eval=False,
    )
    val_prob = booster.predict(dval)
    y_true = val_df["target"].to_numpy()
    val_pred = (val_prob >= 0.5).astype(int)
    metrics = {
        "target": "basis_compress",
        "rows": int(len(frame)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "val_accuracy": float((val_pred == y_true).mean()) if len(y_true) else None,
        "val_auc": None if len(set(y_true)) < 2 else float(_auc(y_true, val_prob)),
        "best_iteration": int(booster.best_iteration) if booster.best_iteration is not None else None,
        "horizon_bars": HORIZON,
        "base_rate": float(frame["target"].mean()),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    load_booster.cache_clear()
    return metrics
