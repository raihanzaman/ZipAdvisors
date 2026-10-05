from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from zipadvisors.config import DEFAULT_EVENT_ID, event_catalog
from zipadvisors.db import database_label, is_postgres
from zipadvisors.resolve import resolve_pair, resolve_summary
from zipadvisors.services import board_payload, predict_payload, series_payload

ROOT = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(ROOT / "web" / "templates"),
    static_folder=str(ROOT / "web" / "static"),
    static_url_path="/static",
)


def _css_for(direction: str | None) -> str:
    return {
        "compress": "signal-up",
        "widen": "signal-down",
        "neutral": "signal-neutral",
    }.get(direction or "", "")


def _format_prediction(prediction: dict | None) -> dict:
    empty = {
        "direction": "NO SIGNAL",
        "prob_compress": "—",
        "confidence": "—",
        "basis": "—",
        "leader": "—",
        "source": "—",
        "recent_accuracy": "—",
        "css": "",
    }
    if not prediction:
        return empty
    direction = prediction.get("direction") or "neutral"
    labels = {"compress": "GAP COMPRESSES", "widen": "GAP WIDENS", "neutral": "NO SIGNAL"}
    recent = prediction.get("recent_accuracy")
    empirical = prediction.get("empirical") or {}
    prob = prediction.get("prob_compress")
    conf = prediction.get("confidence")
    basis = prediction.get("basis")
    return {
        "direction": labels.get(direction, direction.upper()),
        "prob_compress": "—" if prob is None else f"{prob:.1%}",
        "confidence": "—" if conf is None else f"{conf:.1%}",
        "basis": "—" if basis is None else f"{basis:+.3f}",
        "leader": str(empirical.get("leader") or prediction.get("leader") or "—").upper(),
        "source": str(prediction.get("source") or "—"),
        "recent_accuracy": f"{recent:.0%}" if isinstance(recent, float) else "—",
        "css": _css_for(direction),
    }


@app.route("/health")
def health():
    return jsonify(ok=True, postgres=is_postgres(), database=database_label(), ingest="live_api")


@app.route("/api/events")
def api_events():
    return jsonify({"events": event_catalog(), "default_event": DEFAULT_EVENT_ID})


@app.route("/api/resolve", methods=["POST"])
def api_resolve():
    payload = request.get_json(silent=True) or {}
    try:
        event = resolve_pair(payload.get("kalshi_url") or "", payload.get("polymarket_url") or "")
        return jsonify(resolve_summary(event))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/board")
def api_board():
    event_id = request.args.get("event_id") or DEFAULT_EVENT_ID
    try:
        return jsonify(board_payload(event_id))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/series")
def api_series():
    event_id = request.args.get("event_id") or DEFAULT_EVENT_ID
    market = request.args.get("market") or ""
    choice = request.args.get("choice") or "yes"
    if not market:
        return jsonify({"error": "Select a contract."}), 400
    try:
        return jsonify(series_payload(event_id, market, choice))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/predict")
def api_predict():
    event_id = request.args.get("event_id") or DEFAULT_EVENT_ID
    market = request.args.get("market") or ""
    threshold = request.args.get("threshold") or "0.08"
    if not market:
        return jsonify({"error": "Select a contract."}), 400
    try:
        payload = predict_payload(event_id, market, threshold=threshold)
        payload["formatted"] = _format_prediction(payload.get("prediction"))
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/")
def index():
    return render_template("index.html", default_event=DEFAULT_EVENT_ID)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
