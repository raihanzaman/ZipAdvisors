from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from db import ensure_schema, is_postgres
from markets import DEFAULT_EVENT_ID, event_catalog, market_catalog
from ml_backend import predict_payload, series_payload

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"

app = Flask(__name__)
ensure_schema()


def _format_volume(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def _format_prediction(prediction: dict | None) -> dict:
    empty = {
        "direction": "N/A",
        "prob_up": "N/A",
        "confidence": "N/A",
        "basis": "N/A",
        "recent_accuracy": "N/A",
        "target": "N/A",
        "css": "",
        "raw": None,
    }
    if not prediction:
        return empty
    direction = prediction["direction"]
    css = {"up": "signal-up", "down": "signal-down"}.get(direction, "signal-neutral")
    recent = prediction.get("recent_accuracy")
    return {
        "direction": direction.upper() if direction != "neutral" else "NO SIGNAL",
        "prob_up": f"{prediction['prob_up']:.1%}",
        "confidence": f"{prediction['confidence']:.1%}",
        "basis": f"{prediction['basis']:+.3f}",
        "recent_accuracy": f"{recent:.0%}" if isinstance(recent, float) else "N/A",
        "target": str(prediction.get("target", "")).title() or "N/A",
        "css": css,
        "raw": prediction,
    }


@app.route("/css/<path:filename>")
def public_css(filename: str):
    return send_from_directory(PUBLIC / "css", filename)


@app.route("/js/<path:filename>")
def public_js(filename: str):
    return send_from_directory(PUBLIC / "js", filename)


@app.route("/health")
def health():
    return jsonify(
        ok=True,
        postgres=is_postgres(),
        backend="supabase" if is_postgres() else "sqlite",
    )


@app.route("/api/events")
def api_events():
    return jsonify({"events": event_catalog(), "default_event": DEFAULT_EVENT_ID})


@app.route("/api/markets")
def api_markets():
    event_id = request.args.get("event_id") or DEFAULT_EVENT_ID
    try:
        return jsonify({"event_id": event_id, "markets": market_catalog(event_id)})
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
        payload = series_payload(event_id, market, choice)
        payload["volume_label"] = _format_volume(payload.get("volume"))
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/predict")
def api_predict():
    event_id = request.args.get("event_id") or DEFAULT_EVENT_ID
    market = request.args.get("market") or ""
    target = request.args.get("target") or "kalshi"
    threshold = request.args.get("threshold") or "0.10"
    if not market:
        return jsonify({"error": "Select a contract."}), 400
    try:
        payload = predict_payload(event_id, market, target=target, threshold=threshold)
        payload["formatted"] = _format_prediction(payload.get("prediction"))
        payload["volume_label"] = _format_volume(payload.get("volume"))
        return jsonify(payload)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/")
def index():
    return render_template("index.html", default_event=DEFAULT_EVENT_ID)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
