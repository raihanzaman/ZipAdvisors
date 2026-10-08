"""JSON API for the Next.js app. The browser does not load templates from here."""

from __future__ import annotations

import os

from flask import Flask, g, jsonify, request
from flask_cors import CORS

from zipadvisors.accounts import (
    create_alert,
    credential_status,
    delete_alert,
    delete_credentials,
    delete_pair,
    edge_series,
    hydrate_user_events,
    list_alerts,
    list_fires,
    list_pairs,
    list_paper,
    open_paper_trade,
    rename_pair,
    save_credentials,
    save_pair,
    upsert_profile,
)
from zipadvisors.auth import current_user, login_required
from zipadvisors.config import DEFAULT_EVENT_ID, event_catalog
from zipadvisors.db import database_label, is_postgres
from zipadvisors.resolve import resolve_pair, resolve_summary
from zipadvisors.schema import ensure_schema, rate_limited
from zipadvisors.services import board_payload, predict_payload, series_payload
from zipadvisors.settings import assert_production_config, auth_configured, is_production

assert_production_config()

app = Flask(__name__)
_origins = [item.strip() for item in os.getenv("CORS_ORIGIN", "http://localhost:3000").split(",") if item.strip()]
CORS(app, resources={r"/api/*": {"origins": _origins}, r"/health": {"origins": _origins}})

try:
    ensure_schema()
except Exception:
    pass


def _css_for(direction: str | None) -> str:
    return {"compress": "signal-up", "widen": "signal-down", "neutral": "signal-neutral"}.get(direction or "", "")


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
        "leader": str(empirical.get("leader") or "—").upper(),
        "source": str(prediction.get("source") or "—"),
        "recent_accuracy": f"{recent:.0%}" if isinstance(recent, float) else "—",
        "css": _css_for(direction),
    }


def _client_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For") or request.remote_addr or "local"
    return forwarded.split(",")[0].strip()


@app.before_request
def _attach_user():
    user = current_user()
    g.user = user
    if not user:
        return
    try:
        upsert_profile(user["id"], user.get("email"))
        hydrate_user_events(user["id"])
    except Exception:
        return


def _db_error(exc: Exception):
    text = str(exc)
    if "DATABASE_URL" in text or "CREDENTIALS_KEY" in text:
        return jsonify({"error": text}), 503
    return jsonify({"error": text}), 400


@app.route("/health")
def health():
    return jsonify(
        ok=True,
        postgres=is_postgres(),
        database=database_label(),
        ingest="live_api",
        auth=auth_configured(),
    )


@app.route("/api/events")
def api_events():
    return jsonify({"events": event_catalog(), "default_event": DEFAULT_EVENT_ID})


@app.route("/api/resolve", methods=["POST"])
def api_resolve():
    if rate_limited(f"resolve:{_client_ip()}", 30, 600):
        return jsonify({"error": "Too many pair lookups. Wait a few minutes."}), 429
    user = getattr(g, "user", None)
    if user and rate_limited(f"resolve-user:{user['id']}", 60, 600):
        return jsonify({"error": "Too many pair lookups for this account."}), 429
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


@app.route("/api/edge")
def api_edge():
    event_id = request.args.get("event_id") or ""
    market = request.args.get("market") or ""
    if not event_id or not market:
        return jsonify({"error": "event_id and market are required."}), 400
    try:
        return jsonify({"points": edge_series(event_id, market)})
    except Exception as exc:
        return _db_error(exc)


@app.route("/api/pairs", methods=["GET"])
@login_required
def api_pairs_list():
    try:
        return jsonify({"pairs": list_pairs(g.user["id"])})
    except Exception as exc:
        return _db_error(exc)


@app.route("/api/pairs", methods=["POST"])
@login_required
def api_pairs_save():
    payload = request.get_json(silent=True) or {}
    event_id = payload.get("event_id") or ""
    if not event_id:
        return jsonify({"error": "event_id is required."}), 400
    try:
        return jsonify(save_pair(g.user["id"], event_id, payload.get("label")))
    except Exception as exc:
        return _db_error(exc)


@app.route("/api/pairs/<pair_id>", methods=["PATCH"])
@login_required
def api_pairs_rename(pair_id: str):
    label = (request.get_json(silent=True) or {}).get("label") or ""
    if not label.strip():
        return jsonify({"error": "label is required."}), 400
    try:
        if not rename_pair(g.user["id"], pair_id, label):
            return jsonify({"error": "Pair not found."}), 404
        return jsonify({"ok": True})
    except Exception as exc:
        return _db_error(exc)


@app.route("/api/pairs/<pair_id>", methods=["DELETE"])
@login_required
def api_pairs_delete(pair_id: str):
    try:
        if not delete_pair(g.user["id"], pair_id):
            return jsonify({"error": "Pair not found."}), 404
        return jsonify({"ok": True})
    except Exception as exc:
        return _db_error(exc)


@app.route("/api/alerts", methods=["GET"])
@login_required
def api_alerts_list():
    try:
        return jsonify({"alerts": list_alerts(g.user["id"]), "fires": list_fires(g.user["id"])})
    except Exception as exc:
        return _db_error(exc)


@app.route("/api/alerts", methods=["POST"])
@login_required
def api_alerts_create():
    payload = request.get_json(silent=True) or {}
    try:
        row = create_alert(
            g.user["id"],
            payload.get("event_id") or "",
            payload.get("market") or "",
            payload.get("market_label") or payload.get("market") or "",
            float(payload.get("min_net_edge") or 0),
            payload.get("channel") or "in_app",
        )
        return jsonify(row)
    except Exception as exc:
        return _db_error(exc)


@app.route("/api/alerts/<alert_id>", methods=["DELETE"])
@login_required
def api_alerts_delete(alert_id: str):
    try:
        if not delete_alert(g.user["id"], alert_id):
            return jsonify({"error": "Alert not found."}), 404
        return jsonify({"ok": True})
    except Exception as exc:
        return _db_error(exc)


@app.route("/api/paper", methods=["GET"])
@login_required
def api_paper_list():
    try:
        return jsonify({"trades": list_paper(g.user["id"])})
    except Exception as exc:
        return _db_error(exc)


@app.route("/api/paper", methods=["POST"])
@login_required
def api_paper_open():
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(
            open_paper_trade(
                g.user["id"],
                payload.get("event_id") or "",
                payload.get("market") or "",
                float(payload.get("size") or 10),
            )
        )
    except Exception as exc:
        return _db_error(exc)


def _credential_body(venue: str, payload: dict) -> dict:
    if venue == "kalshi":
        key_id = (payload.get("key_id") or "").strip()
        pem = (payload.get("private_key_pem") or "").strip()
        if not key_id or "PRIVATE KEY" not in pem:
            raise ValueError("Kalshi needs a key id and a PEM private key.")
        if len(pem) > 16000:
            raise ValueError("PEM is too large.")
        return {"key_id": key_id, "private_key_pem": pem}
    api_key = (payload.get("api_key") or "").strip()
    secret = (payload.get("secret") or "").strip()
    passphrase = (payload.get("passphrase") or "").strip()
    if not api_key or not secret:
        raise ValueError("Polymarket needs an API key and secret.")
    return {"api_key": api_key, "secret": secret, "passphrase": passphrase}


@app.route("/api/credentials", methods=["GET"])
@login_required
def api_credentials_status():
    try:
        return jsonify(credential_status(g.user["id"]))
    except Exception as exc:
        return _db_error(exc)


@app.route("/api/credentials", methods=["POST"])
@login_required
def api_credentials_save():
    payload = request.get_json(silent=True) or {}
    venue = (payload.get("venue") or "").strip()
    try:
        save_credentials(g.user["id"], venue, _credential_body(venue, payload))
        return jsonify(credential_status(g.user["id"]))
    except Exception as exc:
        return _db_error(exc)


@app.route("/api/credentials/<venue>", methods=["DELETE"])
@login_required
def api_credentials_delete(venue: str):
    try:
        delete_credentials(g.user["id"], venue)
        return jsonify(credential_status(g.user["id"]))
    except Exception as exc:
        return _db_error(exc)


@app.route("/")
def index():
    return jsonify(
        service="zipadvisors-api",
        ui="Run the Next.js app in frontend/ and set NEXT_PUBLIC_API_URL to this host.",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG") == "1" and not is_production()
    app.run(host="0.0.0.0", port=port, debug=debug)
