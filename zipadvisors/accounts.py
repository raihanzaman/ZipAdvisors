"""User rows. The server filters by the verified JWT subject."""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import text

from zipadvisors.config import get_event, register_event
from zipadvisors.crypto import encrypt_secret
from zipadvisors.db import get_engine

PAPER_CLOSE_EDGE = 0.001


def _engine():
    engine = get_engine()
    if engine is None:
        raise RuntimeError("DATABASE_URL is not set or Postgres is unreachable.")
    return engine


def _json(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _iso(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def upsert_profile(user_id: str, email: str | None) -> None:
    with _engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO profiles (id, email)
                VALUES (CAST(:id AS uuid), :email)
                ON CONFLICT (id) DO UPDATE SET email = COALESCE(EXCLUDED.email, profiles.email)
                """
            ),
            {"id": user_id, "email": email or None},
        )


def hydrate_user_events(user_id: str) -> None:
    engine = get_engine()
    if engine is None:
        return
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT event_json FROM saved_pairs WHERE user_id = CAST(:id AS uuid)"),
                {"id": user_id},
            ).fetchall()
    except Exception:
        return
    for row in rows:
        event = _json(row[0])
        if isinstance(event, dict) and event.get("id"):
            register_event(event)


def save_pair(user_id: str, event_id: str, label: str | None = None) -> dict:
    event = get_event(event_id)
    stored = dict(event)
    stored["custom"] = True
    name = (label or event.get("label") or event_id).strip()
    with _engine().begin() as conn:
        row = conn.execute(
            text(
                """
                INSERT INTO saved_pairs (user_id, event_id, label, kalshi_url, polymarket_url, event_json)
                VALUES (
                  CAST(:user_id AS uuid), :event_id, :label, :kalshi_url, :polymarket_url,
                  CAST(:event_json AS jsonb)
                )
                ON CONFLICT (user_id, event_id) DO UPDATE SET
                  label = EXCLUDED.label,
                  kalshi_url = EXCLUDED.kalshi_url,
                  polymarket_url = EXCLUDED.polymarket_url,
                  event_json = EXCLUDED.event_json
                RETURNING id, event_id, label, created_at
                """
            ),
            {
                "user_id": user_id,
                "event_id": event_id,
                "label": name,
                "kalshi_url": event.get("kalshi_url") or "",
                "polymarket_url": event.get("polymarket_url") or "",
                "event_json": json.dumps(stored),
            },
        ).mappings().one()
    register_event(stored)
    return {
        "id": str(row["id"]),
        "event_id": row["event_id"],
        "label": row["label"],
        "created_at": _iso(row["created_at"]),
    }


def list_pairs(user_id: str) -> list[dict]:
    with _engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT sp.id, sp.event_id, sp.label, sp.kalshi_url, sp.polymarket_url, sp.created_at,
                       sp.event_json,
                       (
                         SELECT max(es.net_edge) FROM edge_snapshots es
                         WHERE es.event_id = sp.event_id AND es.ts > now() - interval '2 days'
                       ) AS last_edge,
                       (
                         SELECT max(es.ts) FROM edge_snapshots es
                         WHERE es.event_id = sp.event_id
                       ) AS last_scan
                FROM saved_pairs sp
                WHERE sp.user_id = CAST(:user_id AS uuid)
                ORDER BY sp.created_at DESC
                """
            ),
            {"user_id": user_id},
        ).mappings()
        payload = []
        for row in rows:
            event = _json(row["event_json"])
            if isinstance(event, dict) and event.get("id"):
                register_event(event)
            payload.append(
                {
                    "id": str(row["id"]),
                    "event_id": row["event_id"],
                    "label": row["label"],
                    "kalshi_url": row["kalshi_url"],
                    "polymarket_url": row["polymarket_url"],
                    "created_at": _iso(row["created_at"]),
                    "last_edge": None if row["last_edge"] is None else float(row["last_edge"]),
                    "last_scan": _iso(row["last_scan"]),
                }
            )
        return payload


def rename_pair(user_id: str, pair_id: str, label: str) -> bool:
    with _engine().begin() as conn:
        result = conn.execute(
            text(
                """
                UPDATE saved_pairs SET label = :label
                WHERE id = CAST(:id AS uuid) AND user_id = CAST(:user_id AS uuid)
                """
            ),
            {"label": label.strip(), "id": pair_id, "user_id": user_id},
        )
        return bool(result.rowcount)


def delete_pair(user_id: str, pair_id: str) -> bool:
    with _engine().begin() as conn:
        result = conn.execute(
            text(
                """
                DELETE FROM saved_pairs
                WHERE id = CAST(:id AS uuid) AND user_id = CAST(:user_id AS uuid)
                """
            ),
            {"id": pair_id, "user_id": user_id},
        )
        return bool(result.rowcount)


def create_alert(user_id: str, event_id: str, market_slug: str, market_label: str, min_net_edge: float, channel: str) -> dict:
    get_event(event_id)
    channel = "email" if channel == "email" else "in_app"
    with _engine().begin() as conn:
        row = conn.execute(
            text(
                """
                INSERT INTO alerts (user_id, event_id, market_slug, market_label, min_net_edge, channel)
                VALUES (CAST(:user_id AS uuid), :event_id, :market_slug, :market_label, :min_net_edge, :channel)
                RETURNING id, event_id, market_slug, market_label, min_net_edge, channel, active, created_at
                """
            ),
            {
                "user_id": user_id,
                "event_id": event_id,
                "market_slug": market_slug,
                "market_label": market_label,
                "min_net_edge": float(min_net_edge),
                "channel": channel,
            },
        ).mappings().one()
    return _alert_row(row)


def list_alerts(user_id: str) -> list[dict]:
    with _engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, event_id, market_slug, market_label, min_net_edge, channel, active, created_at
                FROM alerts
                WHERE user_id = CAST(:user_id AS uuid) AND active
                ORDER BY created_at DESC
                """
            ),
            {"user_id": user_id},
        ).mappings()
        return [_alert_row(row) for row in rows]


def delete_alert(user_id: str, alert_id: str) -> bool:
    with _engine().begin() as conn:
        result = conn.execute(
            text(
                """
                UPDATE alerts SET active = false
                WHERE id = CAST(:id AS uuid) AND user_id = CAST(:user_id AS uuid)
                """
            ),
            {"id": alert_id, "user_id": user_id},
        )
        return bool(result.rowcount)


def list_fires(user_id: str, limit: int = 20) -> list[dict]:
    with _engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, alert_id, event_id, market_slug, net_edge, fired_at, emailed_at, seen
                FROM alert_fires
                WHERE user_id = CAST(:user_id AS uuid)
                ORDER BY fired_at DESC
                LIMIT :limit
                """
            ),
            {"user_id": user_id, "limit": int(limit)},
        ).mappings()
        return [
            {
                "id": int(row["id"]),
                "alert_id": None if row["alert_id"] is None else str(row["alert_id"]),
                "event_id": row["event_id"],
                "market_slug": row["market_slug"],
                "net_edge": None if row["net_edge"] is None else float(row["net_edge"]),
                "fired_at": _iso(row["fired_at"]),
                "emailed": row["emailed_at"] is not None,
                "seen": bool(row["seen"]),
            }
            for row in rows
        ]


def open_paper_trade(user_id: str, event_id: str, market_slug: str, size: float = 10) -> dict:
    from zipadvisors.market_data import fetch_quotes

    quotes = fetch_quotes(event_id)
    match = next((row for row in quotes["markets"] if row["id"] == market_slug), None)
    if not match or not match.get("best"):
        raise ValueError("No priced combo on that contract.")
    best = match["best"]
    k_yes = None if not match.get("kalshi") else match["kalshi"].get("yes_mid")
    p_yes = None if not match.get("polymarket") else match["polymarket"].get("yes_mid")
    with _engine().begin() as conn:
        row = conn.execute(
            text(
                """
                INSERT INTO paper_trades (
                  user_id, event_id, market_slug, market_label, combo_id, combo_label,
                  size, entry_net_edge, entry_cost, fee, kalshi_yes, polymarket_yes
                )
                VALUES (
                  CAST(:user_id AS uuid), :event_id, :market_slug, :market_label, :combo_id, :combo_label,
                  :size, :entry_net_edge, :entry_cost, :fee, :kalshi_yes, :polymarket_yes
                )
                RETURNING id, opened_at
                """
            ),
            {
                "user_id": user_id,
                "event_id": event_id,
                "market_slug": market_slug,
                "market_label": match.get("label"),
                "combo_id": best.get("id"),
                "combo_label": best.get("label"),
                "size": float(size),
                "entry_net_edge": best.get("net_edge"),
                "entry_cost": best.get("cost"),
                "fee": best.get("fee"),
                "kalshi_yes": k_yes,
                "polymarket_yes": p_yes,
            },
        ).mappings().one()
    return {
        "id": str(row["id"]),
        "opened_at": _iso(row["opened_at"]),
        "combo_label": best.get("label"),
        "entry_net_edge": best.get("net_edge"),
        "size": float(size),
    }


def list_paper(user_id: str) -> list[dict]:
    with _engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, event_id, market_slug, market_label, combo_label, size,
                       entry_net_edge, exit_net_edge, pnl, opened_at, closed_at
                FROM paper_trades
                WHERE user_id = CAST(:user_id AS uuid)
                ORDER BY opened_at DESC
                LIMIT 40
                """
            ),
            {"user_id": user_id},
        ).mappings()
        return [
            {
                "id": str(row["id"]),
                "event_id": row["event_id"],
                "market_slug": row["market_slug"],
                "market_label": row["market_label"],
                "combo_label": row["combo_label"],
                "size": None if row["size"] is None else float(row["size"]),
                "entry_net_edge": None if row["entry_net_edge"] is None else float(row["entry_net_edge"]),
                "exit_net_edge": None if row["exit_net_edge"] is None else float(row["exit_net_edge"]),
                "pnl": None if row["pnl"] is None else float(row["pnl"]),
                "opened_at": _iso(row["opened_at"]),
                "closed_at": _iso(row["closed_at"]),
                "open": row["closed_at"] is None,
            }
            for row in rows
        ]


def edge_series(event_id: str, market_slug: str, limit: int = 500) -> list[dict]:
    engine = get_engine()
    if engine is None:
        return []
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT ts, net_edge FROM edge_snapshots
                WHERE event_id = :event_id AND market_slug = :market_slug
                ORDER BY ts DESC
                LIMIT :limit
                """
            ),
            {"event_id": event_id, "market_slug": market_slug, "limit": int(limit)},
        ).fetchall()
    points = []
    for ts, net_edge in reversed(rows):
        if net_edge is None or ts is None:
            continue
        points.append({"time": int(ts.timestamp()), "value": float(net_edge)})
    return points


def save_credentials(user_id: str, venue: str, secret: dict) -> None:
    if venue not in {"kalshi", "polymarket"}:
        raise ValueError("Venue must be kalshi or polymarket.")
    token = encrypt_secret(secret)
    with _engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO venue_credentials (user_id, venue, ciphertext, nonce, key_version)
                VALUES (CAST(:user_id AS uuid), :venue, :ciphertext, '', 1)
                ON CONFLICT (user_id, venue) DO UPDATE SET
                  ciphertext = EXCLUDED.ciphertext,
                  key_version = 1,
                  updated_at = now()
                """
            ),
            {"user_id": user_id, "venue": venue, "ciphertext": token},
        )


def credential_status(user_id: str) -> dict:
    linked = {"kalshi": False, "polymarket": False}
    with _engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT venue FROM venue_credentials
                WHERE user_id = CAST(:user_id AS uuid)
                """
            ),
            {"user_id": user_id},
        ).fetchall()
    for row in rows:
        if row[0] in linked:
            linked[row[0]] = True
    return linked


def delete_credentials(user_id: str, venue: str) -> bool:
    with _engine().begin() as conn:
        result = conn.execute(
            text(
                """
                DELETE FROM venue_credentials
                WHERE user_id = CAST(:user_id AS uuid) AND venue = :venue
                """
            ),
            {"user_id": user_id, "venue": venue},
        )
        return bool(result.rowcount)


def _alert_row(row) -> dict:
    return {
        "id": str(row["id"]),
        "event_id": row["event_id"],
        "market_slug": row["market_slug"],
        "market_label": row["market_label"],
        "min_net_edge": float(row["min_net_edge"]),
        "channel": row["channel"],
        "active": bool(row["active"]),
        "created_at": _iso(row["created_at"]),
    }
