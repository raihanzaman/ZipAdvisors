"""Cross-venue complementary-contract arb from live bids/asks.

Same outcome, two venues. Buying YES on one and NO on the other pays $1
if the contracts resolve against each other. Edge is 1 minus that cost,
minus a simple Kalshi fee estimate.
"""

from __future__ import annotations


def kalshi_taker_fee(price: float | None) -> float:
    if price is None:
        return 0.0
    p = min(max(float(price), 0.01), 0.99)
    return 0.07 * p * (1.0 - p)


PAPER_SIZE = 10


def _leg_size(side: dict | None, yes: bool) -> float | None:
    if not side:
        return None
    if yes:
        value = side.get("yes_ask_size")
    else:
        value = side.get("no_ask_size")
        if value is None:
            value = side.get("yes_bid_size")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _combo_size(*legs: float | None) -> float | None:
    known = [leg for leg in legs if leg is not None]
    if len(known) != len(legs):
        return None
    return min(known) if known else None


def _ask(side: dict | None, yes: bool) -> float | None:
    if not side:
        return None
    key = "yes_ask" if yes else "no_ask"
    value = side.get(key)
    if value is not None:
        return float(value)
    mid_key = "yes_mid" if yes else "no_mid"
    fallback = side.get(mid_key)
    if fallback is None and not yes:
        yes_mid = side.get("yes_mid")
        fallback = None if yes_mid is None else 1.0 - float(yes_mid)
    return None if fallback is None else float(fallback)


def attach_arb(row: dict) -> dict:
    kalshi = row.get("kalshi")
    poly = row.get("polymarket")
    combos = []

    k_yes = _ask(kalshi, True)
    k_no = _ask(kalshi, False)
    p_yes = _ask(poly, True)
    p_no = _ask(poly, False)

    if k_yes is not None and p_no is not None:
        cost = k_yes + p_no
        fee = kalshi_taker_fee(k_yes)
        size = _combo_size(_leg_size(kalshi, True), _leg_size(poly, False))
        combos.append(
            {
                "id": "buy_kalshi_yes_poly_no",
                "label": "Buy Kalshi YES + Poly NO",
                "cost": cost,
                "gross_edge": 1.0 - cost,
                "net_edge": 1.0 - cost - fee,
                "fee": fee,
                "size": size,
                "size_limited": size is not None and size < PAPER_SIZE,
            }
        )
    if p_yes is not None and k_no is not None:
        cost = p_yes + k_no
        fee = kalshi_taker_fee(k_no)
        size = _combo_size(_leg_size(poly, True), _leg_size(kalshi, False))
        combos.append(
            {
                "id": "buy_poly_yes_kalshi_no",
                "label": "Buy Poly YES + Kalshi NO",
                "cost": cost,
                "gross_edge": 1.0 - cost,
                "net_edge": 1.0 - cost - fee,
                "fee": fee,
                "size": size,
                "size_limited": size is not None and size < PAPER_SIZE,
            }
        )
    if k_yes is not None and k_no is not None:
        cost = k_yes + k_no
        fee = kalshi_taker_fee(k_yes) + kalshi_taker_fee(k_no)
        size = _combo_size(_leg_size(kalshi, True), _leg_size(kalshi, False))
        combos.append(
            {
                "id": "kalshi_yes_no",
                "label": "Kalshi YES + NO (same venue)",
                "cost": cost,
                "gross_edge": 1.0 - cost,
                "net_edge": 1.0 - cost - fee,
                "fee": fee,
                "size": size,
                "size_limited": size is not None and size < PAPER_SIZE,
            }
        )
    if p_yes is not None and p_no is not None:
        cost = p_yes + p_no
        size = _combo_size(_leg_size(poly, True), _leg_size(poly, False))
        combos.append(
            {
                "id": "poly_yes_no",
                "label": "Poly YES + NO (same venue)",
                "cost": cost,
                "gross_edge": 1.0 - cost,
                "net_edge": 1.0 - cost,
                "fee": 0.0,
                "size": size,
                "size_limited": size is not None and size < PAPER_SIZE,
            }
        )

    best = None
    if combos:
        best = max(combos, key=lambda item: item["net_edge"])

    k_mid = None if not kalshi else kalshi.get("yes_mid")
    p_mid = None if not poly else poly.get("yes_mid")
    basis = None
    if k_mid is not None and p_mid is not None:
        basis = float(k_mid) - float(p_mid)

    qualities = [side.get("quality") for side in (kalshi, poly) if side]
    quality = "book" if qualities and all(q == "book" for q in qualities) else "indicative"

    row["basis"] = basis
    row["combos"] = combos
    row["best"] = best
    row["net_edge"] = None if best is None else best["net_edge"]
    row["gross_edge"] = None if best is None else best["gross_edge"]
    row["quality"] = "size-limited" if best and best.get("size_limited") else quality
    row["size_limited"] = bool(best and best.get("size_limited"))
    row["book_size"] = None if not best else best.get("size")
    row["tradable"] = bool(best and best["net_edge"] > 0)
    return row
