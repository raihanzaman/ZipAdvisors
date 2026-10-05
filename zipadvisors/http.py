from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from zipadvisors.config import USER_AGENT


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return number


def as_prob(value: Any) -> float | None:
    """Normalize a price that may be 0–1 or 0–100 cents."""
    number = as_float(value)
    if number is None:
        return None
    if number > 1.5:
        number = number / 100.0
    return max(0.0, min(1.0, number))


def parse_json_field(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def fetch_json(url: str, timeout: int = 45, headers: dict[str, str] | None = None) -> Any:
    req_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    key_id = os.getenv("KALSHI_KEY_ID") or os.getenv("KALSHI_API_KEY_ID")
    if key_id and "kalshi.com" in url and "KALSHI-ACCESS-KEY" not in req_headers:
        req_headers["KALSHI-ACCESS-KEY"] = key_id
    req = urllib.request.Request(url, headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"HTTP {exc.code} for {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc.reason}") from exc


class TtlCache:
    def __init__(self) -> None:
        self._data: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Any | None:
        item = self._data.get(key)
        if item is None:
            return None
        value, expires = item
        if time.time() > expires:
            self._data.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any, ttl: float) -> Any:
        self._data[key] = (value, time.time() + ttl)
        return value


CACHE = TtlCache()
