"""Encrypt venue credentials at rest. Decrypt is for a future order path only."""

from __future__ import annotations

import json
import os

from cryptography.fernet import Fernet


def _fernet() -> Fernet:
    key = os.getenv("CREDENTIALS_KEY") or ""
    if not key:
        raise RuntimeError("CREDENTIALS_KEY is not set. Generate one with Fernet.generate_key().")
    return Fernet(key.encode("utf-8"))


def encrypt_secret(payload: dict) -> str:
    token = _fernet().encrypt(json.dumps(payload).encode("utf-8"))
    return token.decode("utf-8")


def decrypt_secret(token: str) -> dict:
    """Not used by request handlers. Live orders are not implemented."""
    raw = _fernet().decrypt(token.encode("utf-8"))
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Credential payload is not an object.")
    return data
