"""Verify Supabase access tokens. Public market routes do not require one."""

from __future__ import annotations

import os
from functools import wraps

import jwt
from flask import g, jsonify, request
from jwt import PyJWKClient

_jwks: PyJWKClient | None = None


def _jwks_client() -> PyJWKClient:
    global _jwks
    if _jwks is None:
        base = (os.getenv("SUPABASE_URL") or "").rstrip("/")
        if not base:
            raise RuntimeError("SUPABASE_URL is not set.")
        _jwks = PyJWKClient(f"{base}/auth/v1/.well-known/jwks.json")
    return _jwks


def verify_access_token(token: str) -> dict:
    secret = os.getenv("SUPABASE_JWT_SECRET")
    if secret:
        payload = jwt.decode(token, secret, algorithms=["HS256"], audience="authenticated")
    else:
        signing_key = _jwks_client().get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256", "RS256"],
            audience="authenticated",
        )
    user_id = payload.get("sub")
    if not user_id:
        raise jwt.InvalidTokenError("Token has no subject.")
    return {"id": str(user_id), "email": payload.get("email") or ""}


def current_user() -> dict | None:
    header = request.headers.get("Authorization") or ""
    if not header.startswith("Bearer "):
        return None
    token = header[7:].strip()
    if not token:
        return None
    try:
        return verify_access_token(token)
    except Exception:
        return None


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        user = current_user()
        if user is None:
            return jsonify({"error": "Sign in required."}), 401
        g.user = user
        return view(*args, **kwargs)

    return wrapper
