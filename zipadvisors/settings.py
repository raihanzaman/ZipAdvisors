"""Production configuration checks. Local runs stay optional."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def is_production() -> bool:
    return os.getenv("ENV") == "production" or os.getenv("VERCEL") == "1"


def public_auth_config() -> dict:
    return {
        "supabaseUrl": os.getenv("SUPABASE_URL") or "",
        "supabaseAnonKey": os.getenv("SUPABASE_ANON_KEY") or "",
    }


def auth_configured() -> bool:
    cfg = public_auth_config()
    return bool(cfg["supabaseUrl"] and cfg["supabaseAnonKey"] and (os.getenv("SUPABASE_JWT_SECRET") or cfg["supabaseUrl"]))


def assert_production_config() -> None:
    if not is_production():
        return
    missing = [name for name in ("SECRET_KEY", "DATABASE_URL", "SUPABASE_URL", "SUPABASE_ANON_KEY", "CREDENTIALS_KEY") if not os.getenv(name)]
    if not os.getenv("SUPABASE_JWT_SECRET") and not os.getenv("SUPABASE_URL"):
        missing.append("SUPABASE_JWT_SECRET")
    if os.getenv("FLASK_DEBUG") == "1":
        raise RuntimeError("Set FLASK_DEBUG=0 when ENV=production or VERCEL=1.")
    if missing:
        raise RuntimeError("Missing production environment variables: " + ", ".join(missing))
