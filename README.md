# ZipAdvisors

Flask dashboard for **paired Kalshi / Polymarket contracts**. It charts live ticks with [Lightweight Charts](https://tradingview.github.io/lightweight-charts/) and runs XGBoost to predict the next short-horizon YES move on the venue you selected.

The store is a single `ticks` table on **Supabase (Postgres)**, with SQLite as a local fallback. Scrapers hit the public Kalshi and Polymarket APIs (no Selenium). Only allowlisted events in `markets.py` are trained and predicted; the first one is **MLB World Series Champion 2026**.

## What it does

- **Price** — both venues on each pane (solid = that venue, dashed = the other)
- **Volatility** — YES vs rolling standard deviation
- **XGBoost direction** — whether the **selected venue** moves up over the next 5 aligned bars. Features are lagged returns, spreads, momentum, volatility, and the Kalshi−Polymarket basis
- **Confidence threshold** — minimum `|P(up) − 0.5|` to emit UP/DOWN. The XGBoost view also plots P(up)

XGBoost is limited to a few liquid MLB contracts: Dodgers, Brewers, Yankees, Rays, Phillies, Red Sox. Add more events in `markets.py`.

This is a classroom demo, not a trading system.

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # then paste DATABASE_URL
```

### Connect Supabase

1. Create a project at [supabase.com](https://supabase.com).
2. Save the **database password**.
3. **Project Settings → Database → Connect**.
4. Copy the **Transaction pooler** URI (port `6543`) or **Session pooler** (port `5432`). Use the pooler on Vercel **and** on Windows/WSL — `db.*.supabase.co` is IPv6-only and often fails with “could not translate host name”. If Postgres is unreachable, the app falls back to `data/markets.db`.
5. Put it in `.env`:

```
DATABASE_URL=postgresql://postgres.YOUR_PROJECT_REF:YOUR_PASSWORD@aws-0-YOUR_REGION.pooler.supabase.com:6543/postgres?sslmode=require
```

Optional API keys (not required; the app talks to Postgres directly):

```
SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
SUPABASE_ANON_KEY=eyJ...
```

Load `supabase/schema.sql` in the SQL editor, or let the app/scrapers create `ticks`.

### Collect ticks and train

Scrapers **cannot** run on Vercel. Run them on your machine (two terminals):

```bash
python kalshi_scraper.py "https://kalshi.com/markets/kxmlb/world-series/kxmlb-26"
python polymarket_scraper.py "https://polymarket.com/event/mlb-world-series-champion-2026"
```

Each process backfills ~14 days of hourly history, then polls every 5 minutes. Stop with Ctrl+C.

```bash
python train_model.py
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

`python seed.py --clear` wipes `ticks`. `python seed.py --force` loads **synthetic** NBA/Fed rows — do not mix that with live MLB tests.

## Deploy on Vercel

Vercel serves the Flask UI and JSON APIs. It does **not** scrape. Ticks and trained `models/xgb_*.json` files must already exist.

1. Push this repo to GitHub (never commit `.env`).
2. Import at [vercel.com/new](https://vercel.com/new). Framework preset can stay Other; the entrypoint is `app.py`.
3. **Settings → Environment Variables**
   - `DATABASE_URL` = Supabase **pooler** URI (`sslmode=require`)
   - `FLASK_DEBUG=0`
4. Root directory: repo root. Python version is pinned in `.python-version` (`3.12`).
5. Deploy.
6. Keep scrapers running locally (or on a small always-on box). Retrain with `python train_model.py` and redeploy if you want the new boosters on Vercel.

If the function fails to connect, switch `DATABASE_URL` from `db.PROJECT.supabase.co:5432` to the pooler host. If the build exceeds size limits, the usual culprit is `xgboost`; keep `plotly` out of `requirements.txt`.

## Add another paired event

Edit `TRACKED_EVENTS` in `markets.py` with both venue IDs, a canonical `event_id`, and the contracts you want XGBoost to train on. Restart the scrapers with that event’s URLs.

## Retrain

```bash
python train_model.py
```

Writes `models/xgb_kalshi.json`, `models/xgb_polymarket.json`, and `models/metrics.json`.
