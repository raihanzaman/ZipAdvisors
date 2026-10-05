# ZipAdvisors

Live **Kalshi × Polymarket** board for one allowlisted event (MLB World Series Champion 2026). The app pulls public market data on demand — no always-on scraper.

- **Contracts** — paired YES prices from both venues
- **Charts** — hourly history from the Kalshi candlestick API and Polymarket CLOB
- **Arb board** — complementary YES/NO cost vs $1, net of an estimated Kalshi taker fee
- **Spread model** — whether the Kalshi−Poly gap is likely to compress over the next ~5 hourly bars (not a next-tick price call)

Classroom demo, not a trading system. Fees, latency, and size will eat paper edge.

## Layout

```
app.py                 Flask entry (Vercel + local)
zipadvisors/           API clients, arb, model
web/                   Bloomberg-style UI
models/                trained spread booster (optional)
supabase/              optional schema
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). Quotes refresh about every 45s; history is fetched when you select a contract.

Optional booster (uses the same live APIs):

```bash
python -m zipadvisors.train
```

Until that file exists, the UI uses an empirical compress rate from the selected contract’s own history.

## API keys

**You do not need keys for this app.** Market data is public.

### Kalshi (optional)

Public Trade API: `https://api.elections.kalshi.com/trade-api/v2`  
Docs: [Kalshi API](https://docs.kalshi.com/)

Keys are only required if you later place orders or want authenticated rate limits:

1. Log in at [kalshi.com](https://kalshi.com).
2. Profile / account menu → **Settings** → **API Keys**.
3. Create a key. Kalshi gives you a **Key ID** and a **private key** (PEM).
4. You can put the Key ID in `.env` as `KALSHI_KEY_ID`. This app sends it as `KALSHI-ACCESS-KEY` on Kalshi requests. Full RSA request signing is not implemented because we only read public markets.

### Polymarket (optional)

Public Gamma + CLOB:

- `https://gamma-api.polymarket.com`
- `https://clob.polymarket.com`

Docs: [Polymarket developers](https://docs.polymarket.com/)

No API key is required for event quotes or price history. Trading would need a wallet / CLOB credentials; this project does not trade.

## Supabase

**Optional.** The board and charts talk to Kalshi and Polymarket directly. Quotes are cached in memory for ~45 seconds.

If you still want Postgres (inspect data, future persistence, Vercel later):

1. Create a project at [supabase.com](https://supabase.com). Save the database password.
2. **Project Settings → Database → Connect**.
3. Copy the **Transaction pooler** URI (port `6543`, host `*.pooler.supabase.com`). Avoid `db.*.supabase.co` on Windows; it is IPv6-only.
4. Put it in `.env`:

```
DATABASE_URL=postgresql://postgres.YOUR_REF:YOUR_PASSWORD@aws-0-YOUR_REGION.pooler.supabase.com:6543/postgres?sslmode=require
```

5. SQL editor: run `supabase/schema.sql` if you want the old `ticks` table. The current app does not write to it.

`/health` reports `"database": "supabase"` when that URI connects, otherwise `"none"`.

## Add another event

Edit `TRACKED_EVENTS` in `zipadvisors/config.py` with both venue IDs and the contract slug map.

## Deploy on Vercel

Same as before: `app.py` is the entrypoint. Set `FLASK_DEBUG=0`. `DATABASE_URL` is optional. Scrapers are gone, so you do not need a laptop process after deploy.
