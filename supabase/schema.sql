-- ZipAdvisors ticks table for Supabase (Postgres).
-- SQL Editor → New query → Run.
-- Scrapers write live Kalshi / Polymarket ticks. Do not load synthetic seed data
-- if you are testing the allowlisted MLB markets.

create table if not exists ticks (
    id bigint generated always as identity primary key,
    venue text not null check (venue in ('kalshi', 'polymarket')),
    event_id text not null,
    market_name text not null,
    yes_price double precision not null,
    no_price double precision not null,
    trading_volume double precision,
    ts timestamptz not null default now()
);

create index if not exists idx_ticks_lookup
    on ticks (venue, event_id, market_name, ts);
