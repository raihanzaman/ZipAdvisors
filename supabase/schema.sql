-- Optional. The live board does not require Postgres.
-- SQL Editor → New query → Run if you want a ticks table for later persistence.

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
