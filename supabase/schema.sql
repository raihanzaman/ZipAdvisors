-- ZipAdvisors account schema. Run in the Supabase SQL editor.
-- The Flask app also applies this on startup when DATABASE_URL is set.
-- The postgres role used by DATABASE_URL bypasses RLS. Policies protect the Data API.

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

create table if not exists profiles (
    id uuid primary key references auth.users (id) on delete cascade,
    email text,
    created_at timestamptz not null default now()
);

create table if not exists saved_pairs (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users (id) on delete cascade,
    event_id text not null,
    label text not null,
    kalshi_url text not null default '',
    polymarket_url text not null default '',
    event_json jsonb not null,
    created_at timestamptz not null default now(),
    unique (user_id, event_id)
);

create table if not exists alerts (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users (id) on delete cascade,
    event_id text not null,
    market_slug text not null,
    market_label text,
    min_net_edge double precision not null,
    channel text not null default 'in_app' check (channel in ('in_app', 'email')),
    active boolean not null default true,
    created_at timestamptz not null default now()
);

create table if not exists alert_fires (
    id bigint generated always as identity primary key,
    alert_id uuid references alerts (id) on delete cascade,
    user_id uuid not null references auth.users (id) on delete cascade,
    event_id text,
    market_slug text,
    net_edge double precision,
    fired_at timestamptz not null default now(),
    emailed_at timestamptz,
    seen boolean not null default false
);

create table if not exists paper_trades (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users (id) on delete cascade,
    event_id text not null,
    market_slug text not null,
    market_label text,
    combo_id text,
    combo_label text,
    size double precision not null default 10,
    entry_net_edge double precision,
    entry_cost double precision,
    fee double precision,
    kalshi_yes double precision,
    polymarket_yes double precision,
    opened_at timestamptz not null default now(),
    closed_at timestamptz,
    exit_net_edge double precision,
    pnl double precision
);

create table if not exists venue_credentials (
    user_id uuid not null references auth.users (id) on delete cascade,
    venue text not null check (venue in ('kalshi', 'polymarket')),
    ciphertext text not null,
    nonce text not null default '',
    key_version int not null default 1,
    updated_at timestamptz not null default now(),
    primary key (user_id, venue)
);

create table if not exists quote_cache (
    cache_key text primary key,
    payload jsonb not null,
    expires_at timestamptz not null
);

create table if not exists rate_limits (
    bucket text primary key,
    hits int not null,
    window_start timestamptz not null
);

create table if not exists edge_snapshots (
    id bigint generated always as identity primary key,
    event_id text not null,
    market_slug text not null,
    net_edge double precision,
    basis double precision,
    ts timestamptz not null default now()
);

create index if not exists idx_edge_lookup
    on edge_snapshots (event_id, market_slug, ts desc);

create index if not exists idx_alert_fires_user
    on alert_fires (user_id, fired_at desc);

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id, email)
  values (new.id, new.email)
  on conflict (id) do update set email = excluded.email;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

alter table profiles enable row level security;
alter table saved_pairs enable row level security;
alter table alerts enable row level security;
alter table alert_fires enable row level security;
alter table paper_trades enable row level security;
alter table venue_credentials enable row level security;
alter table quote_cache enable row level security;
alter table rate_limits enable row level security;
alter table edge_snapshots enable row level security;

drop policy if exists profiles_own on profiles;
create policy profiles_own on profiles
  for all using (auth.uid() = id) with check (auth.uid() = id);

drop policy if exists saved_pairs_own on saved_pairs;
create policy saved_pairs_own on saved_pairs
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists alerts_own on alerts;
create policy alerts_own on alerts
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists alert_fires_own on alert_fires;
create policy alert_fires_own on alert_fires
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists paper_own on paper_trades;
create policy paper_own on paper_trades
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists credentials_own on venue_credentials;
create policy credentials_own on venue_credentials
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

revoke all on quote_cache from anon, authenticated;
revoke all on rate_limits from anon, authenticated;
revoke all on edge_snapshots from anon, authenticated;
