-- Paper trading schema. Run once in the Supabase SQL editor.
--
-- Two kinds of table live here, and the split is the important part:
--
--   Market-wide   decisions, model_versions  -- one row per candle, shared by
--                 every user, written only by the service role.
--   Per-user      profiles, positions, risk_events, equity_snapshots -- fenced
--                 off by row level security so a user can only ever read their own.
--
-- The 5-minute job runs inference ONCE and writes one `decisions` row, then fans
-- out per user through the risk engine. Prediction depends on the market; sizing
-- and approval depend on the user. Keeping that boundary in the schema is what
-- stops per-user work from multiplying model inference or Jev calls.

-- ---------------------------------------------------------------- profiles ---
create table if not exists profiles (
  user_id           uuid primary key references auth.users(id) on delete cascade,
  created_at        timestamptz not null default now(),

  virtual_capital   numeric(18,2) not null default 10000 check (virtual_capital > 0),
  trading_allocation numeric(5,4) not null default 0.50
                     check (trading_allocation > 0 and trading_allocation <= 1),
  risk_profile      text not null default 'balanced'
                     check (risk_profile in ('conservative','balanced','aggressive')),

  daily_profit_target numeric(5,4) not null default 0.02 check (daily_profit_target > 0),
  max_daily_loss      numeric(5,4) not null default 0.02 check (max_daily_loss > 0),

  autonomous        boolean not null default true,
  active            boolean not null default true
);

-- --------------------------------------------------------------- decisions ---
-- One row per 5m candle. This is the explainability record: everything the
-- system knew and every step it took, so "why did the AI trade?" is a lookup
-- rather than a reconstruction.
create table if not exists decisions (
  id              bigserial primary key,
  candle_time     timestamptz not null unique,
  created_at      timestamptz not null default now(),

  price           numeric(18,2) not null,
  prediction      text not null check (prediction in ('UP','DOWN','NEUTRAL')),
  confidence      real not null check (confidence >= 0 and confidence <= 1),
  prob_up         real, prob_down real, prob_neutral real,

  regime          text not null,
  atr_pct         real,

  strategy        text not null,
  action          text not null check (action in ('BUY','SELL','HOLD')),
  -- Where the strategy choice came from. 'fallback' means Jev was unreachable or
  -- rate limited and the deterministic table decided instead; worth being able to
  -- query, because a week of silent fallback would otherwise look like Jev.
  decided_by      text not null default 'jev' check (decided_by in ('jev','fallback')),
  jev_confidence  real,
  jev_raw         jsonb,

  model_version   text not null
);
create index if not exists decisions_candle_idx on decisions (candle_time desc);

-- --------------------------------------------------------------- positions ---
-- Open and closed positions in one table. A closed position IS the trade record;
-- a separate orders table would only matter with partial fills, which paper
-- trading does not have.
create table if not exists positions (
  id            bigserial primary key,
  user_id       uuid not null references auth.users(id) on delete cascade,
  decision_id   bigint references decisions(id),
  created_at    timestamptz not null default now(),

  side          text not null check (side in ('LONG','SHORT')),
  status        text not null default 'OPEN' check (status in ('OPEN','CLOSED')),
  strategy      text,

  qty           numeric(18,8) not null check (qty > 0),
  entry_price   numeric(18,2) not null,
  entry_time    timestamptz not null,
  stop_loss     numeric(18,2) not null,
  take_profit   numeric(18,2) not null,

  exit_price    numeric(18,2),
  exit_time     timestamptz,
  exit_reason   text check (exit_reason in ('STOP_LOSS','TAKE_PROFIT','HORIZON','MANUAL')),

  fees          numeric(18,4) not null default 0,
  pnl           numeric(18,4),
  pnl_pct       real
);
create index if not exists positions_user_idx on positions (user_id, status, entry_time desc);

-- ------------------------------------------------------------- risk_events ---
-- Every rejection, with the rule that fired. Without this the dashboard can only
-- say "no trade" and never why, which is the question users actually ask.
create table if not exists risk_events (
  id          bigserial primary key,
  user_id     uuid not null references auth.users(id) on delete cascade,
  decision_id bigint references decisions(id),
  created_at  timestamptz not null default now(),
  rule        text not null,
  detail      text,
  approved    boolean not null
);
create index if not exists risk_events_user_idx on risk_events (user_id, created_at desc);

-- -------------------------------------------------------- equity_snapshots ---
create table if not exists equity_snapshots (
  user_id     uuid not null references auth.users(id) on delete cascade,
  candle_time timestamptz not null,
  equity      numeric(18,2) not null,
  cash        numeric(18,2) not null,
  open_pnl    numeric(18,4) not null default 0,
  primary key (user_id, candle_time)
);

-- ----------------------------------------------------------- model_versions ---
create table if not exists model_versions (
  version        text primary key,
  created_at     timestamptz not null default now(),
  trained_until  timestamptz,
  window_days    int,
  metrics        jsonb not null,
  -- Promotion is gated on validation, so a row existing is not the same as it
  -- being live. Exactly one row may be in production at a time.
  status         text not null default 'candidate'
                  check (status in ('candidate','production','rejected')),
  promoted_at    timestamptz,
  notes          text
);
create unique index if not exists one_production_model
  on model_versions (status) where status = 'production';

-- --------------------------------------------------------------------- RLS ---
alter table profiles          enable row level security;
alter table positions         enable row level security;
alter table risk_events       enable row level security;
alter table equity_snapshots  enable row level security;
alter table decisions         enable row level security;
alter table model_versions    enable row level security;

-- Users read and write only their own rows. The service role key used by the
-- 5-minute job bypasses RLS entirely, so no policy is needed for the writer.
create policy own_profile on profiles
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy own_positions on positions
  for select using (auth.uid() = user_id);

create policy own_risk_events on risk_events
  for select using (auth.uid() = user_id);

create policy own_equity on equity_snapshots
  for select using (auth.uid() = user_id);

-- Market-wide tables are readable by any signed-in user, writable only by the
-- service role.
create policy read_decisions on decisions
  for select to authenticated using (true);

create policy read_models on model_versions
  for select to authenticated using (true);

-- ------------------------------------------------------------------ signup ---
-- A profile row must exist before the 5-minute job can consider a user, and the
-- job must not be the thing that creates it. Trigger keeps that guaranteed.
create or replace function handle_new_user()
returns trigger language plpgsql security definer set search_path = public as $$
begin
  insert into profiles (user_id) values (new.id) on conflict do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function handle_new_user();
