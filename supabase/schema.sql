-- ETF trend model schema. Run once in the Supabase SQL editor.
--
-- One shared model portfolio on real ETF prices. The public tables are
-- read-only to everyone (the dashboard needs no login); only the service role
-- -- the daily job in GitHub Actions -- writes. Broker orders are private.

-- ----------------------------------------------------------- etf_decisions ---
-- One row per US session: what the model saw and decided after that close.
-- Written once and never updated, so the live track record cannot be edited.
create table if not exists etf_decisions (
  date        date primary key,                 -- the US session whose close was used
  created_at  timestamptz not null default now(),
  live        boolean not null,                 -- true: recorded right after that close, before the next open
  btc_votes   smallint not null check (btc_votes between 0 and 8),
  btc_vol     real not null check (btc_vol between 0 and 1),
  gold_votes  smallint not null check (gold_votes between 0 and 8),
  gold_vol    real not null check (gold_vol between 0 and 1),
  w_btc       real not null check (w_btc between 0 and 1),
  w_gold      real not null check (w_gold between 0 and 1),
  w_cash      real not null check (w_cash between -0.000001 and 1),
  btc_close   numeric not null,
  gold_close  numeric not null,
  tbill       real,
  usdinr      real
);

-- -------------------------------------------------------------- etf_equity ---
-- The model portfolio and buy & hold IBIT from IBIT's launch, $10,000 each,
-- after Indian tax and all charges (liquidation value) and before them.
create table if not exists etf_equity (
  date         date primary key,
  model_net    numeric not null,
  model_gross  numeric not null,
  hold_net     numeric not null,
  hold_gross   numeric not null,
  btc_close    numeric not null,
  gold_close   numeric not null
);

-- --------------------------------------------------------- etf_equity_slab ---
-- The model and buy & hold after tax at each other Indian slab rate (incl. 4% cess);
-- etf_equity holds the default, the top 31.2% slab.
create table if not exists etf_equity_slab (
  date       date not null,
  slab       real not null,
  model_net  numeric not null,
  hold_net   numeric not null,
  primary key (slab, date)
);

-- -------------------------------------------------------------- etf_alerts ---
-- Rebalance alerts for people who trade by hand: the split they were told to move to.
create table if not exists etf_alerts (
  date       date primary key references etf_decisions(date),
  created_at timestamptz not null default now(),
  w_btc      real not null,
  w_gold     real not null,
  w_cash     real not null,
  delivered  boolean not null
);

-- -------------------------------------------------------------- etf_trades ---
create table if not exists etf_trades (
  date   date not null,
  etf    text not null,
  side   text not null check (side in ('BUY', 'SELL')),
  units  numeric not null,
  price  numeric not null,
  value  numeric not null,
  gain   numeric not null,
  primary key (date, etf, side)
);

-- ---------------------------------------------------------------- etf_runs ---
-- Every automation run: the heartbeat the dashboard shows.
create table if not exists etf_runs (
  id          bigserial primary key,
  started_at  timestamptz not null default now(),
  job         text not null check (job in ('decide', 'execute')),
  status      text not null check (status in ('ok', 'skipped', 'error')),
  detail      text
);
create index if not exists etf_runs_recent on etf_runs (started_at desc);

-- -------------------------------------------------------------- etf_orders ---
-- Orders sent to a broker. Private: RLS on and no read policy.
create table if not exists etf_orders (
  id                bigserial primary key,
  created_at        timestamptz not null default now(),
  decision_date     date not null references etf_decisions(date),
  mode              text not null check (mode in ('paper', 'live')),
  symbol            text not null,
  side              text not null check (side in ('buy', 'sell')),
  qty               numeric,
  notional          numeric,
  broker_order_id   text,
  status            text,
  filled_qty        numeric,
  filled_avg_price  numeric,
  unique (decision_date, mode, symbol, side)
);

-- --------------------------------------------------------------------- RLS ---
alter table etf_decisions enable row level security;
alter table etf_equity    enable row level security;
alter table etf_trades    enable row level security;
alter table etf_runs      enable row level security;
alter table etf_orders    enable row level security;
alter table etf_equity_slab enable row level security;
alter table etf_alerts    enable row level security;

create policy public_read on etf_decisions for select to anon, authenticated using (true);
create policy public_read on etf_equity    for select to anon, authenticated using (true);
create policy public_read on etf_trades    for select to anon, authenticated using (true);
create policy public_read on etf_runs      for select to anon, authenticated using (true);
create policy public_read on etf_equity_slab for select to anon, authenticated using (true);
create policy public_read on etf_alerts      for select to anon, authenticated using (true);
