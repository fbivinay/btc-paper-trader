"use client";

import { useCallback, useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";
import type { Decision, Position, Profile } from "@/lib/supabase";
import PriceChart from "@/components/PriceChart";
import {
  ModeSelector, OpenPositions, Panel, Portfolio, TradeHistory,
} from "@/components/Trading";

export default function Dashboard() {
  const [ready, setReady] = useState(false);
  const [email, setEmail] = useState<string | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [decision, setDecision] = useState<Decision | null>(null);
  const [price, setPrice] = useState<number | null>(null);

  const load = useCallback(async () => {
    const db = supabase();
    const { data: { user } } = await db.auth.getUser();
    if (!user) { window.location.href = "/login"; return; }
    setEmail(user.email ?? null);

    const [p, pos, d] = await Promise.all([
      db.from("profiles").select("*").eq("user_id", user.id).single(),
      db.from("positions").select("*").order("entry_time", { ascending: false }).limit(200),
      db.from("decisions").select("*").order("candle_time", { ascending: false }).limit(1),
    ]);
    setProfile(p.data ?? null);
    setPositions(pos.data ?? []);
    setDecision(d.data?.[0] ?? null);
    setReady(true);
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 20_000);
    return () => clearInterval(t);
  }, [load]);

  // Live price, independent of the 5-minute loop, so unrealised P&L and the
  // position table move continuously instead of stepping once per candle.
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const r = await fetch(
          "https://data-api.binance.vision/api/v3/ticker/price?symbol=BTCUSDT");
        const j = await r.json();
        if (alive) setPrice(Number(j.price));
      } catch { /* a missed tick just leaves the previous price */ }
    };
    tick();
    const t = setInterval(tick, 5_000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  if (!ready)
    return <main className="min-h-dvh bg-zinc-950 p-6 text-sm text-zinc-500">Loading…</main>;

  const lapsed = profile?.preference_until
    && new Date(profile.preference_until) < new Date();

  return (
    <main className="min-h-dvh bg-zinc-950 text-zinc-100">
      <header className="border-b border-zinc-900">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-2 px-4 py-3">
          <div className="flex items-baseline gap-2">
            <span className="text-sm font-semibold tracking-tight">BTC Paper Trader</span>
            <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-zinc-400">
              virtual
            </span>
          </div>
          <div className="flex items-center gap-3 text-xs text-zinc-500">
            <a href="/model" className="hover:text-zinc-300">Model</a>
            <span className="hidden sm:inline">{email}</span>
            <button
              onClick={async () => { await supabase().auth.signOut(); window.location.href = "/login"; }}
              className="rounded border border-zinc-800 px-2 py-1 hover:border-zinc-600"
            >
              Sign out
            </button>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-6xl space-y-4 px-4 py-5">
        {lapsed && (
          <div className="rounded-md border border-amber-900/60 bg-amber-950/25 px-4 py-2.5 text-sm text-amber-300">
            Your trading preference expired on{" "}
            {new Date(profile!.preference_until!).toLocaleString()}. No new positions
            will open until you set a new one. Open positions are still managed to
            their stop or target.
          </div>
        )}

        <Portfolio profile={profile} positions={positions} price={price} />
        <PriceChart positions={positions} />

        <div className="grid gap-4 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <OpenPositions positions={positions} price={price} />
          </div>
          <ModeSelector profile={profile} onSaved={load} />
        </div>

        <TradeHistory positions={positions} />

        {decision && (
          <Panel title="Latest signal">
            <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
              <Item k="Prediction" v={decision.prediction}
                    tone={decision.prediction === "UP" ? "up"
                        : decision.prediction === "DOWN" ? "down" : undefined} />
              <Item k="Confidence" v={`${(decision.confidence * 100).toFixed(1)}%`} />
              <Item k="Regime" v={decision.regime.replace(/_/g, " ")} />
              <Item k="Strategy" v={decision.strategy.replace(/_/g, " ")} />
              <Item k="Action" v={decision.action}
                    tone={decision.action === "BUY" ? "up"
                        : decision.action === "SELL" ? "down" : undefined} />
              <span className="text-xs text-zinc-600">
                {new Date(decision.candle_time).toLocaleTimeString()} ·{" "}
                <a href="/model" className="underline decoration-zinc-700 hover:text-zinc-400">
                  how this is decided
                </a>
              </span>
            </div>
          </Panel>
        )}

        <p className="pb-6 text-xs text-zinc-600">
          Paper trading only. No real orders, no real funds. Not investment advice.
        </p>
      </div>
    </main>
  );
}

function Item({ k, v, tone }: { k: string; v: string; tone?: "up" | "down" }) {
  return (
    <span>
      <span className="text-zinc-500">{k} </span>
      <span className={tone === "up" ? "text-emerald-400"
                     : tone === "down" ? "text-rose-400" : "text-zinc-100"}>{v}</span>
    </span>
  );
}
