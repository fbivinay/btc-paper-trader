"use client";

import { useCallback, useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";
import type {
  Decision, Equity, ModelVersion, Position, Profile, RiskEvent,
} from "@/lib/supabase";
import {
  Card, EdgePanel, EquityChart, LatestDecision, Performance, Positions,
  RecentRejections, Stat,
} from "@/components/Panels";

const usd = (n: number) =>
  `$${n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

export default function Dashboard() {
  const [ready, setReady] = useState(false);
  const [email, setEmail] = useState<string | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [decision, setDecision] = useState<Decision | null>(null);
  const [risk, setRisk] = useState<RiskEvent | null>(null);
  const [events, setEvents] = useState<RiskEvent[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [equity, setEquity] = useState<Equity[]>([]);
  const [model, setModel] = useState<ModelVersion | null>(null);

  const load = useCallback(async () => {
    const db = supabase();
    const { data: { user } } = await db.auth.getUser();
    if (!user) { window.location.href = "/login"; return; }
    setEmail(user.email ?? null);

    // Every one of these is fenced by RLS: the per-user tables return only this
    // user's rows, and decisions/model_versions are readable by any signed-in user.
    const [p, d, ev, pos, eq, mv] = await Promise.all([
      db.from("profiles").select("*").eq("user_id", user.id).single(),
      db.from("decisions").select("*").order("candle_time", { ascending: false }).limit(1),
      db.from("risk_events").select("*").order("created_at", { ascending: false }).limit(200),
      db.from("positions").select("*").order("entry_time", { ascending: false }).limit(100),
      db.from("equity_snapshots").select("*").order("candle_time", { ascending: true }).limit(500),
      db.from("model_versions").select("*").eq("status", "production").limit(1),
    ]);

    setProfile(p.data ?? null);
    setDecision(d.data?.[0] ?? null);
    setEvents(ev.data ?? []);
    setPositions(pos.data ?? []);
    setEquity(eq.data ?? []);
    setModel(mv.data?.[0] ?? null);
    setRisk(
      d.data?.[0]
        ? (ev.data ?? []).find((e: RiskEvent) => e.decision_id === d.data[0].id) ?? null
        : null
    );
    setReady(true);
  }, []);

  useEffect(() => {
    load();
    // The loop writes one candle every 5 minutes; polling every 30s keeps the
    // page fresh without a realtime subscription to maintain.
    const t = setInterval(load, 30_000);
    return () => clearInterval(t);
  }, [load]);

  if (!ready)
    return <main className="min-h-dvh bg-zinc-950 p-6 text-sm text-zinc-500">Loading…</main>;

  const capital = Number(profile?.virtual_capital ?? 0);
  const latestEquity = equity.length ? Number(equity[equity.length - 1].equity) : capital;
  const totalPnl = latestEquity - capital;
  const openCount = positions.filter((p) => p.status === "OPEN").length;

  return (
    <main className="min-h-dvh bg-zinc-950 text-zinc-100">
      <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
        <header className="flex flex-wrap items-baseline justify-between gap-2">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">BTC Paper Trader</h1>
            <p className="mt-1 text-sm text-zinc-500">
              Simulated trading, virtual capital, no real orders.
            </p>
          </div>
          <div className="flex items-center gap-3 text-xs text-zinc-500">
            <span>{email}</span>
            <button
              onClick={async () => { await supabase().auth.signOut(); window.location.href = "/login"; }}
              className="rounded border border-zinc-800 px-2 py-1 hover:border-zinc-600"
            >
              Sign out
            </button>
          </div>
        </header>

        <div className="mt-6 grid gap-4">
          <Card title="Portfolio">
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <Stat label="Equity" value={usd(latestEquity)} />
              <Stat label="Total P&L" value={`${totalPnl > 0 ? "+" : ""}${usd(totalPnl)}`}
                    tone={totalPnl > 0 ? "up" : totalPnl < 0 ? "down" : undefined} />
              <Stat label="Open positions" value={String(openCount)}
                    sub={`${profile?.risk_profile ?? "—"} profile`} />
              <Stat label="Starting capital" value={usd(capital)}
                    sub={`${((profile?.trading_allocation ?? 0) * 100).toFixed(0)}% allocated`} />
            </div>
          </Card>

          <EdgePanel model={model} />
          <LatestDecision d={decision} risk={risk} />
          <EquityChart data={equity} capital={capital} />

          <div className="grid gap-4 sm:grid-cols-2">
            <Performance rows={positions} />
            <RecentRejections events={events} />
          </div>

          <Positions rows={positions} />
        </div>

        <footer className="mt-10 border-t border-zinc-900 pt-6 text-xs leading-relaxed text-zinc-500">
          <p>
            Paper trading only. This system places no real orders and holds no
            real funds. Nothing here is investment advice.
          </p>
          <p className="mt-2">
            The model was selected by walk-forward validation across multiple
            training windows and scored on trading economics rather than
            classification accuracy. It does not currently clear transaction
            costs, and the panel above shows that gap rather than hiding it.
          </p>
        </footer>
      </div>
    </main>
  );
}
