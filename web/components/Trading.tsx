"use client";

import { useState } from "react";
import { supabase } from "@/lib/supabase";
import type { Position, Profile } from "@/lib/supabase";

const usd = (n: number | null | undefined, d = 2) =>
  n === null || n === undefined ? "—"
    : `$${n.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d })}`;

export function Panel({ title, right, children }: {
  title: string; right?: React.ReactNode; children: React.ReactNode;
}) {
  return (
    <section className="rounded-lg border border-zinc-800 bg-zinc-900/40">
      <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-2.5">
        <h2 className="text-xs font-medium uppercase tracking-wider text-zinc-500">{title}</h2>
        {right}
      </div>
      <div className="p-4">{children}</div>
    </section>
  );
}

export function Portfolio({ profile, positions, price }: {
  profile: Profile | null; positions: Position[]; price: number | null;
}) {
  if (!profile) return null;

  const capital = Number(profile.virtual_capital);
  const tradingCapital = capital * Number(profile.trading_allocation);
  const closed = positions.filter((p) => p.status === "CLOSED");
  const open = positions.filter((p) => p.status === "OPEN");

  const realised = closed.reduce((a, p) => a + Number(p.pnl ?? 0), 0);
  const openNotional = open.reduce((a, p) => a + Number(p.qty) * Number(p.entry_price), 0);
  // Unrealised marks against the live price, so this moves between candles
  // rather than only when the loop writes a snapshot.
  const unrealised = price
    ? open.reduce((a, p) =>
        a + (price - Number(p.entry_price)) * Number(p.qty) * (p.side === "LONG" ? 1 : -1), 0)
    : 0;

  const available = capital + realised - openNotional;
  const totalPnl = realised + unrealised;

  const startOfDay = new Date(); startOfDay.setUTCHours(0, 0, 0, 0);
  const dailyPnl = closed
    .filter((p) => p.exit_time && new Date(p.exit_time) >= startOfDay)
    .reduce((a, p) => a + Number(p.pnl ?? 0), 0) + unrealised;

  return (
    <Panel title="Portfolio">
      <div className="grid grid-cols-2 gap-x-4 gap-y-5 sm:grid-cols-5">
        <Metric label="Virtual capital" value={usd(capital)} />
        <Metric label="Trading capital" value={usd(tradingCapital)}
                sub={`${(Number(profile.trading_allocation) * 100).toFixed(0)}% allocated`} />
        <Metric label="Available balance" value={usd(available)}
                sub={openNotional > 0 ? `${usd(openNotional)} in positions` : "nothing at risk"} />
        <Metric label="Daily P&L" value={`${dailyPnl >= 0 ? "+" : ""}${usd(dailyPnl)}`}
                tone={dailyPnl > 0 ? "up" : dailyPnl < 0 ? "down" : undefined}
                sub={`limit −${usd(Number(profile.max_daily_loss) * capital, 0)}`} />
        <Metric label="Total P&L" value={`${totalPnl >= 0 ? "+" : ""}${usd(totalPnl)}`}
                tone={totalPnl > 0 ? "up" : totalPnl < 0 ? "down" : undefined}
                sub={`${((totalPnl / capital) * 100).toFixed(2)}%`} />
      </div>
    </Panel>
  );
}

function Metric({ label, value, sub, tone }: {
  label: string; value: string; sub?: string; tone?: "up" | "down";
}) {
  return (
    <div>
      <div className="text-xs text-zinc-500">{label}</div>
      <div className={`mt-1 text-lg font-semibold tabular-nums ${
        tone === "up" ? "text-emerald-400" : tone === "down" ? "text-rose-400" : "text-zinc-100"}`}>
        {value}
      </div>
      {sub && <div className="mt-0.5 text-[11px] text-zinc-600">{sub}</div>}
    </div>
  );
}

export function OpenPositions({ positions, price }: { positions: Position[]; price: number | null }) {
  const open = positions.filter((p) => p.status === "OPEN");

  return (
    <Panel title={`Open positions (${open.length})`}>
      {open.length === 0 ? (
        <p className="text-sm text-zinc-500">
          Nothing open. The system opens a position only when a signal clears your
          mode&apos;s confidence threshold.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs tabular-nums">
            <thead className="text-zinc-500">
              <tr>
                <Th>Side</Th><Th>Size</Th><Th>Entry</Th><Th>Current</Th>
                <Th>Stop / Target</Th><Th>Strategy</Th><Th right>Unrealised</Th>
              </tr>
            </thead>
            <tbody>
              {open.map((p) => {
                const cur = price ?? Number(p.entry_price);
                const pnl = (cur - Number(p.entry_price)) * Number(p.qty) *
                            (p.side === "LONG" ? 1 : -1);
                return (
                  <tr key={p.id} className="border-t border-zinc-800/70 text-zinc-300">
                    <Td><span className={p.side === "LONG" ? "text-emerald-400" : "text-rose-400"}>
                      {p.side}</span></Td>
                    <Td>{Number(p.qty).toFixed(6)} BTC</Td>
                    <Td>{usd(Number(p.entry_price), 0)}</Td>
                    <Td>{usd(cur, 0)}</Td>
                    <Td className="text-zinc-500">
                      {usd(Number(p.stop_loss), 0)} / {usd(Number(p.take_profit), 0)}
                    </Td>
                    <Td className="text-zinc-400">{p.strategy?.replace(/_/g, " ") ?? "—"}</Td>
                    <Td right className={pnl >= 0 ? "text-emerald-400" : "text-rose-400"}>
                      {pnl >= 0 ? "+" : ""}{usd(pnl)}
                    </Td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

export function TradeHistory({ positions }: { positions: Position[] }) {
  const closed = positions.filter((p) => p.status === "CLOSED");
  const wins = closed.filter((p) => Number(p.pnl) > 0).length;
  const total = closed.reduce((a, p) => a + Number(p.pnl ?? 0), 0);

  return (
    <Panel
      title={`Trade history (${closed.length})`}
      right={closed.length > 0 ? (
        <span className="text-xs text-zinc-500">
          {((wins / closed.length) * 100).toFixed(0)}% win rate ·{" "}
          <span className={total >= 0 ? "text-emerald-400" : "text-rose-400"}>
            {total >= 0 ? "+" : ""}{usd(total)}
          </span>
        </span>
      ) : undefined}
    >
      {closed.length === 0 ? (
        <p className="text-sm text-zinc-500">No completed trades yet.</p>
      ) : (
        <div className="max-h-80 overflow-auto">
          <table className="w-full text-left text-xs tabular-nums">
            <thead className="sticky top-0 bg-zinc-900 text-zinc-500">
              <tr>
                <Th>Closed</Th><Th>Side</Th><Th>Entry</Th><Th>Exit</Th>
                <Th>Reason</Th><Th right>P&L</Th>
              </tr>
            </thead>
            <tbody>
              {closed.map((p) => (
                <tr key={p.id} className="border-t border-zinc-800/70 text-zinc-300">
                  <Td className="text-zinc-500">
                    {p.exit_time ? new Date(p.exit_time).toLocaleString([], {
                      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—"}
                  </Td>
                  <Td><span className={p.side === "LONG" ? "text-emerald-400" : "text-rose-400"}>
                    {p.side}</span></Td>
                  <Td>{usd(Number(p.entry_price), 0)}</Td>
                  <Td>{usd(Number(p.exit_price), 0)}</Td>
                  <Td className="text-zinc-500">
                    {p.exit_reason?.replace(/_/g, " ").toLowerCase() ?? "—"}
                  </Td>
                  <Td right className={Number(p.pnl) >= 0 ? "text-emerald-400" : "text-rose-400"}>
                    {Number(p.pnl) >= 0 ? "+" : ""}{usd(Number(p.pnl))}
                    <span className="ml-1 text-zinc-600">
                      ({(Number(p.pnl_pct) * 100).toFixed(2)}%)
                    </span>
                  </Td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

const MODES = [
  { id: "conservative", icon: "🛡️", name: "Conservative",
    ai: "Strategy and trades within low-risk limits" },
  { id: "balanced", icon: "⚖️", name: "Balanced",
    ai: "Strategy and trades within medium-risk limits" },
  { id: "aggressive", icon: "🚀", name: "Aggressive",
    ai: "Strategy and trades within high-risk limits" },
  { id: "ai_autonomous", icon: "🤖", name: "AI Autonomous",
    ai: "Allocation, strategy, entry, exit and position size" },
  { id: "max_winrate", icon: "⚠️", name: "Max Win Rate",
    ai: "Tiny targets, wide stops — 80% of trades win, the account still loses" },
] as const;

const DURATIONS = [
  { label: "1 day", hours: 24 },
  { label: "1 week", hours: 24 * 7 },
  { label: "1 month", hours: 24 * 30 },
  { label: "Until I change it", hours: 0 },
];

/** Shown whenever max_winrate is selected. The mode is real and selectable;
 *  hiding what it does would make the dashboard dishonest. */
function WinRateWarning() {
  return (
    <div className="mt-3 rounded border border-rose-900/60 bg-rose-950/25 p-3 text-xs
                    leading-relaxed text-rose-200">
      <div className="font-medium">This mode is a demonstration, not a strategy.</div>
      <p className="mt-1.5 text-rose-300/90">
        Measured over 90 days out of sample: <strong>80.1% of trades closed in
        profit</strong> and the account still <strong>lost 17.6%</strong>. Each win
        takes 0.20%; each loss gives back 5.00%. Two losses erase eight wins.
      </p>
      <p className="mt-1.5 text-rose-300/90">
        A high win rate can be manufactured by moving the take-profit closer and the
        stop further away. It says nothing about whether a system makes money. Watch
        the equity curve, not the percentage.
      </p>
    </div>
  );
}

export function ModeSelector({ profile, onSaved }: {
  profile: Profile | null; onSaved: () => void;
}) {
  const [saving, setSaving] = useState(false);
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState(profile?.risk_profile ?? "balanced");
  const [hours, setHours] = useState(0);
  const [capital, setCapital] = useState(String(profile?.virtual_capital ?? 10000));
  const [alloc, setAlloc] = useState(String((profile?.trading_allocation ?? 0.5) * 100));
  const [maxLoss, setMaxLoss] = useState(String((profile?.max_daily_loss ?? 0.02) * 100));

  const current = MODES.find((m) => m.id === (profile?.risk_profile ?? "balanced"))!;

  async function save() {
    setSaving(true);
    const until = hours
      ? new Date(Date.now() + hours * 3600_000).toISOString()
      : null;
    await supabase().from("profiles").update({
      risk_profile: mode,
      virtual_capital: Number(capital),
      trading_allocation: Number(alloc) / 100,
      max_daily_loss: Number(maxLoss) / 100,
      preference_until: until,
    }).eq("user_id", profile!.user_id);
    setSaving(false);
    setOpen(false);
    onSaved();
  }

  return (
    <Panel
      title="Trading mode"
      right={
        <button onClick={() => setOpen(!open)}
                className="text-xs text-zinc-400 hover:text-zinc-100">
          {open ? "Cancel" : "Change"}
        </button>
      }
    >
      {!open ? (
        <div className="flex items-center gap-3">
          <span className="text-2xl">{current.icon}</span>
          <div>
            <div className="text-sm font-medium text-zinc-100">{current.name}</div>
            <div className="text-xs text-zinc-500">AI controls: {current.ai}</div>
            {profile?.risk_profile === "max_winrate" && <WinRateWarning />}
            {profile?.preference_until && (
              <div className="mt-0.5 text-[11px] text-amber-500">
                Active until {new Date(profile.preference_until).toLocaleString()}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="grid gap-2 sm:grid-cols-2">
            {MODES.map((m) => (
              <button key={m.id} onClick={() => setMode(m.id)}
                className={`rounded-md border p-3 text-left transition ${
                  mode === m.id ? "border-zinc-500 bg-zinc-800/60"
                                : "border-zinc-800 hover:border-zinc-700"}`}>
                <div className="flex items-center gap-2">
                  <span>{m.icon}</span>
                  <span className="text-sm font-medium text-zinc-100">{m.name}</span>
                </div>
                <p className="mt-1 text-[11px] leading-snug text-zinc-500">
                  You set {m.id === "ai_autonomous" ? "capital and max loss" : "capital and allocation"}.
                  AI controls {m.ai.toLowerCase()}.
                </p>
              </button>
            ))}
          </div>

          <div className="grid gap-3 sm:grid-cols-3">
            <Field label="Virtual capital ($)" value={capital} onChange={setCapital} />
            {mode !== "ai_autonomous" && (
              <Field label="Allocation (%)" value={alloc} onChange={setAlloc} />
            )}
            <Field label="Max daily loss (%)" value={maxLoss} onChange={setMaxLoss} />
          </div>

          <div>
            <div className="mb-1.5 text-xs text-zinc-500">Follow this preference for</div>
            <div className="flex flex-wrap gap-1.5">
              {DURATIONS.map((d) => (
                <button key={d.label} onClick={() => setHours(d.hours)}
                  className={`rounded px-2.5 py-1 text-xs ${
                    hours === d.hours ? "bg-zinc-700 text-zinc-100"
                                      : "border border-zinc-800 text-zinc-400 hover:text-zinc-200"}`}>
                  {d.label}
                </button>
              ))}
            </div>
          </div>

          {mode === "max_winrate" && <WinRateWarning />}

          <button onClick={save} disabled={saving}
            className="rounded-md bg-zinc-100 px-4 py-2 text-sm font-medium text-zinc-900
                       hover:bg-white disabled:opacity-50">
            {saving ? "Saving…" : "Save"}
          </button>
        </div>
      )}
    </Panel>
  );
}

function Field({ label, value, onChange }: {
  label: string; value: string; onChange: (v: string) => void;
}) {
  return (
    <label className="block">
      <span className="text-xs text-zinc-500">{label}</span>
      <input value={value} onChange={(e) => onChange(e.target.value)} inputMode="decimal"
        className="mt-1 w-full rounded-md border border-zinc-800 bg-zinc-950 px-2.5 py-1.5
                   text-sm tabular-nums outline-none focus:border-zinc-600" />
    </label>
  );
}

const Th = ({ children, right }: { children: React.ReactNode; right?: boolean }) => (
  <th className={`px-1 py-1.5 font-medium ${right ? "text-right" : ""}`}>{children}</th>
);
const Td = ({ children, right, className = "" }: {
  children: React.ReactNode; right?: boolean; className?: string;
}) => <td className={`px-1 py-2 ${right ? "text-right" : ""} ${className}`}>{children}</td>;
