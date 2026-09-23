"use client";

import {
  Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { Decision, Equity, ModelVersion, Position, RiskEvent } from "@/lib/supabase";

const pct = (n: number | null | undefined, d = 2) =>
  n === null || n === undefined ? "—" : `${(n * 100).toFixed(d)}%`;
const usd = (n: number | null | undefined, d = 2) =>
  n === null || n === undefined ? "—" : `$${n.toLocaleString(undefined, {
    minimumFractionDigits: d, maximumFractionDigits: d })}`;

export function Card({ title, children, tone = "default" }: {
  title: string; children: React.ReactNode; tone?: "default" | "warn";
}) {
  return (
    <section className={`rounded-lg border p-4 ${
      tone === "warn" ? "border-amber-900/60 bg-amber-950/20" : "border-zinc-800 bg-zinc-900/40"}`}>
      <h2 className="text-xs font-medium uppercase tracking-wider text-zinc-500">{title}</h2>
      <div className="mt-3">{children}</div>
    </section>
  );
}

export function Stat({ label, value, sub, tone }: {
  label: string; value: string; sub?: string; tone?: "up" | "down";
}) {
  return (
    <div>
      <div className="text-xs text-zinc-500">{label}</div>
      <div className={`mt-1 text-xl font-semibold tabular-nums ${
        tone === "up" ? "text-emerald-400" : tone === "down" ? "text-rose-400" : "text-zinc-100"}`}>
        {value}
      </div>
      {sub && <div className="mt-0.5 text-xs text-zinc-500">{sub}</div>}
    </div>
  );
}

/**
 * The centrepiece. The model's gross edge per trade is smaller than the round
 * trip cost, so the honest thing is to show that arithmetic rather than a
 * prediction alone -- otherwise the UI implies a profitable signal that the
 * walk-forward says does not exist.
 */
export function EdgePanel({ model }: { model: ModelVersion | null }) {
  if (!model) return <Card title="Model"><p className="text-sm text-zinc-500">No production model.</p></Card>;
  const edge = model.metrics.gross_edge ?? 0;
  const cost = model.metrics.cost ?? 0.0025;
  const net = edge - cost;
  const clears = net > 0;

  return (
    <Card title="Does the edge cover the fees?" tone={clears ? "default" : "warn"}>
      <div className="grid grid-cols-3 gap-4">
        <Stat label="Gross edge / trade" value={pct(edge, 3)} tone="up" />
        <Stat label="Round-trip cost" value={`−${pct(cost, 3)}`} tone="down" />
        <Stat label="Net" value={pct(net, 3)} tone={clears ? "up" : "down"} />
      </div>

      <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-zinc-800">
        <div className="h-full bg-emerald-500"
             style={{ width: `${Math.min(100, (edge / cost) * 100)}%` }} />
      </div>
      <p className="mt-2 text-xs text-zinc-400">
        The model captures{" "}
        <span className="font-medium text-zinc-200">{((edge / cost) * 100).toFixed(0)}%</span>{" "}
        of what a round trip costs. It needs more than 100% to make money.
      </p>

      <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-1 text-xs">
        <Row k="Model" v={model.version} />
        <Row k="Trained on" v={model.window_days ? `${model.window_days} days` : "—"} />
        <Row k="Walk-forward" v={`${model.metrics.folds_profitable ?? 0}/${
          model.metrics.n_folds ?? 0} folds profitable`} />
        <Row k="Horizon" v={model.metrics.horizon ? `${(model.metrics.horizon * 5) / 60}h` : "—"} />
      </dl>

      {model.notes?.includes("FORCED") && (
        <p className="mt-3 rounded border border-amber-900/60 bg-amber-950/30 p-2 text-xs text-amber-300">
          This model did not pass the promotion gate. It is running so the system
          can be demonstrated end to end, and that override is recorded in the
          database. It is expected to decline most signals.
        </p>
      )}
    </Card>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <>
      <dt className="text-zinc-500">{k}</dt>
      <dd className="text-right tabular-nums text-zinc-300">{v}</dd>
    </>
  );
}

export function LatestDecision({ d, risk }: { d: Decision | null; risk: RiskEvent | null }) {
  if (!d) return <Card title="Latest decision"><p className="text-sm text-zinc-500">Waiting for the first candle.</p></Card>;
  const tone = d.prediction === "UP" ? "up" : d.prediction === "DOWN" ? "down" : undefined;

  return (
    <Card title="Why the system did what it did">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="BTC" value={usd(d.price, 0)} />
        <Stat label="Prediction" value={d.prediction} sub={pct(d.confidence, 1)} tone={tone} />
        <Stat label="Regime" value={d.regime.replace(/_/g, " ")} />
        <Stat label="Action" value={d.action}
              tone={d.action === "BUY" ? "up" : d.action === "SELL" ? "down" : undefined} />
      </div>

      <ol className="mt-5 space-y-2 text-sm">
        <Step n={1} label="Model predicts"
              detail={`${d.prediction} at ${pct(d.confidence, 1)} confidence`} />
        <Step n={2} label="Regime identified" detail={d.regime.replace(/_/g, " ")} />
        <Step n={3} label={`Strategy chosen by ${d.decided_by === "jev" ? "Jev" : "fallback table"}`}
              detail={d.strategy.replace(/_/g, " ")}
              note={d.decided_by === "fallback"
                ? "Jev was unreachable or rate limited, so the deterministic table decided."
                : undefined} />
        <Step n={4} label="Risk engine"
              detail={risk ? (risk.approved ? "approved" : risk.rule.replace(/_/g, " ")) : "no verdict recorded"}
              note={risk?.detail ?? undefined} />
      </ol>
    </Card>
  );
}

function Step({ n, label, detail, note }: {
  n: number; label: string; detail: string; note?: string;
}) {
  return (
    <li className="flex gap-3">
      <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full
                       bg-zinc-800 text-[10px] text-zinc-400">{n}</span>
      <div>
        <span className="text-zinc-400">{label}: </span>
        <span className="text-zinc-100">{detail}</span>
        {note && <div className="mt-0.5 text-xs text-zinc-500">{note}</div>}
      </div>
    </li>
  );
}

export function EquityChart({ data, capital }: { data: Equity[]; capital: number }) {
  if (data.length < 2)
    return <Card title="Equity"><p className="text-sm text-zinc-500">
      Not enough history yet. The curve appears once a few candles have been processed.
    </p></Card>;

  const rows = data.map((e) => ({
    t: new Date(e.candle_time).getTime(),
    equity: Number(e.equity),
  }));

  return (
    <Card title="Equity curve">
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={rows} margin={{ top: 4, right: 4, left: 4, bottom: 4 }}>
            <defs>
              <linearGradient id="eq" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#34d399" stopOpacity={0.35} />
                <stop offset="100%" stopColor="#34d399" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#27272a" vertical={false} />
            <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} scale="time"
                   tickFormatter={(t) => new Date(t).toLocaleTimeString([], {
                     hour: "2-digit", minute: "2-digit" })}
                   stroke="#52525b" fontSize={11} tickLine={false} axisLine={false} />
            <YAxis domain={["auto", "auto"]} stroke="#52525b" fontSize={11}
                   tickLine={false} axisLine={false} width={64}
                   tickFormatter={(v) => `$${Math.round(v).toLocaleString()}`} />
            <Tooltip
              contentStyle={{ background: "#18181b", border: "1px solid #3f3f46",
                              borderRadius: 6, fontSize: 12 }}
              labelFormatter={(t) => new Date(t as number).toLocaleString()}
              formatter={(v) => [usd(Number(v)), "equity"]} />
            <Area type="monotone" dataKey="equity" stroke="#34d399" strokeWidth={1.5}
                  fill="url(#eq)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-2 text-xs text-zinc-500">Starting capital {usd(capital, 0)}.</p>
    </Card>
  );
}

export function Positions({ rows }: { rows: Position[] }) {
  const open = rows.filter((r) => r.status === "OPEN");
  const closed = rows.filter((r) => r.status === "CLOSED");

  return (
    <Card title={`Positions (${open.length} open, ${closed.length} closed)`}>
      {rows.length === 0 ? (
        <p className="text-sm text-zinc-500">
          No positions yet. The risk engine declines any signal below your
          confidence gate, which is most of them.
        </p>
      ) : (
        <div className="-mx-1 overflow-x-auto">
          <table className="w-full text-left text-xs tabular-nums">
            <thead className="text-zinc-500">
              <tr>
                <th className="px-1 py-1 font-medium">Side</th>
                <th className="px-1 py-1 font-medium">Entry</th>
                <th className="px-1 py-1 font-medium">SL / TP</th>
                <th className="px-1 py-1 font-medium">Strategy</th>
                <th className="px-1 py-1 font-medium">Exit</th>
                <th className="px-1 py-1 text-right font-medium">P&amp;L</th>
              </tr>
            </thead>
            <tbody className="text-zinc-300">
              {rows.slice(0, 25).map((p) => (
                <tr key={p.id} className="border-t border-zinc-800/70">
                  <td className={`px-1 py-1.5 font-medium ${
                    p.side === "LONG" ? "text-emerald-400" : "text-rose-400"}`}>{p.side}</td>
                  <td className="px-1 py-1.5">{usd(p.entry_price, 0)}</td>
                  <td className="px-1 py-1.5 text-zinc-500">
                    {usd(p.stop_loss, 0)} / {usd(p.take_profit, 0)}
                  </td>
                  <td className="px-1 py-1.5 text-zinc-400">
                    {p.strategy?.replace(/_/g, " ") ?? "—"}
                  </td>
                  <td className="px-1 py-1.5 text-zinc-500">
                    {p.exit_reason?.replace(/_/g, " ").toLowerCase() ?? "open"}
                  </td>
                  <td className={`px-1 py-1.5 text-right ${
                    (p.pnl ?? 0) > 0 ? "text-emerald-400"
                      : (p.pnl ?? 0) < 0 ? "text-rose-400" : "text-zinc-500"}`}>
                    {p.pnl === null ? "—" : `${p.pnl > 0 ? "+" : ""}${usd(p.pnl)}`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

export function RecentRejections({ events }: { events: RiskEvent[] }) {
  const counts = events.reduce<Record<string, number>>((acc, e) => {
    if (!e.approved) acc[e.rule] = (acc[e.rule] ?? 0) + 1;
    return acc;
  }, {});
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);

  return (
    <Card title="Why trades were declined">
      {sorted.length === 0 ? (
        <p className="text-sm text-zinc-500">Nothing declined yet.</p>
      ) : (
        <ul className="space-y-1.5 text-sm">
          {sorted.map(([rule, n]) => (
            <li key={rule} className="flex items-baseline justify-between gap-3">
              <span className="text-zinc-300">{rule.replace(/_/g, " ")}</span>
              <span className="text-zinc-500 tabular-nums">{n}</span>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

export function Performance({ rows }: { rows: Position[] }) {
  const closed = rows.filter((r) => r.status === "CLOSED" && r.pnl !== null);
  if (closed.length === 0)
    return <Card title="Performance"><p className="text-sm text-zinc-500">
      No closed trades yet.</p></Card>;

  const pnls = closed.map((c) => Number(c.pnl));
  const wins = pnls.filter((p) => p > 0);
  const losses = pnls.filter((p) => p <= 0);
  const total = pnls.reduce((a, b) => a + b, 0);
  const pf = losses.length
    ? wins.reduce((a, b) => a + b, 0) / Math.abs(losses.reduce((a, b) => a + b, 0))
    : Infinity;

  // Peak-to-trough on the realised curve, seeded at 0 so an opening loss counts.
  let run = 0, peak = 0, dd = 0;
  for (const p of pnls) { run += p; peak = Math.max(peak, run); dd = Math.min(dd, run - peak); }

  return (
    <Card title="Performance">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="Trades" value={String(closed.length)} />
        <Stat label="Win rate" value={`${((wins.length / closed.length) * 100).toFixed(0)}%`} />
        <Stat label="Total P&L" value={`${total > 0 ? "+" : ""}${usd(total)}`}
              tone={total > 0 ? "up" : total < 0 ? "down" : undefined} />
        <Stat label="Max drawdown" value={usd(dd)} tone={dd < 0 ? "down" : undefined} />
      </div>
      <p className="mt-3 text-xs text-zinc-500">
        Profit factor {pf === Infinity ? "∞" : pf.toFixed(2)}. All figures are net
        of a {pct(0.0025, 2)} round-trip cost.
      </p>
    </Card>
  );
}
