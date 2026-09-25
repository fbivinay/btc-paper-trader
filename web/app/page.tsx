"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import EquityChart from "@/components/EquityChart";
import { all, db, type Alert, type Decision, type EquityRow, type Run, type SlabRow, type Trade } from "@/lib/supabase";
import { CAPITAL, money, pct, stats, type Stats } from "@/lib/stats";

type View = "net" | "gross";
const VIEW_LABEL: Record<View, string> = { net: "After tax & charges", gross: "Before tax & charges" };
// Indian slab rates; the stored values include the 4% cess. 31.2% is the default in etf_equity.
const SLABS: [number, string][] = [[0.312, "30%"], [0.26, "25%"], [0.208, "20%"], [0.156, "15%"],
                                   [0.104, "10%"], [0.052, "5%"]];

export default function Dashboard() {
  const [eq, setEq] = useState<EquityRow[]>([]);
  const [dec, setDec] = useState<Decision[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [firstLive, setFirstLive] = useState<string | null>(null);
  const [view, setView] = useState<View>("net");
  const [slab, setSlab] = useState(0.312);
  const [slabRows, setSlabRows] = useState<SlabRow[] | null>(null);
  const [lastAlert, setLastAlert] = useState<Alert | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [e, d, t, r, l, a] = await Promise.all([
          all<EquityRow>("etf_equity", "date"),
          db.from("etf_decisions").select("*").order("date", { ascending: false }).limit(2),
          db.from("etf_trades").select("*").order("date", { ascending: false }).limit(500),
          db.from("etf_runs").select("*").order("started_at", { ascending: false }).limit(5),
          db.from("etf_decisions").select("date").eq("live", true).order("date").limit(1),
          db.from("etf_alerts").select("*").order("date", { ascending: false }).limit(1),
        ]);
        setEq(e);
        setDec((d.data ?? []) as Decision[]);
        setTrades((t.data ?? []) as Trade[]);
        setRuns((r.data ?? []) as Run[]);
        setFirstLive(l.data?.[0]?.date ?? null);
        setLastAlert((a.data?.[0] ?? null) as Alert | null);
        setError(null);
      } catch (x) {
        setError(x instanceof Error ? x.message : String(x));
      }
    }
    load();
    const t = setInterval(load, 300_000);
    return () => clearInterval(t);
  }, []);

  // Other tax slabs load on demand; the default top slab is already in etf_equity.
  useEffect(() => {
    if (slab === 0.312) return;
    let alive = true;
    all<SlabRow>("etf_equity_slab", "date", { slab })
      .then((rows) => { if (alive) setSlabRows(rows); })
      .catch((x) => setError(x instanceof Error ? x.message : String(x)));
    return () => { alive = false; };
  }, [slab]);

  const series = useMemo(() => {
    const dates = eq.map((r) => r.date);
    const col = (k: keyof EquityRow) => eq.map((r) => Number(r[k]));
    // Rows for the chosen slab, once they have arrived; the top slab lives in etf_equity itself.
    const rows = slab !== 0.312 && slabRows?.length && Math.abs(Number(slabRows[0].slab) - slab) < 1e-4
      ? slabRows : null;
    const bySlab = new Map((rows ?? []).map((r) => [r.date, r]));
    const net = (k: "model_net" | "hold_net") =>
      rows ? eq.map((r) => Number(bySlab.get(r.date)?.[k] ?? r[k])) : col(k);
    return { dates, model_net: net("model_net"), model_gross: col("model_gross"),
             hold_net: net("hold_net"), hold_gross: col("hold_gross") };
  }, [eq, slabRows, slab]);

  const lines = useMemo(() => [
    { label: "Model", color: "#34d399", dates: series.dates,
      values: view === "net" ? series.model_net : series.model_gross },
    { label: "Buy & hold IBIT", color: "#a1a1aa", dates: series.dates,
      values: view === "net" ? series.hold_net : series.hold_gross },
  ], [series, view]);

  if (error)
    return <Shell><p className="text-sm text-rose-400">Could not load data: {error}</p></Shell>;
  if (!eq.length || !dec.length)
    return <Shell><p className="text-sm text-zinc-500">Loading…</p></Shell>;

  const s = {
    model_net: stats(series.dates, series.model_net),
    model_gross: stats(series.dates, series.model_gross),
    hold_net: stats(series.dates, series.hold_net),
    hold_gross: stats(series.dates, series.hold_gross),
  };
  const d = dec[0];
  const latest = (["decide", "execute"] as const)
    .map((j) => runs.find((r) => r.job === j)).filter((r): r is Run => !!r);

  return (
    <Shell runs={latest} lastClose={d.date}>
      <Today d={d} prev={dec[1]} alert={lastAlert} />

      <Section title="Results" note={`$${CAPITAL.toLocaleString()} each, from IBIT's launch on ${fmtDate(series.dates[0])} to ${fmtDate(d.date)}`}>
        <div className="mb-3 flex gap-1">
          {(["net", "gross"] as View[]).map((v) => (
            <button key={v} onClick={() => setView(v)}
              className={`rounded px-3 py-1.5 text-xs ${view === v ? "bg-zinc-700 text-zinc-100" : "text-zinc-400 hover:text-zinc-200"}`}>
              {VIEW_LABEL[v]}
            </button>
          ))}
        </div>
        {view === "net" && (
          <div className="mb-3 flex flex-wrap items-center gap-1 text-xs">
            <span className="mr-1 text-zinc-500">Your income-tax slab</span>
            {SLABS.map(([v, label]) => (
              <button key={v} onClick={() => setSlab(v)}
                className={`rounded px-2 py-1 ${slab === v ? "bg-zinc-700 text-zinc-100" : "text-zinc-400 hover:text-zinc-200"}`}>
                {label}
              </button>
            ))}
          </div>
        )}
        <div className="grid gap-3 sm:grid-cols-2">
          <StatCard title="Model" color="text-emerald-400" st={view === "net" ? s.model_net : s.model_gross}
                    end={(view === "net" ? series.model_net : series.model_gross).at(-1)!} />
          <StatCard title="Buy & hold IBIT" color="text-zinc-300" st={view === "net" ? s.hold_net : s.hold_gross}
                    end={(view === "net" ? series.hold_net : series.hold_gross).at(-1)!} />
        </div>
        <div className="mt-4 rounded-md border border-zinc-800">
          <EquityChart lines={lines} />
        </div>
        <p className="mt-2 text-xs text-zinc-500">
          {view === "net"
            ? `After tax & charges: what you would get back in hand if you sold everything that day — gains held under 24 months taxed at your slab (${(slab * 100).toFixed(1)}% with cess), 13% after 24 months, 1.5% forex each way, 0.05% per trade, T-bill income taxed at slab. The slab is your marginal rate; ask a CA which applies to you.`
            : "Before tax & charges: the same trades with no tax, no forex markup and no trading costs. The gap between the two views is what India and the middlemen take."}
        </p>
      </Section>

      <Section title="All four side by side" note="Calendar-year returns; 2024 starts at IBIT's launch, the latest year is year-to-date">
        <Comparison s={s} slab={SLABS.find(([v]) => v === slab)![1]} />
        <p className="mt-2 text-xs text-zinc-500">
          After-tax values are what you would get if you sold that day, so a year can look better after tax
          than before: once IBIT has been held 24 months (from January 2026 for buy &amp; hold) the tax on selling
          falls from your slab rate to 13%, and the saving shows up in that year.
        </p>
      </Section>

      <Section title="Buy & hold IBIT" note="The benchmark: buy on launch day and never sell">
        <BuyHold net={s.hold_net} gross={s.hold_gross} model={s.model_net} />
      </Section>

      <Section title="Your money" note="Today's split for any amount, at the latest close">
        <Calculator d={d} />
      </Section>

      <Section title="Trades" note={`${trades.length} trades by the model portfolio. Gains are realised, before tax`}>
        <Trades rows={trades} />
      </Section>

      <Section title="How the model decides" note="Once a day, after the US close; traded at the next open">
        <HowItWorks firstLive={firstLive} />
      </Section>
    </Shell>
  );
}

/* ------------------------------------------------------------------ pieces */

function Shell({ children, runs = [], lastClose }: { children: React.ReactNode; runs?: Run[]; lastClose?: string }) {
  return (
    <main className="min-h-dvh bg-zinc-950 text-zinc-100">
      <header className="border-b border-zinc-900">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-2 px-4 py-3">
          <div className="flex flex-wrap items-baseline gap-2">
            <span className="text-sm font-semibold tracking-tight">Bitcoin ETF Trend Model</span>
            <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-zinc-400">
              paper · real prices
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-500">
            {lastClose && <span>data to {fmtDate(lastClose)} close</span>}
            {runs.map((r) => <Heartbeat key={r.id} run={r} />)}
            <Link href="/go-live" className="rounded border border-zinc-800 px-2 py-1 text-zinc-300 hover:border-zinc-600">
              Go live with real money
            </Link>
          </div>
        </div>
      </header>
      <div className="mx-auto max-w-5xl space-y-4 px-4 py-5">
        {children}
        <p className="pb-6 text-xs leading-relaxed text-zinc-600">
          Paper portfolio on real market prices. Not investment advice. Tax figures follow the Indian rules
          for foreign ETFs as understood at the time of writing; confirm them with a Chartered Accountant
          before putting real money in. Past results do not promise future ones.
        </p>
      </div>
    </main>
  );
}

function Heartbeat({ run }: { run: Run }) {
  const tone = run.status === "error" ? "bg-rose-500" : run.status === "ok" ? "bg-emerald-500" : "bg-amber-400";
  return (
    <span className="flex items-center gap-1.5" title={run.detail ?? ""}>
      <span className={`h-2 w-2 rounded-full ${tone}`} />
      {run.job} {run.status}, {ago(run.started_at)}
    </span>
  );
}

function Section({ title, note, children }: { title: string; note?: string; children: React.ReactNode }) {
  return (
    <section className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-4">
      <div className="mb-3">
        <h2 className="text-sm font-medium text-zinc-200">{title}</h2>
        {note && <p className="text-xs text-zinc-500">{note}</p>}
      </div>
      {children}
    </section>
  );
}

function Today({ d, prev, alert }: { d: Decision; prev?: Decision; alert: Alert | null }) {
  const parts = [
    { k: "IBIT (Bitcoin)", w: d.w_btc, was: prev?.w_btc, c: "bg-amber-500" },
    { k: "GLD (gold)", w: d.w_gold, was: prev?.w_gold, c: "bg-yellow-300" },
    { k: "T-bills", w: d.w_cash, was: prev?.w_cash, c: "bg-sky-500" },
  ];
  return (
    <Section title="Today's decision"
             note={`From the ${fmtDate(d.date)} close, traded at the next open${d.live ? " · recorded live" : ""}`}>
      <div className="flex h-7 w-full overflow-hidden rounded">
        {parts.map((p) => p.w > 0.005 && (
          <div key={p.k} className={`${p.c} flex items-center justify-center text-[11px] font-medium text-zinc-950`}
               style={{ width: `${p.w * 100}%` }}>
            {p.w >= 0.08 ? `${Math.round(p.w * 100)}%` : ""}
          </div>
        ))}
      </div>
      <div className="mt-3 grid gap-3 sm:grid-cols-3">
        {parts.map((p) => (
          <div key={p.k} className="rounded-md border border-zinc-800 px-3 py-2">
            <div className="text-xs text-zinc-500">{p.k}</div>
            <div className="text-xl font-semibold tabular-nums">{(p.w * 100).toFixed(0)}%</div>
            {p.was !== undefined && Math.abs(p.w - p.was) > 0.005 && (
              <div className="text-xs text-zinc-500">was {(p.was * 100).toFixed(0)}%</div>
            )}
          </div>
        ))}
      </div>
      <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-xs text-zinc-400">
        <span>Bitcoin trend votes <b className="text-zinc-200">{d.btc_votes}/8</b></span>
        <span>Bitcoin volatility sizing <b className="text-zinc-200">×{Number(d.btc_vol).toFixed(2)}</b></span>
        <span>Gold trend votes <b className="text-zinc-200">{d.gold_votes}/8</b></span>
        <span>Gold volatility sizing <b className="text-zinc-200">×{Number(d.gold_vol).toFixed(2)}</b></span>
        {d.tbill !== null && <span>T-bill yield <b className="text-zinc-200">{(d.tbill * 100).toFixed(2)}%</b></span>}
      </div>
      <p className="mt-3 text-xs text-zinc-500">
        {alert
          ? <>Last rebalance alert: {fmtDate(alert.date)} — IBIT {(alert.w_btc * 100).toFixed(0)}%, gold{" "}
              {(alert.w_gold * 100).toFixed(0)}%, T-bills {(alert.w_cash * 100).toFixed(0)}%.{" "}</>
          : <>No rebalance alert sent yet; the first comes with the first live decision.{" "}</>}
        Trading by hand in any broker app?{" "}
        <Link href="/go-live#alerts" className="underline decoration-zinc-700 hover:text-zinc-300">
          Get these on your phone
        </Link>.
      </p>
    </Section>
  );
}

function StatCard({ title, color, st, end }: { title: string; color: string; st: Stats; end: number }) {
  return (
    <div className="rounded-md border border-zinc-800 px-3 py-3">
      <div className={`text-xs ${color}`}>{title}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums">{money(end)}</div>
      <div className="mt-1 grid grid-cols-3 gap-2 text-xs">
        <Kv k="total" v={pct(st.total)} />
        <Kv k="per year" v={pct(st.perYear)} />
        <Kv k="worst drop" v={pct(st.worstDrop)} />
      </div>
    </div>
  );
}

function Kv({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <div className="text-zinc-500">{k}</div>
      <div className="tabular-nums text-zinc-200">{v}</div>
    </div>
  );
}

function Comparison({ s, slab }: { s: Record<"model_net" | "model_gross" | "hold_net" | "hold_gross", Stats>; slab: string }) {
  const rows: [string, Stats][] = [
    [`Model — after tax & charges (${slab} slab)`, s.model_net],
    ["Model — before tax & charges", s.model_gross],
    [`Buy & hold — after tax & charges (${slab} slab)`, s.hold_net],
    ["Buy & hold — before tax & charges", s.hold_gross],
  ];
  const years = s.model_net.years.map(([y]) => y);
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] text-sm">
        <thead className="text-xs text-zinc-500">
          <tr>
            <th className="py-1 text-left font-normal"></th>
            <th className="py-1 text-right font-normal">per year</th>
            <th className="py-1 text-right font-normal">worst drop</th>
            {years.map((y) => <th key={y} className="py-1 text-right font-normal">{y}</th>)}
          </tr>
        </thead>
        <tbody className="tabular-nums">
          {rows.map(([name, st]) => (
            <tr key={name} className="border-t border-zinc-800">
              <td className="py-1.5 pr-2 text-zinc-300">{name}</td>
              <td className="py-1.5 text-right">{pct(st.perYear)}</td>
              <td className="py-1.5 text-right text-rose-300">{pct(st.worstDrop, 0)}</td>
              {st.years.map(([y, r]) => (
                <td key={y} className={`py-1.5 text-right ${r < 0 ? "text-rose-400" : "text-emerald-400"}`}>{pct(r, 0)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function BuyHold({ net, gross, model }: { net: Stats; gross: Stats; model: Stats }) {
  const tax = gross.perYear - net.perYear;
  return (
    <div className="grid gap-3 text-sm sm:grid-cols-3">
      <Fact k="After tax & charges" v={`${pct(net.perYear)} a year`} sub={`worst drop ${pct(net.worstDrop, 0)}`} />
      <Fact k="Before tax & charges" v={`${pct(gross.perYear)} a year`} sub={`tax and charges cost ${(tax * 100).toFixed(1)} points a year`} />
      <Fact k="Model vs buy & hold (after tax)" v={`${pct(model.perYear - net.perYear)} a year`}
            sub={`worst drop ${pct(model.worstDrop, 0)} vs ${pct(net.worstDrop, 0)}`} />
    </div>
  );
}

function Fact({ k, v, sub }: { k: string; v: string; sub: string }) {
  return (
    <div className="rounded-md border border-zinc-800 px-3 py-2">
      <div className="text-xs text-zinc-500">{k}</div>
      <div className="text-lg font-semibold tabular-nums">{v}</div>
      <div className="text-xs text-zinc-500">{sub}</div>
    </div>
  );
}

function Calculator({ d }: { d: Decision }) {
  const [inr, setInr] = useState(100000);
  const rate = Number(d.usdinr) || 0;
  const usd = rate ? inr / rate : 0;
  const rows = [
    { k: "IBIT (Bitcoin ETF)", w: d.w_btc, px: Number(d.btc_close) },
    { k: "GLD (gold ETF)", w: d.w_gold, px: Number(d.gold_close) },
    { k: "T-bills (e.g. SGOV)", w: d.w_cash, px: 0 },
  ];
  return (
    <div>
      <label className="flex flex-wrap items-center gap-2 text-sm">
        <span className="text-zinc-400">Amount (₹)</span>
        <input type="number" min={0} step={1000} value={inr}
               onChange={(e) => setInr(Math.max(0, Number(e.target.value)))}
               className="w-40 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 tabular-nums" />
        <span className="text-xs text-zinc-500">≈ ${usd.toLocaleString("en-US", { maximumFractionDigits: 0 })} at ₹{rate.toFixed(2)}/$</span>
      </label>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[420px] text-sm">
          <thead className="text-xs text-zinc-500">
            <tr><th className="text-left font-normal">Hold</th><th className="text-right font-normal">share</th>
              <th className="text-right font-normal">₹</th><th className="text-right font-normal">$</th>
              <th className="text-right font-normal">≈ units</th></tr>
          </thead>
          <tbody className="tabular-nums">
            {rows.map((r) => (
              <tr key={r.k} className="border-t border-zinc-800">
                <td className="py-1.5 text-zinc-300">{r.k}</td>
                <td className="py-1.5 text-right">{(r.w * 100).toFixed(0)}%</td>
                <td className="py-1.5 text-right">{money(inr * r.w, "₹")}</td>
                <td className="py-1.5 text-right">{money(usd * r.w)}</td>
                <td className="py-1.5 text-right">{r.px ? (usd * r.w / r.px).toFixed(2) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-zinc-500">
        Fractional units are possible with most US brokers. This is the model&rsquo;s split, not a recommendation.
      </p>
    </div>
  );
}

function Trades({ rows }: { rows: Trade[] }) {
  const [n, setN] = useState(15);
  return (
    <div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[560px] text-sm">
          <thead className="text-xs text-zinc-500">
            <tr><th className="text-left font-normal">Date</th><th className="text-left font-normal">ETF</th>
              <th className="text-left font-normal">Side</th><th className="text-right font-normal">Units</th>
              <th className="text-right font-normal">Price</th><th className="text-right font-normal">Value</th>
              <th className="text-right font-normal">Realised gain</th></tr>
          </thead>
          <tbody className="tabular-nums">
            {rows.slice(0, n).map((t) => (
              <tr key={`${t.date}${t.etf}${t.side}`} className="border-t border-zinc-800">
                <td className="py-1.5 text-zinc-400">{fmtDate(t.date)}</td>
                <td className="py-1.5">{t.etf}</td>
                <td className={`py-1.5 ${t.side === "BUY" ? "text-emerald-400" : "text-rose-400"}`}>{t.side}</td>
                <td className="py-1.5 text-right">{Number(t.units).toFixed(2)}</td>
                <td className="py-1.5 text-right">${Number(t.price).toFixed(2)}</td>
                <td className="py-1.5 text-right">{money(Number(t.value))}</td>
                <td className={`py-1.5 text-right ${t.side === "BUY" ? "text-zinc-600" : Number(t.gain) < 0 ? "text-rose-400" : "text-emerald-400"}`}>
                  {t.side === "BUY" ? "—" : money(Number(t.gain))}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {n < rows.length && (
        <button onClick={() => setN(rows.length)} className="mt-2 text-xs text-zinc-400 hover:text-zinc-200">
          Show all {rows.length}
        </button>
      )}
    </div>
  );
}

function HowItWorks({ firstLive }: { firstLive: string | null }) {
  return (
    <div className="space-y-2 text-sm text-zinc-300">
      <ol className="list-decimal space-y-1 pl-5">
        <li>Eight trend signals vote on Bitcoin: is the price above its 20, 50, 100 and 200-day average, and
            higher than 1, 3, 6 and 12 months ago? The share of &ldquo;yes&rdquo; votes is the Bitcoin weight.</li>
        <li>When Bitcoin swings harder than its usual level of the past year, the weight is scaled down.</li>
        <li>Whatever Bitcoin leaves goes to gold, sized by the same eight votes on gold&rsquo;s price.</li>
        <li>The rest earns T-bill interest. No leverage, no short selling — both are banned for Indians under LRS anyway.</li>
      </ol>
      <p className="text-xs text-zinc-500">
        Nothing in the model is fitted to past prices; every number is a standard lookback chosen in advance.
        It was picked over 15 alternatives — including an AI that re-picks rules every year and a gradient-boosting
        model — on 2019–2023 data only, then tested on IBIT&rsquo;s real prices from 2024.{" "}
        {firstLive
          ? `Decisions from ${fmtDate(firstLive)} on were recorded live, before the next open; earlier days replay the same rules.`
          : "Every day shown so far replays the same rules on history; decisions become live from the first evening run."}
      </p>
    </div>
  );
}

/* ----------------------------------------------------------------- helpers */

function fmtDate(iso: string) {
  return new Date(iso + (iso.length === 10 ? "T00:00:00" : "")).toLocaleDateString("en-GB",
    { day: "numeric", month: "short", year: "numeric" });
}

function ago(iso: string) {
  const m = Math.round((Date.now() - Date.parse(iso)) / 60_000);
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  return h < 48 ? `${h} h ago` : `${Math.round(h / 24)} days ago`;
}
