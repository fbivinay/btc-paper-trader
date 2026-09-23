"use client";

import { useEffect, useRef, useState } from "react";
import {
  CandlestickSeries, createChart, createSeriesMarkers, type IChartApi,
  type ISeriesApi, type UTCTimestamp,
} from "lightweight-charts";
import type { Position } from "@/lib/supabase";

// Binance's public market-data mirror. Same host the trading loop uses, and it
// sets permissive CORS headers so the browser can read it directly -- no proxy
// route and no server-side fetch needed.
const HOST = "https://data-api.binance.vision";

type Kline = { time: UTCTimestamp; open: number; high: number; low: number; close: number };

const INTERVALS = [
  { label: "5m", v: "5m" },
  { label: "15m", v: "15m" },
  { label: "1h", v: "1h" },
  { label: "4h", v: "4h" },
  { label: "1D", v: "1d" },
];

export default function PriceChart({ positions }: { positions: Position[] }) {
  const box = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const series = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const [interval, setInterval_] = useState("5m");
  const [last, setLast] = useState<number | null>(null);
  const [change, setChange] = useState<number | null>(null);

  // Create the chart once. Recreating it on every data refresh would reset the
  // user's pan and zoom every few seconds.
  useEffect(() => {
    if (!box.current) return;
    const c = createChart(box.current, {
      layout: { background: { color: "transparent" }, textColor: "#a1a1aa", attributionLogo: false },
      grid: { vertLines: { color: "#1f1f23" }, horzLines: { color: "#1f1f23" } },
      rightPriceScale: { borderColor: "#27272a" },
      timeScale: { borderColor: "#27272a", timeVisible: true, secondsVisible: false },
      crosshair: { mode: 0 },
      autoSize: true,
    });
    const s = c.addSeries(CandlestickSeries, {
      upColor: "#22c55e", downColor: "#ef4444", borderVisible: false,
      wickUpColor: "#22c55e", wickDownColor: "#ef4444",
    });
    chart.current = c;
    series.current = s;
    return () => { c.remove(); chart.current = null; series.current = null; };
  }, []);

  useEffect(() => {
    let alive = true;

    async function load() {
      const r = await fetch(
        `${HOST}/api/v3/klines?symbol=BTCUSDT&interval=${interval}&limit=500`
      );
      if (!r.ok || !alive) return;
      const raw: (string | number)[][] = await r.json();
      const bars: Kline[] = raw.map((k) => ({
        time: (Number(k[0]) / 1000) as UTCTimestamp,
        open: Number(k[1]), high: Number(k[2]), low: Number(k[3]), close: Number(k[4]),
      }));
      series.current?.setData(bars);

      const first = bars[0], latest = bars[bars.length - 1];
      setLast(latest.close);
      setChange((latest.close - first.open) / first.open);

      // Mark where the system actually entered and exited.
      const marks = positions.flatMap((p) => {
        const out = [{
          time: (new Date(p.entry_time).getTime() / 1000) as UTCTimestamp,
          position: (p.side === "LONG" ? "belowBar" : "aboveBar") as "belowBar" | "aboveBar",
          color: p.side === "LONG" ? "#22c55e" : "#ef4444",
          shape: (p.side === "LONG" ? "arrowUp" : "arrowDown") as "arrowUp" | "arrowDown",
          text: p.side === "LONG" ? "BUY" : "SELL",
        }];
        if (p.exit_time)
          out.push({
            time: (new Date(p.exit_time).getTime() / 1000) as UTCTimestamp,
            position: (p.side === "LONG" ? "aboveBar" : "belowBar") as "belowBar" | "aboveBar",
            color: (p.pnl ?? 0) >= 0 ? "#22c55e" : "#ef4444",
            shape: "circle" as unknown as "arrowUp",
            text: `${(p.pnl ?? 0) >= 0 ? "+" : ""}${(p.pnl ?? 0).toFixed(0)}`,
          });
        return out;
      }).sort((a, b) => (a.time as number) - (b.time as number));

      if (series.current && marks.length) createSeriesMarkers(series.current, marks);
    }

    load();
    const t = window.setInterval(load, 15_000);
    return () => { alive = false; window.clearInterval(t); };
  }, [interval, positions]);

  const up = (change ?? 0) >= 0;

  return (
    <section className="rounded-lg border border-zinc-800 bg-zinc-900/40">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800 px-4 py-3">
        <div className="flex items-baseline gap-3">
          <span className="text-sm font-medium text-zinc-300">BTC/USDT</span>
          <span className="text-2xl font-semibold tabular-nums">
            {last === null ? "—" : `$${last.toLocaleString(undefined, {
              minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          </span>
          <span className={`text-sm tabular-nums ${up ? "text-emerald-400" : "text-rose-400"}`}>
            {change === null ? "" : `${up ? "+" : ""}${(change * 100).toFixed(2)}%`}
          </span>
        </div>
        <div className="flex gap-1">
          {INTERVALS.map((i) => (
            <button
              key={i.v} onClick={() => setInterval_(i.v)}
              className={`rounded px-2 py-1 text-xs ${
                interval === i.v ? "bg-zinc-700 text-zinc-100"
                                 : "text-zinc-500 hover:text-zinc-300"}`}
            >
              {i.label}
            </button>
          ))}
        </div>
      </div>
      <div ref={box} className="h-[380px] w-full" />
    </section>
  );
}
