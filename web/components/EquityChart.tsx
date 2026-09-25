"use client";

import { useEffect, useRef } from "react";
import { createChart, LineSeries, type IChartApi, type ISeriesApi } from "lightweight-charts";

type Line = { label: string; color: string; dates: string[]; values: number[] };

export default function EquityChart({ lines }: { lines: Line[] }) {
  const box = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const series = useRef<ISeriesApi<"Line">[]>([]);

  // Create once; recreating on every data change would reset the user's zoom.
  useEffect(() => {
    if (!box.current) return;
    const c = createChart(box.current, {
      layout: { background: { color: "transparent" }, textColor: "#a1a1aa", attributionLogo: false },
      grid: { vertLines: { color: "#1f1f23" }, horzLines: { color: "#1f1f23" } },
      rightPriceScale: { borderColor: "#27272a" },
      timeScale: { borderColor: "#27272a" },
      crosshair: { mode: 0 },
      autoSize: true,
    });
    chart.current = c;
    return () => { c.remove(); chart.current = null; series.current = []; };
  }, []);

  useEffect(() => {
    const c = chart.current;
    if (!c) return;
    series.current.forEach((s) => c.removeSeries(s));
    series.current = lines.map((l) => {
      const s = c.addSeries(LineSeries, { color: l.color, lineWidth: 2, title: l.label,
                                          priceLineVisible: false });
      s.setData(l.dates.map((d, i) => ({ time: d, value: l.values[i] })));
      return s;
    });
    c.timeScale().fitContent();
  }, [lines]);

  return <div ref={box} className="h-[320px] w-full" />;
}
