export const CAPITAL = 10_000; // the model portfolio's virtual starting dollars (ml/etf_daily.py)

export type Stats = {
  total: number;
  perYear: number;
  worstDrop: number;
  years: [string, number][];
};

/** Total return, compound yearly return, worst peak-to-trough drop, calendar-year returns. */
export function stats(dates: string[], values: number[]): Stats {
  const last = values[values.length - 1];
  const days = (Date.parse(dates[dates.length - 1]) - Date.parse(dates[0])) / 86_400_000;
  let peak = -Infinity, worstDrop = 0;
  const yearEnd = new Map<string, number>();
  values.forEach((v, i) => {
    peak = Math.max(peak, v);
    worstDrop = Math.min(worstDrop, v / peak - 1);
    yearEnd.set(dates[i].slice(0, 4), v);
  });
  let prev = CAPITAL;
  const years: [string, number][] = [];
  for (const [y, v] of yearEnd) {
    years.push([y, v / prev - 1]);
    prev = v;
  }
  return {
    total: last / CAPITAL - 1,
    perYear: days > 0 ? Math.pow(last / CAPITAL, 365.25 / days) - 1 : 0,
    worstDrop,
    years,
  };
}

export const pct = (x: number, digits = 1) =>
  `${x > 0 ? "+" : ""}${(x * 100).toFixed(digits)}%`;

export const money = (x: number, cur = "$") =>
  `${x < 0 ? "-" : ""}${cur}${Math.abs(x).toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
