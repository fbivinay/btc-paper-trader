import { createClient } from "@supabase/supabase-js";

// The anon key is public by design: it ships in the browser bundle, and row
// level security makes every table it can reach read-only. Broker orders have
// no read policy at all, so they never leave the database.
export const db = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
  { auth: { persistSession: false } },
);

export type Decision = {
  date: string;
  created_at: string;
  live: boolean;
  btc_votes: number;
  btc_vol: number;
  gold_votes: number;
  gold_vol: number;
  w_btc: number;
  w_gold: number;
  w_cash: number;
  btc_close: number;
  gold_close: number;
  tbill: number | null;
  usdinr: number | null;
};

export type EquityRow = {
  date: string;
  model_net: number;
  model_gross: number;
  hold_net: number;
  hold_gross: number;
  btc_close: number;
  gold_close: number;
};

export type Trade = {
  date: string;
  etf: string;
  side: "BUY" | "SELL";
  units: number;
  price: number;
  value: number;
  gain: number;
};

export type Run = {
  id: number;
  started_at: string;
  job: "decide" | "execute";
  status: "ok" | "skipped" | "error";
  detail: string | null;
};

/** Every row of a table, past PostgREST's 1,000-row page limit. */
export async function all<T>(table: string, order: string): Promise<T[]> {
  const out: T[] = [];
  for (let from = 0; ; from += 1000) {
    const { data, error } = await db.from(table).select("*")
      .order(order, { ascending: true }).range(from, from + 999);
    if (error) throw error;
    out.push(...(data as T[]));
    if (!data || data.length < 1000) return out;
  }
}
