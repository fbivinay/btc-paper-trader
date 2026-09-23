import { createBrowserClient } from "@supabase/ssr";

// The anon key is public by design: it ships in the browser bundle and every
// query it makes is fenced by row level security. ml/test_rls.py proves that
// fence holds -- a signed-in user cannot read another user's positions even by
// naming their user_id explicitly.
export const supabase = () =>
  createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  );

export type Decision = {
  id: number;
  candle_time: string;
  price: number;
  prediction: "UP" | "DOWN" | "NEUTRAL";
  confidence: number;
  prob_up: number | null;
  prob_down: number | null;
  prob_neutral: number | null;
  regime: string;
  atr_pct: number | null;
  strategy: string;
  action: "BUY" | "SELL" | "HOLD";
  decided_by: "jev" | "fallback";
  jev_confidence: number | null;
  model_version: string;
};

export type Position = {
  id: number;
  decision_id: number | null;
  side: "LONG" | "SHORT";
  status: "OPEN" | "CLOSED";
  strategy: string | null;
  qty: number;
  entry_price: number;
  entry_time: string;
  stop_loss: number;
  take_profit: number;
  exit_price: number | null;
  exit_time: string | null;
  exit_reason: string | null;
  fees: number;
  pnl: number | null;
  pnl_pct: number | null;
};

export type Profile = {
  user_id: string;
  virtual_capital: number;
  trading_allocation: number;
  risk_profile: "conservative" | "balanced" | "aggressive" | "ai_autonomous" | "max_winrate";
  daily_profit_target: number;
  max_daily_loss: number;
  autonomous: boolean;
  active: boolean;
  preference_until: string | null;
};

export type RiskEvent = {
  id: number;
  decision_id: number | null;
  created_at: string;
  rule: string;
  detail: string | null;
  approved: boolean;
};

export type ModelVersion = {
  version: string;
  window_days: number | null;
  status: string;
  promoted_at: string | null;
  notes: string | null;
  metrics: {
    gross_edge?: number;
    cost?: number;
    gate?: string;
    sharpe?: number;
    folds_profitable?: number;
    n_folds?: number;
    horizon?: number;
    checks?: Record<string, boolean>;
  };
};

export type Equity = { candle_time: string; equity: number; cash: number; open_pnl: number };
