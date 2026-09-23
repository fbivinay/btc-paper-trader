"use client";

import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";
import type { Decision, ModelVersion, RiskEvent } from "@/lib/supabase";
import { Panel } from "@/components/Trading";

/**
 * Everything about how the system decides, kept off the trading screen.
 *
 * This page exists because the honest numbers matter and the dashboard is not
 * the place for them -- a user checking their balance does not need walk-forward
 * fold counts, but anyone asking "should I trust this" should be able to find
 * them in one click.
 */
export default function ModelPage() {
  const [model, setModel] = useState<ModelVersion | null>(null);
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [events, setEvents] = useState<RiskEvent[]>([]);

  useEffect(() => {
    (async () => {
      const db = supabase();
      const { data: { user } } = await db.auth.getUser();
      if (!user) { window.location.href = "/login"; return; }
      const [m, d, e] = await Promise.all([
        db.from("model_versions").select("*").eq("status", "production").limit(1),
        db.from("decisions").select("*").order("candle_time", { ascending: false }).limit(100),
        db.from("risk_events").select("*").order("created_at", { ascending: false }).limit(300),
      ]);
      setModel(m.data?.[0] ?? null);
      setDecisions(d.data ?? []);
      setEvents(e.data ?? []);
    })();
  }, []);

  const edge = model?.metrics.gross_edge ?? 0;
  const cost = model?.metrics.cost ?? 0.0025;
  const byJev = decisions.filter((d) => d.decided_by === "jev").length;

  const declines = events.reduce<Record<string, number>>((a, e) => {
    if (!e.approved) a[e.rule] = (a[e.rule] ?? 0) + 1;
    return a;
  }, {});

  return (
    <main className="min-h-dvh bg-zinc-950 text-zinc-100">
      <header className="border-b border-zinc-900">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-4 py-3">
          <span className="text-sm font-semibold">How it decides</span>
          <a href="/" className="text-xs text-zinc-500 hover:text-zinc-300">← Back to trading</a>
        </div>
      </header>

      <div className="mx-auto max-w-4xl space-y-4 px-4 py-5">
        <Panel title="Does the edge cover the fees?">
          <div className="grid grid-cols-3 gap-4">
            <N label="Gross edge / trade" v={`${(edge * 100).toFixed(3)}%`} tone="up" />
            <N label="Round-trip cost" v={`−${(cost * 100).toFixed(3)}%`} tone="down" />
            <N label="Net" v={`${((edge - cost) * 100).toFixed(3)}%`}
               tone={edge > cost ? "up" : "down"} />
          </div>
          <div className="mt-4 h-2 overflow-hidden rounded-full bg-zinc-800">
            <div className="h-full bg-emerald-500"
                 style={{ width: `${Math.min(100, (edge / cost) * 100)}%` }} />
          </div>
          <p className="mt-2 text-xs text-zinc-400">
            The model captures{" "}
            <span className="text-zinc-200">{((edge / cost) * 100).toFixed(0)}%</span> of what a
            round trip costs. It needs more than 100% to be profitable, which is why
            the system declines most signals.
          </p>
          {model?.notes?.includes("FORCED") && (
            <p className="mt-3 rounded border border-amber-900/60 bg-amber-950/30 p-2 text-xs text-amber-300">
              This model did not pass the automated promotion gate. It runs so the
              system can be demonstrated end to end, and the override is recorded
              in the database.
            </p>
          )}
        </Panel>

        <Panel title="Model">
          <dl className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-xs">
            <K k="Version" v={model?.version ?? "—"} />
            <K k="Training window" v={model?.window_days ? `${model.window_days} days` : "—"} />
            <K k="Prediction horizon"
               v={model?.metrics.horizon ? `${(model.metrics.horizon * 5) / 60} hours` : "—"} />
            <K k="Walk-forward"
               v={`${model?.metrics.folds_profitable ?? 0}/${model?.metrics.n_folds ?? 0} folds profitable`} />
          </dl>
          <p className="mt-3 text-xs leading-relaxed text-zinc-500">
            Selected by walk-forward validation across 180, 365 and 730-day training
            windows, scored on trading economics rather than classification accuracy.
            Accuracy is misleading here: a model that never trades would score highly
            and earn nothing.
          </p>
        </Panel>

        <Panel title={`Strategy layer (last ${decisions.length} candles)`}>
          <p className="text-sm text-zinc-300">
            Jev chose {byJev}, the fallback table chose {decisions.length - byJev}.
          </p>
          <p className="mt-2 text-xs leading-relaxed text-zinc-500">
            Jev is rate limited on its free tier. When a call fails, a deterministic
            table picks the strategy from the same prediction and regime, so a
            decision is always made. Which path ran is recorded per candle.
          </p>
        </Panel>

        <Panel title="Why trades were declined">
          {Object.keys(declines).length === 0 ? (
            <p className="text-sm text-zinc-500">Nothing declined yet.</p>
          ) : (
            <ul className="space-y-1.5 text-sm">
              {Object.entries(declines).sort((a, b) => b[1] - a[1]).map(([rule, n]) => (
                <li key={rule} className="flex items-baseline justify-between">
                  <span className="text-zinc-300">{rule.replace(/_/g, " ")}</span>
                  <span className="tabular-nums text-zinc-500">{n}</span>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
    </main>
  );
}

const N = ({ label, v, tone }: { label: string; v: string; tone?: "up" | "down" }) => (
  <div>
    <div className="text-xs text-zinc-500">{label}</div>
    <div className={`mt-1 text-xl font-semibold tabular-nums ${
      tone === "up" ? "text-emerald-400" : tone === "down" ? "text-rose-400" : ""}`}>{v}</div>
  </div>
);

const K = ({ k, v }: { k: string; v: string }) => (
  <>
    <dt className="text-zinc-500">{k}</dt>
    <dd className="text-right text-zinc-300">{v}</dd>
  </>
);
