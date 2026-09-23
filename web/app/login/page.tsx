"use client";

import { useState } from "react";
import { supabase } from "@/lib/supabase";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [mode, setMode] = useState<"in" | "up">("in");
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setMsg(null);
    const db = supabase();
    const { error } =
      mode === "in"
        ? await db.auth.signInWithPassword({ email, password })
        : await db.auth.signUp({ email, password });
    if (error) setMsg(error.message);
    else if (mode === "up") setMsg("Account created. Check your email if confirmation is on, then sign in.");
    else window.location.href = "/";
    setBusy(false);
  }

  return (
    <main className="min-h-dvh flex items-center justify-center p-6 bg-zinc-950 text-zinc-100">
      <div className="w-full max-w-sm">
        <h1 className="text-2xl font-semibold tracking-tight">BTC Paper Trader</h1>
        <p className="mt-2 text-sm text-zinc-400">
          Simulated trading with virtual capital. No real money, no real orders.
        </p>

        <form onSubmit={submit} className="mt-8 space-y-3">
          <input
            type="email" required value={email} onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com" autoComplete="email"
            className="w-full rounded-md bg-zinc-900 border border-zinc-800 px-3 py-2 text-sm
                       outline-none focus:border-zinc-600"
          />
          <input
            type="password" required minLength={8} value={password}
            onChange={(e) => setPassword(e.target.value)} placeholder="password"
            autoComplete={mode === "in" ? "current-password" : "new-password"}
            className="w-full rounded-md bg-zinc-900 border border-zinc-800 px-3 py-2 text-sm
                       outline-none focus:border-zinc-600"
          />
          <button
            type="submit" disabled={busy}
            className="w-full rounded-md bg-zinc-100 text-zinc-900 px-3 py-2 text-sm font-medium
                       hover:bg-white disabled:opacity-50"
          >
            {busy ? "…" : mode === "in" ? "Sign in" : "Create account"}
          </button>
        </form>

        {msg && <p className="mt-4 text-sm text-amber-400">{msg}</p>}

        <button
          onClick={() => { setMode(mode === "in" ? "up" : "in"); setMsg(null); }}
          className="mt-6 text-sm text-zinc-400 hover:text-zinc-200"
        >
          {mode === "in" ? "Need an account? Sign up" : "Have an account? Sign in"}
        </button>

        <p className="mt-10 text-xs leading-relaxed text-zinc-500">
          This system is a research project. Its model does not currently have a
          tradeable edge after fees, and the dashboard shows you exactly why.
        </p>
      </div>
    </main>
  );
}
