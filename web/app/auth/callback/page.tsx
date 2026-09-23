"use client";

import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";

/**
 * Lands here from the email confirmation link.
 *
 * createBrowserClient uses the PKCE flow, so the link comes back with a `code`
 * query parameter that must be exchanged for a session. Without this exchange
 * the user is bounced to /login already confirmed but still signed out, which
 * looks exactly like the confirmation having failed.
 *
 * Older Supabase links instead put tokens in the URL hash, which the client
 * picks up on its own -- so if there is no code, just check for a session.
 */
export default function AuthCallback() {
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      const db = supabase();
      const params = new URLSearchParams(window.location.search);

      // Never render error_description from the URL. React escapes it, so it is
      // not XSS, but it is attacker-controlled text on a page the user trusts:
      // /auth/callback?error_description=Account+locked,+call+1-800-... would
      // display as though the app said it. Map the machine-readable code to our
      // own copy instead, and keep anything unrecognised generic.
      const errorCode = params.get("error_code") ?? params.get("error");
      if (errorCode) {
        setError(
          {
            access_denied: "That confirmation link was declined. Request a new one.",
            otp_expired: "That confirmation link has expired. Sign up again to get a new one.",
            server_error: "Sign-in is temporarily unavailable. Try again shortly.",
          }[errorCode] ?? "That confirmation link is not valid. Try signing in, or sign up again."
        );
        return;
      }

      const code = params.get("code");
      if (code) {
        const { error } = await db.auth.exchangeCodeForSession(code);
        if (error) { setError(error.message); return; }
        window.location.replace("/");
        return;
      }

      const { data } = await db.auth.getSession();
      window.location.replace(data.session ? "/" : "/login");
    })();
  }, []);

  return (
    <main className="min-h-dvh flex items-center justify-center bg-zinc-950 p-6 text-zinc-100">
      {error ? (
        <div className="max-w-sm text-center">
          <p className="text-sm text-amber-400">{error}</p>
          <a href="/login" className="mt-4 inline-block text-sm text-zinc-400 hover:text-zinc-200">
            Back to sign in
          </a>
        </div>
      ) : (
        <p className="text-sm text-zinc-500">Confirming…</p>
      )}
    </main>
  );
}
