"""Prove row level security actually isolates users. Run against a real project.

    python ml/test_rls.py

Enabling RLS and writing policies does not mean they work -- a policy with the
wrong USING clause, or a table where RLS is enabled but no policy matches, both
fail silently in the direction of leaking. This signs in as two real users and
checks that each sees only their own rows.

Creates two throwaway users and deletes them at the end.
"""

import json
import os
import sys
import urllib.error
import urllib.request
import uuid

from db import load_env

load_env()
URL = os.environ["SUPABASE_URL"].rstrip("/")
SERVICE = os.environ["SUPABASE_SERVICE_KEY"]
ANON = os.environ.get("SUPABASE_ANON_KEY", "")

PASSWORD = "Test-" + uuid.uuid4().hex[:12] + "!aZ9"


def call(path, method="GET", body=None, key=None, token=None, prefer=None):
    headers = {"apikey": key or ANON, "Content-Type": "application/json",
               "Authorization": f"Bearer {token or key or ANON}",
               "User-Agent": "btc-paper-trader-test/1.0"}
    if prefer:
        headers["Prefer"] = prefer
    req = urllib.request.Request(
        f"{URL}{path}", data=json.dumps(body).encode() if body is not None else None,
        method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]


def create_user(email):
    code, body = call("/auth/v1/admin/users", "POST",
                      {"email": email, "password": PASSWORD, "email_confirm": True},
                      key=SERVICE)
    assert code in (200, 201), f"create user failed: {code} {body}"
    return body["id"]


def sign_in(email):
    code, body = call("/auth/v1/token?grant_type=password", "POST",
                      {"email": email, "password": PASSWORD}, key=ANON)
    assert code == 200, f"sign in failed: {code} {body}"
    return body["access_token"]


def main():
    if not ANON:
        sys.exit("SUPABASE_ANON_KEY must be set")

    tag = uuid.uuid4().hex[:8]
    email_a, email_b = f"rls-a-{tag}@example.com", f"rls-b-{tag}@example.com"
    user_a = user_b = None
    failures = []

    def check(label, condition, detail=""):
        print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
        if not condition:
            failures.append(label)

    try:
        user_a, user_b = create_user(email_a), create_user(email_b)
        print(f"created two users\n")

        # The signup trigger must have made a profile for each, without the
        # 5-minute job ever touching them.
        print("signup trigger")
        _, profiles = call(f"/rest/v1/profiles?select=user_id&user_id=in.({user_a},{user_b})",
                           key=SERVICE)
        check("profile auto-created for both users", len(profiles) == 2,
              f"found {len(profiles)}")
        if profiles:
            _, p = call(f"/rest/v1/profiles?select=*&user_id=eq.{user_a}", key=SERVICE)
            check("profile defaults applied",
                  p[0]["risk_profile"] == "balanced" and float(p[0]["virtual_capital"]) == 10000,
                  f"{p[0]['risk_profile']}, {p[0]['virtual_capital']}")

        # Seed a decision and a position owned by A only.
        print("\nseeding data owned by user A")
        code, dec = call("/rest/v1/decisions", "POST", {
            "candle_time": "2026-09-23T00:00:00Z", "price": 112000, "prediction": "UP",
            "confidence": 0.71, "regime": "uptrend", "strategy": "trend_following",
            "action": "BUY", "decided_by": "fallback", "model_version": "test",
        }, key=SERVICE, prefer="return=representation")
        check("service role can write decisions", code in (200, 201), str(dec)[:80])
        decision_id = dec[0]["id"] if isinstance(dec, list) and dec else None

        code, pos = call("/rest/v1/positions", "POST", {
            "user_id": user_a, "decision_id": decision_id, "side": "LONG",
            "qty": 0.01, "entry_price": 112000, "entry_time": "2026-09-23T00:05:00Z",
            "stop_loss": 111000, "take_profit": 113500, "strategy": "trend_following",
        }, key=SERVICE, prefer="return=representation")
        check("service role can write positions", code in (200, 201), str(pos)[:80])

        # The actual isolation test.
        print("\nrow level security")
        token_a, token_b = sign_in(email_a), sign_in(email_b)

        _, a_sees = call("/rest/v1/positions?select=*", token=token_a, key=ANON)
        check("user A sees their own position", isinstance(a_sees, list) and len(a_sees) == 1,
              f"{len(a_sees) if isinstance(a_sees, list) else a_sees} rows")

        _, b_sees = call("/rest/v1/positions?select=*", token=token_b, key=ANON)
        check("user B cannot see user A's position",
              isinstance(b_sees, list) and len(b_sees) == 0,
              f"{len(b_sees) if isinstance(b_sees, list) else b_sees} rows")

        # Even naming the row explicitly must not reveal it.
        _, b_targeted = call(f"/rest/v1/positions?select=*&user_id=eq.{user_a}",
                             token=token_b, key=ANON)
        check("user B cannot read A's row by explicit filter",
              isinstance(b_targeted, list) and len(b_targeted) == 0,
              f"{len(b_targeted) if isinstance(b_targeted, list) else b_targeted} rows")

        _, b_profiles = call("/rest/v1/profiles?select=*", token=token_b, key=ANON)
        check("user B sees only their own profile",
              isinstance(b_profiles, list) and len(b_profiles) == 1
              and b_profiles[0]["user_id"] == user_b,
              f"{len(b_profiles) if isinstance(b_profiles, list) else b_profiles} rows")

        # Anonymous callers get nothing at all.
        print("\nanonymous access")
        _, anon_pos = call("/rest/v1/positions?select=*", key=ANON)
        check("anon sees no positions", isinstance(anon_pos, list) and len(anon_pos) == 0,
              f"{len(anon_pos) if isinstance(anon_pos, list) else anon_pos} rows")

        _, anon_dec = call("/rest/v1/decisions?select=*", key=ANON)
        check("anon sees no decisions (policy is authenticated-only)",
              isinstance(anon_dec, list) and len(anon_dec) == 0,
              f"{len(anon_dec) if isinstance(anon_dec, list) else anon_dec} rows")

        _, auth_dec = call("/rest/v1/decisions?select=*", token=token_b, key=ANON)
        check("signed-in user CAN see market-wide decisions",
              isinstance(auth_dec, list) and len(auth_dec) >= 1,
              f"{len(auth_dec) if isinstance(auth_dec, list) else auth_dec} rows")

        # A user must not be able to forge a position for themselves or anyone else.
        print("\nwrite protection")
        code, _ = call("/rest/v1/positions", "POST", {
            "user_id": user_b, "side": "LONG", "qty": 99, "entry_price": 1,
            "entry_time": "2026-09-23T00:05:00Z", "stop_loss": 0.5, "take_profit": 2,
        }, token=token_b, key=ANON)
        check("user cannot insert their own position", code in (401, 403, 404, 405, 42501),
              f"HTTP {code}")

        code, _ = call(f"/rest/v1/positions?user_id=eq.{user_a}", "PATCH",
                       {"pnl": 999999}, token=token_b, key=ANON)
        _, after = call(f"/rest/v1/positions?select=pnl&user_id=eq.{user_a}", key=SERVICE)
        check("user B cannot modify user A's position",
              after and after[0]["pnl"] is None, f"pnl={after[0]['pnl'] if after else '?'}")

        # The RPC endpoint the advisor flagged must stay shut.
        print("\nprivilege escalation")
        code, _ = call("/rest/v1/rpc/handle_new_user", "POST", {}, key=ANON)
        check("anon cannot call handle_new_user()", code in (401, 403, 404), f"HTTP {code}")

    finally:
        print("\ncleanup")
        for uid in (user_a, user_b):
            if uid:
                call(f"/auth/v1/admin/users/{uid}", "DELETE", key=SERVICE)
        call("/rest/v1/decisions?model_version=eq.test", "DELETE", key=SERVICE)
        print("  test users and rows removed")

    print()
    if failures:
        sys.exit(f"{len(failures)} RLS CHECK(S) FAILED: {', '.join(failures)}")
    print("all RLS checks passed")


if __name__ == "__main__":
    main()
