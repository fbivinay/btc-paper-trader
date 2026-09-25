"""Thin PostgREST client for Supabase.

No supabase-py dependency: with the service-role key this is two headers on an
HTTP request, and keeping it dependency-free means the GitHub Actions job needs
no install step beyond torch/pandas.

The service-role key bypasses RLS by design -- the daily job is the only
writer. It must never reach the browser; it lives only in a GitHub Actions
secret and the gitignored .env. The dashboard uses the anon key, read-only by RLS.
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path


class DBError(RuntimeError):
    pass


def load_env(path: str | None = None) -> None:
    """Read a local .env into os.environ without overwriting what is already set.

    Exists so the service-role key can live in a gitignored file rather than
    being pasted around. Real environment variables win, which is what makes
    GitHub Actions secrets take precedence in CI.
    """
    env = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


class DB:
    def __init__(self, url: str | None = None, key: str | None = None, timeout: float = 15.0):
        if url is None and key is None:
            load_env()
        self.url = (url or os.environ.get("SUPABASE_URL", "")).rstrip("/")
        self.key = key or os.environ.get("SUPABASE_SERVICE_KEY", "")
        self.timeout = timeout
        if not self.url or not self.key:
            raise DBError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

    def _request(self, method: str, path: str, body=None, prefer: str | None = None):
        headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "User-Agent": "btc-paper-trader/1.0",
        }
        if prefer:
            headers["Prefer"] = prefer
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"{self.url}/rest/v1/{path}", data=data,
                                     method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            # PostgREST puts the real reason in the body; the status alone is useless.
            try:
                detail = e.read().decode()[:500]
            except Exception:
                detail = ""
            raise DBError(f"{method} {path} -> HTTP {e.code}: {detail}") from None

    def select(self, table: str, query: str = "", limit: int | None = None) -> list:
        path = f"{table}?{query}" if query else f"{table}?select=*"
        if limit:
            path += f"&limit={limit}"
        return self._request("GET", path) or []

    def insert(self, table: str, rows, upsert_on: str | None = None,
               keep_existing: bool = False) -> list:
        """upsert_on: overwrite rows that clash on that key; keep_existing: skip them."""
        prefer = "return=representation"
        if upsert_on:
            prefer += ",resolution=" + ("ignore-duplicates" if keep_existing else "merge-duplicates")
        path = f"{table}?on_conflict={upsert_on}" if upsert_on else table
        return self._request("POST", path, rows, prefer=prefer) or []

    def update(self, table: str, query: str, patch: dict) -> list:
        return self._request("PATCH", f"{table}?{query}", patch,
                             prefer="return=representation") or []

    def rpc(self, fn: str, args: dict | None = None):
        return self._request("POST", f"rpc/{fn}", args or {})

    def ping(self) -> bool:
        """Cheap reachability and auth check."""
        self._request("GET", "etf_runs?select=id&limit=1")
        return True


def _self_check() -> None:
    """Offline checks only -- URL and header construction, not the network."""
    try:
        DB(url="", key="")
    except DBError:
        pass
    else:
        raise AssertionError("empty config must raise")

    db = DB(url="https://x.supabase.co/", key="k")
    assert db.url == "https://x.supabase.co", "trailing slash must be stripped"

    seen = {}

    def fake(method, path, body=None, prefer=None):
        seen.update(method=method, path=path, body=body, prefer=prefer)
        return []

    db._request = fake
    db.select("etf_decisions", "order=date.desc", limit=5)
    assert seen["path"] == "etf_decisions?order=date.desc&limit=5", seen["path"]

    db.insert("etf_equity", {"a": 1}, upsert_on="date")
    assert seen["path"] == "etf_equity?on_conflict=date"
    assert "merge-duplicates" in seen["prefer"], seen["prefer"]

    # Decisions are written once and never overwritten.
    db.insert("etf_decisions", {"a": 1}, upsert_on="date", keep_existing=True)
    assert "ignore-duplicates" in seen["prefer"] and "merge" not in seen["prefer"], seen["prefer"]

    db.insert("etf_runs", {"a": 1})
    assert seen["path"] == "etf_runs" and "resolution" not in seen["prefer"]

    db.update("etf_orders", "id=eq.7", {"status": "filled"})
    assert seen["method"] == "PATCH" and seen["path"] == "etf_orders?id=eq.7"

    # load_env must not clobber a real environment variable (CI secrets win).
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / ".env"
        f.write_text('\n'.join(['# comment', 'SUPA_TEST_A=fromfile',
                                'SUPA_TEST_B="quoted"', '', 'badline', '']))
        os.environ["SUPA_TEST_A"] = "fromenv"
        load_env(str(f))
        assert os.environ["SUPA_TEST_A"] == "fromenv", "real env must win over .env"
        assert os.environ["SUPA_TEST_B"] == "quoted", "quotes must be stripped"
    load_env("/definitely/not/here/.env")   # missing file must be a no-op

    print("db self-check passed")


if __name__ == "__main__":
    _self_check()
