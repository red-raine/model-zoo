"""key_lease — per-app key checkout/checkin for the n8n quota pipeline.

Implements the "checkout models + quota, run, check back in" lease so each app
uses its own key, never the master. The n8n pipeline calls these; scripts use
the env vars it returns.

Usage:
    python key_lease.py checkout --app pool-model-zoo --budget 0.10
    python key_lease.py checkin --app pool-model-zoo
    python key_lease.py status --app pool-model-zoo
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

LEASE_DB = Path(r"B:\datasets\local\key-leases.db")

# app -> (key_env_var, ledger path). Keys themselves live in the warden/LiteLLM
# (managed via `lite keys generate`); this module tracks the LEASE state.
APPS = {
    "pool-model-zoo": {"key_env": "LITELLM_KEY_MODELZOO"},
    "pool-sweep": {"key_env": "LITELLM_KEY_SWEEP"},
    "pool-eval": {"key_env": "LITELLM_KEY_EVAL"},
}


def _db() -> sqlite3.Connection:
    LEASE_DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(LEASE_DB))
    c.execute("CREATE TABLE IF NOT EXISTS leases("
              "app TEXT PRIMARY KEY, state TEXT, budget REAL, checked_out_at REAL, "
              "checked_in_at REAL, spend REAL)")
    return c


def checkout(app: str, budget: float) -> dict:
    c = _db()
    now = time.time()
    c.execute(
        "INSERT INTO leases(app,state,budget,checked_out_at,checked_in_at,spend) "
        "VALUES(?,?,?,?,NULL,0) "
        "ON CONFLICT(app) DO UPDATE SET "
        "state='checked_out', budget=excluded.budget, "
        "checked_out_at=excluded.checked_out_at, checked_in_at=NULL, spend=0",
        (app, now, budget, now))
    c.commit()
    return {"app": app, "state": "checked_out", "budget": budget,
            "key_env": APPS[app]["key_env"], "checked_out_at": now}


def checkin(app: str, spend: float = 0.0) -> dict:
    c = _db()
    now = time.time()
    c.execute("UPDATE leases SET state='checked_in', checked_in_at=?, spend=? WHERE app=?",
              (now, spend, app))
    c.commit()
    return {"app": app, "state": "checked_in", "checked_in_at": now, "spend": spend}


def status(app: str) -> dict:
    c = _db()
    row = c.execute("SELECT app,state,budget,checked_out_at,checked_in_at,spend "
                    "FROM leases WHERE app=?", (app,)).fetchone()
    if not row:
        return {"app": app, "state": "never"}
    return {"app": row[0], "state": row[1], "budget": row[2],
            "checked_out_at": row[3], "checked_in_at": row[4], "spend": row[5]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="key_lease")
    sub = ap.add_subparsers(dest="cmd", required=True)
    co = sub.add_parser("checkout"); co.add_argument("--app", required=True, choices=sorted(APPS))
    co.add_argument("--budget", type=float, default=0.10)
    ci = sub.add_parser("checkin"); ci.add_argument("--app", required=True, choices=sorted(APPS))
    ci.add_argument("--spend", type=float, default=0.0)
    st = sub.add_parser("status"); st.add_argument("--app", required=True, choices=sorted(APPS))
    args = ap.parse_args(argv)
    if args.cmd == "checkout":
        print(json.dumps(checkout(args.app, args.budget)))
    elif args.cmd == "checkin":
        print(json.dumps(checkin(args.app, args.spend)))
    else:
        print(json.dumps(status(args.app)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())