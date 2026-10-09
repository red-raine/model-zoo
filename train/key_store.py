r"""key_store — per-app LiteLLM API key access (CONSTITUTIONAL LAW).

LAW (AGENTS.md, 2026-10-08):
  - One key PER APP. Never the master key (LITELLM_MASTER_KEY).
  - All keys requested + tracked per app; models/quota stated when requesting.
  - Keys live ONLY in C:\Users\myste\.config\opencode\keys\<app>.key (outside
    repos, never committed). Set via askpass.ps1 (Windows popup).
  - Apps checkout models/quota via key_lease.py; unpaid/extra models must be
    requested and added to the key's model list by the stack master.

Usage:
    from key_store import get_key, KEY_DIR
    key = get_key("pool-model-zoo")   # raises KeyError if not registered

The chosen model must ALSO be checked against the key's allowed models; this
module only returns the stored key. Model allowance is enforced at request time
by LiteLLM (the key's model list). We list the app's allowed models here for
requesting/checkout (key_lease), not as an escape hatch.
"""

from __future__ import annotations

import os
from pathlib import Path

KEY_DIR = Path(os.environ.get("OPENCODE_KEY_DIR",
                              Path.home() / ".config" / "opencode" / "keys"))
KEY_DIR.mkdir(parents=True, exist_ok=True)


def get_key(app: str) -> str:
    """Return the stored API key for `app`. Raises KeyError if missing."""
    f = KEY_DIR / f"{app}.key"
    if not f.exists():
        raise KeyError(
            f"No key registered for app '{app}'. "
            f"Run askpass.ps1 first:\n"
            f"  powershell -ExecutionPolicy Bypass -File "
            r"E:\vibe_coding\dev\bitnet runner\tools\tm\askpass.ps1"
            f" -App {app}\n"
            f"Or request the key from the stack master (per-app key, never master).")
    key = f.read_text(encoding="utf-8").strip()
    if len(key) < 8:
        raise KeyError(f"Key for '{app}' looks empty/short at {f} — re-run askpass.ps1.")
    return key


def apps() -> list[str]:
    """List registered per-app keys (for key_lease / checkout)."""
    return [f.stem for f in KEY_DIR.glob("*.key")]


def is_registered(app: str) -> bool:
    return (KEY_DIR / f"{app}.key").exists()


if __name__ == "__main__":
    print("registered apps:", apps())