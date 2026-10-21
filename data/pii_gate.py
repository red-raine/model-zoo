"""pii_gate.py — G-PII. Detect and redact personal data and secrets in corpora.

WHY THIS FILE EXISTS
--------------------
`sources_ext.py` sets `flags.needs_deid` on sources and NOTHING READS IT. There is
no de-identification tool anywhere in the family. The red-team gate
(`d6_redteam_litellm.py`) tests a *model's output*; nothing ever tested the
*corpus input*. `clamav` is running and is malware-only, and is not wired to any
dataset gate.

For a therapy/EQ corpus (counsel-chat, Amod, fadodr, michelleyunun,
hope_therapy...) this is not theoretical: those rows are real people describing
real distress, frequently with names, places and contact details.

Design: detect, then REDACT by default (dropping rows loses signal; silently
keeping PII is unacceptable). Secrets are treated harder: a corpus containing a
live API key is a security incident, not a quality issue.

Policy (operator, 2026-10-20): internal-only. We still redact — an internal
corpus leaks the same way an external one does, just later.

USAGE
-----
    python tools/data/pii_gate.py --self-test
    python tools/data/pii_gate.py --input A.jsonl --output B.jsonl \
        --report R.json [--mode redact|flag|drop] [--fields text,completion]

Exit: 0 clean, 4 secrets found (still writes output), 1 error, 2 self-test fail
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import time

# --------------------------------------------------------------------------
# Detectors.  (pattern, replacement, severity, label)
# severity: "secret" = credential/security incident, "pii" = personal data
# --------------------------------------------------------------------------

DETECTORS: list[tuple[re.Pattern, str, str, str]] = [
    # ---- credentials first: these are incidents, not quality issues ------
    (re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}"), "[REDACTED_ANTHROPIC_KEY]", "secret", "anthropic_key"),
    (re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}"), "[REDACTED_OPENAI_KEY]", "secret", "openai_key"),
    (re.compile(r"\bhf_[A-Za-z0-9]{30,}"), "[REDACTED_HF_TOKEN]", "secret", "hf_token"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"), "[REDACTED_GITHUB_TOKEN]", "secret", "github_token"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}"), "[REDACTED_GITHUB_PAT]", "secret", "github_pat"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY_ID]", "secret", "aws_key_id"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "[REDACTED_GOOGLE_KEY]", "secret", "google_api_key"),
    (re.compile(r"\bxox[baprs]-[0-9A-Za-z\-]{10,}"), "[REDACTED_SLACK_TOKEN]", "secret", "slack_token"),
    (re.compile(r"\bglpat-[0-9A-Za-z_\-]{20,}"), "[REDACTED_GITLAB_TOKEN]", "secret", "gitlab_token"),
    (re.compile(r"\bSG\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}"), "[REDACTED_SENDGRID_KEY]", "secret", "sendgrid_key"),
    (re.compile(r"\b(?:rk|pk)_(?:live|test)_[0-9A-Za-z]{20,}"), "[REDACTED_STRIPE_KEY]", "secret", "stripe_key"),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"), "[REDACTED_JWT]", "secret", "jwt"),
    (re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
                r"[\s\S]{0,4000}?-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"),
     "[REDACTED_PRIVATE_KEY]", "secret", "private_key"),
    (re.compile(r"(?i)\b(?:aws_secret_access_key|secret_key|api[_-]?key|access[_-]?token|"
                r"client[_-]?secret|password|passwd|pwd)\s*[:=]\s*['\"]?([A-Za-z0-9/\+_\-]{16,})['\"]?"),
     r"\1=[REDACTED_SECRET]", "secret", "generic_secret"),
    (re.compile(r"://[^/\s:@]{2,}:[^/\s:@]{2,}@"), "://[REDACTED_CREDENTIALS]@", "secret", "url_credentials"),

    # ---- personal data ---------------------------------------------------
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL]", "pii", "email"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]", "pii", "ssn"),
    # Separators are REQUIRED. With every separator optional this matched
    # `1000000007` (a prime modulus in SQL) and `2026040510` (a timestamp id)
    # and rewrote them to [PHONE] inside real code completions.
    (re.compile(r"(?<![\d\w])"
                r"(?:\+\d{1,3}[\s.\-]?)?"
                r"(?:\(\d{2,4}\)[\s.\-]?\d{3,4}[\s.\-]?\d{3,4}"
                r"|\d{2,4}[\s.\-]\d{3,4}[\s.\-]\d{3,4})"
                r"(?![\d\w])"),
     "[PHONE]", "pii", "phone"),
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "[IP]", "pii", "ipv4"),
    (re.compile(r"\bTR\d{2}\s?(?:\d{4}\s?){5}\d{2}\b"), "[IBAN]", "pii", "iban_tr"),
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), "[IBAN]", "pii", "iban"),
    # Card numbers must be either bare (13-19 digits, no separators) or
    # properly grouped. A per-digit `[ -]?` lets citation ranges such as
    # "Berleant [1985 ... 2013]" through as a Luhn-valid 16-digit number.
    (re.compile(r"(?<![\d])(?:\d{4} \d{4} \d{4} \d{4}"
                r"|\d{4}-\d{4}-\d{4}-\d{4}"
                r"|\d{4} \d{6} \d{5}"
                r"|\d{13,19})(?![\d])"), None, "pii", "card_candidate"),
]

# 11-digit Turkish national id (TCKN) — cheap checksum
_TCKN = re.compile(r"(?<!\d)[1-9]\d{10}(?!\d)")


def _luhn_ok(s: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", s)]
    if not 13 <= len(digits) <= 19:
        return False
    if len(set(digits)) == 1:
        return False
    total, alt = 0, False
    for d in reversed(digits):
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


# --------------------------------------------------------------------------
# Placeholder / known-example suppression.
#
# Without this the gate cries wolf on `your_client_id`, `your_api_key` and the
# canonical AWS documentation credentials (AKIAIOSFODNN7EXAMPLE). A gate that
# cries wolf gets ignored, and then misses a real key. Every entry here was
# found by running the gate on our own spine and inspecting the hits.
# --------------------------------------------------------------------------

PLACEHOLDER_TOKENS = (
    "your_", "your-", "yourkey", "_here", "-here", "xxxx", "example",
    "placeholder", "changeme", "change_me", "dummy", "fake", "sample",
    "redacted", "insert_", "<your", "todo", "test_key", "testkey",
    "my_secret", "mykey", "my_key", "abcdef123456", "insecure",
)

KNOWN_EXAMPLE_VALUES = {
    # AWS public documentation credentials
    "akiaiosfodnn7example",
    "wjalrxutnfemi/k7mdeng/bpxrficyexamplekey",
    "akiaiosfodnn7examplekey",
    # network noise, not personal data
    "0.0.0.0", "127.0.0.1", "255.255.255.255", "localhost",
    # obviously-fake JWT used in every JWT tutorial
    "eyjhbgcioijiuzi1nij9.eyjzdwiioiixmjm0nty3odkwiiwibmftzsi6ikpvag4grg9liwladmirrjoijtiwmzgwmdaifq.signature",
}

KNOWN_TEST_CARDS = {
    "4111111111111111", "4242424242424242", "5555555555554444",
    "378282246310005", "371449635398431", "6011111111111117",
    "30569309025904", "38520000023237", "3530111333300000",
    "4012888888881881", "4222222222222", "4000056655665556",
}


def is_placeholder(value: str, severity: str = "secret") -> bool:
    """True for obvious dummy values.

    Severity-aware on purpose: the entropy floor is right for credentials but
    WRONG for PII. A phone number is inherently low-entropy, so a naive floor
    silently stops flagging every phone number in a therapy corpus — the worst
    possible corpus to go quiet on.
    """
    if not value:
        return True
    v = value.lower().strip().strip("'\"")
    if v in KNOWN_EXAMPLE_VALUES:
        return True
    digits = re.sub(r"\D", "", v)
    if digits and digits in KNOWN_TEST_CARDS:
        return True
    core = re.sub(r"[^a-z0-9]", "", v)
    if not core:
        return True

    if severity != "secret":
        # PII: suppress only the most obvious fill-in markers.
        if any(tok in v for tok in ("<your", "your_", "your-")):
            return True
        if len(set(core)) <= 2:                       # 0000000000, 111111111
            return True
        return False

    # credentials: full suppression
    if any(tok in v for tok in PLACEHOLDER_TOKENS):
        return True
    if len(set(core)) <= 3:                           # aaaaaaaa, abababab
        return True
    if len(core) >= 8:                                # entropy floor
        import collections as _c
        import math as _m
        cnt = _c.Counter(core)
        n = len(core)
        ent = -sum((k / n) * _m.log2(k / n) for k in cnt.values())
        if ent < 2.5:
            return True
    return False


def _value_of(match: re.Match) -> str:
    """The part of the match that is the actual secret, not the surrounding key=."""
    if match.groups():
        for g in match.groups():
            if g:
                return g
    return match.group(0)


def _tckn_ok(s: str) -> bool:
    if len(s) != 11 or s[0] == "0":
        return False
    d = [int(c) for c in s]
    odd = sum(d[0:9:2])
    even = sum(d[1:9:2])
    if (odd * 7 - even) % 10 != d[9]:
        return False
    return sum(d[:10]) % 10 == d[10]


def _phone_ok(s: str) -> bool:
    """Structural check so port lists and version strings are not phone calls.

    "llama-cpp-gpu 8080 8081 8082" matches the shape of a phone number. Real
    phones carry a cue: a + country code, parentheses, or a -/. separator; a
    space-only number has a short (2-3 digit) leading group, as in UK
    "020 7946 0958". Four 4-digit space-separated groups is a port list.
    """
    digits = re.sub(r"\D", "", s)
    if not (7 <= len(digits) <= 15):
        return False
    if len(set(digits)) <= 3:
        return False
    if re.match(r"\s*\+", s) or "(" in s or "-" in s or "." in s:
        return True
    groups = s.split()
    return bool(groups) and len(groups[0]) in (2, 3)


def scan_text(text: str, mode: str = "redact") -> tuple[str, list[tuple[str, str]]]:
    """Return (possibly redacted text, [(severity, label), ...])."""
    hits: list[tuple[str, str]] = []
    out = text

    for pattern, repl, severity, label in DETECTORS:
        if label == "phone":
            def _ph(m):
                s = m.group(0)
                if _phone_ok(s) and not is_placeholder(s, "pii"):
                    hits.append((severity, "phone"))
                    return "[PHONE]"
                return s
            out = pattern.sub(_ph, out)
            continue

        if label == "card_candidate":
            def _card(m):
                s = m.group(0)
                if _luhn_ok(s) and not is_placeholder(s, "pii"):
                    hits.append((severity, "card_number"))
                    return "[CARD]"
                return s
            out = pattern.sub(_card, out)
            continue

        def _sub(m, pattern=pattern, repl=repl, severity=severity, label=label):
            if is_placeholder(_value_of(m), severity):
                return m.group(0)          # suppress the hit, leave text alone
            hits.append((severity, label))
            if mode == "redact" and repl is not None:
                return pattern.sub(repl, m.group(0), count=1)
            return m.group(0)

        out = pattern.sub(_sub, out)

    for m in _TCKN.finditer(out):
        if _tckn_ok(m.group(0)) and not is_placeholder(m.group(0), "pii"):
            hits.append(("pii", "tckn"))
            if mode == "redact":
                out = out.replace(m.group(0), "[TCKN]")
    return out, hits


# --------------------------------------------------------------------------
# Row handling
# --------------------------------------------------------------------------

DEFAULT_FIELDS = ("text", "completion", "response", "prompt", "instruction",
                  "question", "answer", "input", "output", "code")


def scan_row(rec: dict, mode: str, fields: tuple[str, ...]) -> tuple[dict, list]:
    hits: list = []
    for f in fields:
        v = rec.get(f)
        if isinstance(v, str) and v:
            new, h = scan_text(v, mode)
            if h:
                hits.extend(h)
                rec[f] = new
    if isinstance(rec.get("messages"), list):
        for m in rec["messages"]:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                new, h = scan_text(m["content"], mode)
                if h:
                    hits.extend(h)
                    m["content"] = new
    return rec, hits


# --------------------------------------------------------------------------
# Self-test — the gate must prove it can catch things
# --------------------------------------------------------------------------

SELFTEST = [
    ("clean_prose",
     {"text": "The committee adjourned after a long debate about irrigation rights."},
     False),
    ("email",
     {"text": "Contact me at jane.doe@clinic-mail.org for the transcript."}, True),
    ("phone",
     {"text": "Call the clinic on (415) 555-0142 tomorrow."}, True),
    ("openai_key",
     {"text": "export OPENAI_API_KEY=sk-proj-AbCdEf0123456789AbCdEf0123456789"}, True),
    ("hf_token",
     {"text": "use hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345 for that"}, True),
    ("github_pat",
     {"text": "token ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789 here"}, True),
    ("aws_key",
     {"text": "id AKIA2E0A8F3B244C9986 ready"}, True),
    ("private_key",
     {"text": "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY-----"}, True),
    ("jwt",
     {"text": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk"}, True),
    ("url_creds",
     {"text": "postgres://admin:hunter2secret@db.internal:5432/app"}, True),
    ("card_valid",
     {"text": "card 1234 5678 1234 5670 exp 09/29"}, True),
    ("card_invalid_luhn",
     {"text": "invoice 1234 5678 9012 3456 processed"}, False),
    # --- false-positive controls found by running the gate on the real spine ---
    ("placeholder_client_id",
     {"text": "authenticator.authenticate(client_id='your_client_id', auth='tok')"}, False),
    ("placeholder_api_key",
     {"text": 'string apiKey = "your_api_key";'}, False),
    ("aws_docs_example_key",
     {"text": "--aws-access-key AKIAIOSFODNN7EXAMPLE"}, False),
    ("test_card_number",
     {"text": "use card 4111 1111 1111 1111 for the sandbox"}, False),
    ("prime_modulus",
     {"text": "MOD(o.rn, 3)) % 1000000007 AS rolling_hash"}, False),
    ("timestamp_id",
     {"text": "| 2026040510.00 | `git clone https://github.com/repo.git` |"}, False),
    ("port_list",
     {"text": "llama-cpp-gpu 8080 8081 8082 8083 exposed"}, False),
    ("citation_year_range",
     {"text": "the immersion of the appreciator (Berleant [1985 1985b]) and later work 2013"}, False),
    ("no_false_positive_year",
     {"text": "The war ended in 1945 after six years of fighting across Europe."}, False),
    ("no_false_positive_math",
     {"text": "Then 180 + 100 = 280 pages over 14 days, so 20 pages per day."}, False),
]


def run_self_test() -> bool:
    print("pii_gate.py — SELF TEST")
    ok_all = True
    for label, row, expect_hit in SELFTEST:
        _new, hits = scan_row(dict(row), mode="redact", fields=DEFAULT_FIELDS)
        got = bool(hits)
        ok = got == expect_hit
        ok_all = ok_all and ok
        kinds = ",".join(sorted({h[1] for h in hits})) or "-"
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:24s} expected_hit={str(expect_hit):5s} "
              f"got={str(got):5s} kinds={kinds}")

    # redaction must actually remove the value
    r, _ = scan_row({"text": "mail jane.doe@clinic-mail.org now"}, "redact", DEFAULT_FIELDS)
    ok = "jane.doe@clinic-mail.org" not in r["text"] and "[EMAIL]" in r["text"]
    print(f"  [{'PASS' if ok else 'FAIL'}] redaction_removes_value")
    ok_all = ok_all and ok

    # flag mode must NOT mutate
    r2, h2 = scan_row({"text": "mail jane.doe@clinic-mail.org now"}, "flag", DEFAULT_FIELDS)
    ok = "jane.doe@clinic-mail.org" in r2["text"] and h2
    print(f"  [{'PASS' if ok else 'FAIL'}] flag_mode_preserves_text")
    ok_all = ok_all and ok

    print(f"SELF TEST: {'ALL PASS' if ok_all else 'FAILURES'}")
    return ok_all


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def cmd_run(args) -> int:
    total = 0
    rows_with_hits = rows_dropped = 0
    by_label = collections.Counter()
    secrets = 0
    examples: list[dict] = []

    fields = tuple(f.strip() for f in args.fields.split(",") if f.strip())

    fin = open(args.input, encoding="utf-8")
    fout = open(args.output, "w", encoding="utf-8")
    try:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                fout.write(line if line.endswith("\n") else line + "\n")
                continue
            rec, hits = scan_row(rec, args.mode, fields)
            if hits:
                rows_with_hits += 1
                for sev, lab in hits:
                    by_label[lab] += 1
                    if sev == "secret":
                        secrets += 1
                if len(examples) < 20:
                    examples.append({"labels": sorted({h[1] for h in hits}),
                                     "severity": sorted({h[0] for h in hits})})
                if args.mode == "drop":
                    rows_dropped += 1
                    continue
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        fin.close()
        fout.close()

    report = {
        "gate": "pii/v1",
        "input": os.path.abspath(args.input),
        "output": os.path.abspath(args.output),
        "mode": args.mode,
        "fields_scanned": list(fields),
        "rows_total": total,
        "rows_dropped": rows_dropped,
        "rows_with_hits": rows_with_hits,
        "rows_out": total - rows_dropped,
        "hits_by_label": dict(by_label.most_common()),
        "secret_hits": secrets,
        "verdict": "SECRETS_FOUND" if secrets else ("PII_REDACTED" if rows_with_hits else "CLEAN"),
        "examples": examples,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

    print(f"[pii] mode={args.mode} rows={total} with_hits={rows_with_hits} "
          f"dropped={rows_dropped} secrets={secrets}")
    if by_label:
        print(f"[pii] hits={dict(by_label.most_common(15))}")
    print(f"[pii] VERDICT={report['verdict']}")
    return 4 if secrets else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="G-PII: PII + secret detection and redaction.")
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--report")
    ap.add_argument("--mode", choices=["redact", "flag", "drop"], default="redact")
    ap.add_argument("--fields", default=",".join(DEFAULT_FIELDS))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if run_self_test() else 2
    if not args.input or not args.output:
        ap.error("--input and --output required (or --self-test)")
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
