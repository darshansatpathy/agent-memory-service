"""Find and scrub personal data before anything is embedded or stored.

Detectors are patterns plus validators, so a 16-digit order number that fails the Luhn check is not
a card, and 999.1.2.3 is not an address. Each kind has an action:

- `block`: refuse the memory (secrets and card numbers have no business in agent memory);
- `pseudonymise`: replace with a stable keyed token, `[EMAIL:3f9a1c2e]`, so the agent can still tell
  that two memories mention the same address without the address being stored;
- `redact`: replace with the kind, `[PHONE]`.

What was found is recorded as kinds and counts, never values.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

Action = Literal["block", "pseudonymise", "redact", "keep"]


def luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def iban_ok(raw: str) -> bool:
    s = raw.replace(" ", "").upper()
    if not 15 <= len(s) <= 34:
        return False
    rearranged = s[4:] + s[:4]
    return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1


def ssn_ok(raw: str) -> bool:
    area, group, serial = raw.split("-")
    return area not in {"000", "666"} and not area.startswith("9") and group != "00" and serial != "0000"


def ipv4_ok(raw: str) -> bool:
    try:
        ip = ipaddress.ip_address(raw)
    except ValueError:
        return False
    return not (ip.is_loopback or ip.is_unspecified)


def phone_ok(raw: str) -> bool:
    digits = re.sub(r"\D", "", raw)
    return 10 <= len(digits) <= 15 and len(set(digits)) > 2


def issuer_ok(d: str) -> bool:
    """Issuer prefixes and lengths of the major networks, so a long order number that happens to pass Luhn is not a card."""
    n, p2, p4 = len(d), int(d[:2]), int(d[:4])
    return (
        (d[0] == "4" and n in (13, 16, 19))
        or ((51 <= p2 <= 55 or 2221 <= p4 <= 2720) and n == 16)
        or (p2 in (34, 37) and n == 15)
        or ((d.startswith("6011") or d.startswith("65") or 644 <= int(d[:3]) <= 649) and 16 <= n <= 19)
        or (3528 <= p4 <= 3589 and 16 <= n <= 19)
        or ((p2 in (36, 38) or 300 <= int(d[:3]) <= 305) and n == 14)
        or (p2 == 62 and 16 <= n <= 19)
    )


def card_ok(raw: str) -> bool:
    digits = re.sub(r"\D", "", raw)
    return 13 <= len(digits) <= 19 and issuer_ok(digits) and luhn(digits)


@dataclass(frozen=True)
class Detector:
    kind: str
    pattern: re.Pattern[str]
    valid: Callable[[str], bool] = lambda _: True
    group: int = 0


DETECTORS: list[Detector] = [
    # Secrets first: their patterns can contain digits other detectors would claim.
    Detector(
        "SECRET",
        re.compile(
            r"\b(?:sk-[A-Za-z0-9_-]{20,}|sk_live_[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{40,}|"
            r"xox[abpr]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})"
        ),
    ),
    Detector("SECRET", re.compile(r"(?i)\b(?:password|passwd|pwd|passcode|api[_ -]?key|secret)\s*(?:is|:|=)\s*(\S{4,})"), group=1),
    Detector("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")),
    Detector("CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), card_ok),
    Detector("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b"), iban_ok),
    Detector("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), ssn_ok),
    Detector("IP", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), ipv4_ok),
    # Not inside a longer token or a decimal, but a full stop ending the sentence is fine.
    Detector(
        "PHONE",
        re.compile(r"(?<!\w)(?<!\d\.)(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{2,4}(?:[ .-]\d{2,4}){1,4}(?!\w)(?!\.\d)"),
        phone_ok,
    ),
]

DEFAULT_POLICY: dict[str, Action] = {
    "SECRET": "block",
    "CARD": "block",
    "IBAN": "block",
    "SSN": "block",
    "EMAIL": "pseudonymise",
    "PHONE": "pseudonymise",
    "IP": "redact",
}


@dataclass(frozen=True)
class Finding:
    kind: str
    start: int
    end: int


@dataclass
class Scrubbed:
    text: str
    findings: list[Finding] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)

    @property
    def kinds(self) -> dict[str, int]:
        return dict(Counter(f.kind for f in self.findings))


def find(text: str) -> list[Finding]:
    """Non-overlapping findings, earliest detector winning a contested span."""
    taken: list[tuple[int, int]] = []
    out: list[Finding] = []
    for d in DETECTORS:
        for m in d.pattern.finditer(text):
            start, end = m.span(d.group)
            value = m.group(d.group)
            if any(start < e and s < end for s, e in taken) or not d.valid(value):
                continue
            taken.append((start, end))
            out.append(Finding(d.kind, start, end))
    return sorted(out, key=lambda f: f.start)


def pseudonym(kind: str, value: str, key: bytes) -> str:
    norm = value.strip().lower() if kind == "EMAIL" else re.sub(r"\D", "", value)
    return f"[{kind}:{hmac.new(key, f'{kind}:{norm}'.encode(), hashlib.sha256).hexdigest()[:8]}]"


def scrub(text: str, key: bytes, policy: dict[str, Action] | None = None) -> Scrubbed:
    policy = policy or DEFAULT_POLICY
    findings = find(text)
    blocked = sorted({f.kind for f in findings if policy.get(f.kind, "redact") == "block"})
    pieces: list[str] = []
    last = 0
    for f in findings:
        action = policy.get(f.kind, "redact")
        pieces.append(text[last : f.start])
        value = text[f.start : f.end]
        if action == "keep":
            pieces.append(value)
        elif action == "pseudonymise":
            pieces.append(pseudonym(f.kind, value, key))
        else:
            pieces.append(f"[{f.kind}]")
        last = f.end
    pieces.append(text[last:])
    return Scrubbed("".join(pieces), findings, blocked)
