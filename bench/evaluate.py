"""Run the memory benchmark and write docs/evaluation.md.

MEMSVC_DATABASE_URL=postgresql://... python bench/evaluate.py --users 30 --scale 100000 --out docs/evaluation.md
"""

from __future__ import annotations

import argparse

from memsvc.config import Settings
from memsvc.db import migrate
from memsvc.evaluation import main_sync

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=30)
    ap.add_argument("--scale", type=int, default=100_000)
    ap.add_argument("--embedder", default="minilm")
    ap.add_argument("--out")
    ap.add_argument("--min-hit1", type=float, help="fail if hybrid hit@1 falls below this")
    ap.add_argument("--max-stale", type=float, help="fail if hybrid returns an outdated fact first more often than this")
    a = ap.parse_args()
    s = Settings()
    migrate(s.database_url)
    text, results = main_sync(s.database_url, a.embedder, a.users, a.scale, a.out)
    print(text)
    hybrid = results["modes"]["hybrid"]
    failures = []
    if a.min_hit1 is not None and hybrid["all"]["hit@1"] < a.min_hit1:
        failures.append(f"hybrid hit@1 {hybrid['all']['hit@1']:.3f} < {a.min_hit1}")
    if a.max_stale is not None and hybrid["changed_facts"]["stale_first"] > a.max_stale:
        failures.append(f"stale-first {hybrid['changed_facts']['stale_first']:.3f} > {a.max_stale}")
    if hybrid["leaks"] or results["pii_values_stored"]:
        failures.append("cross-user results or personal data stored")
    if failures:
        raise SystemExit("benchmark gate failed: " + "; ".join(failures))
