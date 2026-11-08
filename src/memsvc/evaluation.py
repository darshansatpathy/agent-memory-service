"""A benchmark for agent memory: users whose facts change, conversations to recall, PII, TTLs and other tenants.

Questions are worded differently from the memories they should find. Facts change (a user moves city,
changes job), so a memory that keeps every version returns stale answers; the ablation measures that.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from memsvc import pii
from memsvc.service import Kind, MemoryService, Mode

CITIES = [
    "Lisbon",
    "Porto",
    "Berlin",
    "Toronto",
    "Austin",
    "Nairobi",
    "Osaka",
    "Melbourne",
    "Bogota",
    "Krakow",
    "Dublin",
    "Seoul",
    "Denver",
    "Lyon",
]
JOBS = [
    "nurse",
    "data engineer",
    "high-school teacher",
    "architect",
    "pharmacist",
    "product designer",
    "electrician",
    "accountant",
    "chef",
    "paramedic",
]
PETS = [
    ("dog", "Biscuit"),
    ("cat", "Miso"),
    ("parrot", "Kiwi"),
    ("rabbit", "Clover"),
    ("dog", "Pepper"),
    ("cat", "Tofu"),
    ("tortoise", "Sheldon"),
]
DIETS = ["vegetarian", "vegan", "pescatarian", "gluten-free", "keto"]
SPORTS = ["bouldering", "trail running", "sailing", "tennis", "swimming", "cycling", "judo"]
LANGS = ["Japanese", "Portuguese", "German", "Korean", "Spanish", "Italian", "Swahili"]
PLACES = ["Iceland", "Patagonia", "Vietnam", "Morocco", "the Dolomites", "Tasmania", "Peru", "Norway", "Sri Lanka", "Scotland"]
TOPICS = [
    ("a Python error about circular imports", "the bug with modules importing each other"),
    ("a cover letter for a hospital job", "the job application letter"),
    ("a dispute with the landlord about the deposit", "the argument over getting the rental deposit back"),
    ("a budget spreadsheet for the wedding", "the wedding costs"),
    ("a training plan for a half marathon", "preparing for the 21 km race"),
    ("a birthday gift for their sister", "what to buy their sibling"),
    ("tax forms for freelance income", "filing taxes on self-employment earnings"),
    ("slow Wi-Fi in the home office", "the internet connection being sluggish at their desk"),
]


@dataclass
class Fact:
    subject: str
    statement: str
    question: str


@dataclass
class Case:
    namespace: str
    question: str
    expected: str  # memory key
    stale: str | None = None  # key of a superseded value
    kind: str = "fact"


@dataclass
class Dataset:
    writes: list[tuple[str, Kind, str, str | None, dt.datetime, str]] = field(default_factory=list)  # ns, kind, text, subject, when, key
    cases: list[Case] = field(default_factory=list)
    pii_values: list[str] = field(default_factory=list)
    blocked: list[tuple[str, str]] = field(default_factory=list)


def generate(users: int = 30, seed: int = 7) -> Dataset:
    rnd = random.Random(seed)
    now = dt.datetime.now(dt.UTC)
    d = Dataset()
    for u in range(users):
        ns = f"user-{u:03d}"
        t0 = now - dt.timedelta(days=200)

        def add(kind: Kind, text: str, subject: str | None, days: float, key: str, ns: str = ns, t0: dt.datetime = t0) -> None:
            d.writes.append((ns, kind, text, subject, t0 + dt.timedelta(days=days), key))

        city, city2 = rnd.sample(CITIES, 2)
        job, job2 = rnd.sample(JOBS, 2)
        pet, pet_name = rnd.choice(PETS)
        diet, sport, lang = rnd.choice(DIETS), rnd.choice(SPORTS), rnd.choice(LANGS)
        add("semantic", f"I live in {city} with my partner.", "user.city", 1, "city1")
        add("semantic", f"I work as a {job}.", "user.job", 2, "job1")
        add("semantic", f"We have a {pet} named {pet_name}.", "user.pet", 3, "pet")
        add("semantic", f"I eat a {diet} diet.", "user.diet", 4, "diet")
        add("semantic", f"On weekends I usually go {sport}.", "user.sport", 5, "sport")
        add("semantic", f"I've been learning {lang} for a year.", "user.language", 6, "lang")
        moved = rnd.random() < 0.6
        changed = rnd.random() < 0.5
        if moved:
            add("semantic", f"Big news: we just relocated to {city2}.", "user.city", 120 + rnd.random() * 40, "city2")
        if changed:
            add("semantic", f"I left that job and started as a {job2}.", "user.job", 100 + rnd.random() * 60, "job2")
        d.cases += [
            Case(ns, "Which city is the user based in these days?", "city2" if moved else "city1", "city1" if moved else None),
            Case(ns, "What does the user do for a living?", "job2" if changed else "job1", "job1" if changed else None),
            Case(ns, f"What is the name of their {pet}?", "pet"),
            Case(ns, "Are there foods the user avoids or a way they eat?", "diet"),
            Case(ns, "What hobby keeps them busy on Saturdays and Sundays?", "sport"),
            Case(ns, "Which foreign language are they studying?", "lang"),
        ]
        for i, (topic, paraphrase) in enumerate(rnd.sample(TOPICS, 4)):
            add("episodic", f"Helped the user with {topic}.", None, 20 + i * 30 + rnd.random() * 10, f"topic{i}")
            d.cases.append(Case(ns, f"Have we talked before about {paraphrase}?", f"topic{i}", kind="episode"))
        for i, place in enumerate(rnd.sample(PLACES, 2)):
            add("episodic", f"Planned a two-week trip to {place}, comparing flights and hotels.", None, 60 + i * 50, f"trip{i}")
            d.cases.append(Case(ns, f"What did we work out for the {place} holiday?", f"trip{i}", kind="episode"))
        # Identifiers: two memories that differ only by an order number. Embeddings blur these; text search does not.
        oids = [
            f"{rnd.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rnd.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}-{rnd.randint(10000, 99999)}"
            for _ in range(2)
        ]
        add("episodic", f"Order {oids[0]} arrived damaged, so we arranged a replacement.", None, 170, "order0")
        add("episodic", f"Order {oids[1]} is stuck in customs, so we chased the courier.", None, 172, "order1")
        d.cases += [
            Case(ns, f"What happened with order {oids[0]}?", "order0", kind="identifier"),
            Case(ns, f"Any news on order {oids[1]}?", "order1", kind="identifier"),
        ]
        email = f"{rnd.choice(['ana', 'li', 'sam', 'joe', 'mia'])}.{u}@example.org"
        phone = f"+44 7700 9{u:02d} {rnd.randint(100, 999)}"
        d.pii_values += [email, phone]
        add("episodic", f"The user asked to send the booking confirmation to {email} and to text {phone}.", None, 150, "contact")
        card = rnd.choice(["4111 1111 1111 1111", "5555 5555 5555 4444", "4012 8888 8888 1881"])
        d.blocked.append((ns, f"Charge the deposit to my card {card} please."))
    return d


def ranks(hit_keys: list[str], key: str) -> int | None:
    return hit_keys.index(key) + 1 if key in hit_keys else None


async def run(
    svc: MemoryService, tenant: str, data: Dataset, modes: list[Mode], *, append_only: bool = False, k: int = 5
) -> dict[str, Any]:
    ids: dict[tuple[str, str], str] = {}
    by_id: dict[str, tuple[str, str]] = {}
    for ns, kind, text, subject, when, key in data.writes:
        r = await svc.remember(tenant, ns, kind, text, subject=None if append_only else subject, event_at=when, ttl=None)
        ids[(ns, key)] = r.id
        by_id[r.id] = (ns, key)
    blocked = 0
    for ns, text in data.blocked:
        try:
            await svc.remember(tenant, ns, "episodic", text)
        except Exception:
            blocked += 1
    out: dict[str, Any] = {"users": len({w[0] for w in data.writes}), "memories": len(data.writes), "cases": len(data.cases)}
    out |= {"blocked": blocked, "attempted_blocked": len(data.blocked), "modes": {}}
    for mode in modes:
        rows = []
        latencies = []
        leaks = 0
        for c in data.cases:
            t = time.perf_counter()
            hits = await svc.recall(tenant, c.namespace, c.question, k=k, mode=mode)
            latencies.append(time.perf_counter() - t)
            keys = []
            for h in hits:
                owner = by_id.get(h.id)
                if owner is None or owner[0] != c.namespace:
                    leaks += 1
                    continue
                keys.append(owner[1])
            rank = ranks(keys, c.expected)
            stale_first = bool(c.stale and keys and keys[0] == c.stale)
            rows.append({"kind": c.kind, "rank": rank, "stale_first": stale_first, "changed": c.stale is not None})
        out["modes"][mode] = summarise(rows, latencies, leaks)
    exported = [m["content"] for ns in {w[0] for w in data.writes} for m in await svc.export(tenant, ns)]
    out["pii_values_stored"] = sum(1 for v in data.pii_values for content in exported if v in content)
    out["pii_values_total"] = len(data.pii_values)
    return out


def summarise(rows: list[dict[str, Any]], latencies: list[float], leaks: int) -> dict[str, Any]:
    def agg(sel: list[dict[str, Any]]) -> dict[str, float]:
        n = len(sel) or 1
        return {
            "n": len(sel),
            "hit@1": sum(1 for r in sel if r["rank"] == 1) / n,
            "recall@5": sum(1 for r in sel if r["rank"]) / n,
            "mrr": sum(1 / r["rank"] for r in sel if r["rank"]) / n,
        }

    changed = [r for r in rows if r["changed"]]
    lat = sorted(latencies)
    return {
        "all": agg(rows),
        "facts": agg([r for r in rows if r["kind"] == "fact"]),
        "episodes": agg([r for r in rows if r["kind"] == "episode"]),
        "identifiers": agg([r for r in rows if r["kind"] == "identifier"]),
        "changed_facts": agg(changed) | {"stale_first": sum(r["stale_first"] for r in changed) / (len(changed) or 1)},
        "leaks": leaks,
        "p50_ms": statistics.median(lat) * 1000,
        "p95_ms": lat[int(0.95 * (len(lat) - 1))] * 1000,
    }


def pii_benchmark(seed: int = 11, n: int = 400) -> dict[str, dict[str, float]]:
    """Precision and recall of the scrubber on generated sentences, half with planted values, plus lookalikes."""
    rnd = random.Random(seed)
    gens = {
        "EMAIL": lambda: (
            f"{rnd.choice(['ana', 'j.doe', 'mia_k', 'li+news'])}{rnd.randint(1, 999)}@{rnd.choice(['example.com', 'mail.example.org', 'corp.example.co.uk'])}"
        ),
        "PHONE": lambda: rnd.choice(
            [f"+1 415 555 {rnd.randint(1000, 9999)}", f"(212) 555-{rnd.randint(1000, 9999)}", f"+44 20 7946 {rnd.randint(1000, 9999)}"]
        ),
        "CARD": lambda: rnd.choice(["4111 1111 1111 1111", "5555-5555-5555-4444", "378282246310005", "6011111111111117"]),
        "IBAN": lambda: rnd.choice(["DE89 3704 0044 0532 0130 00", "GB82 WEST 1234 5698 7654 32", "FR14 2004 1010 0505 0001 3M02 606"]),
        "SSN": lambda: f"{rnd.randint(100, 665)}-{rnd.randint(10, 99)}-{rnd.randint(1000, 9999)}",
        "IP": lambda: f"{rnd.randint(11, 223)}.{rnd.randint(0, 255)}.{rnd.randint(0, 255)}.{rnd.randint(1, 254)}",
        "SECRET": lambda: rnd.choice(
            [
                "sk-" + "".join(rnd.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=32)),
                "AKIA" + "".join(rnd.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567", k=16)),
            ]
        ),
    }
    lookalikes = [
        lambda: f"order #{rnd.randint(10**15, 10**16 - 1)}0 shipped",
        lambda: f"version {rnd.randint(1, 20)}.{rnd.randint(0, 9)}.{rnd.randint(0, 30)} is out",
        lambda: f"meet at {rnd.randint(8, 18)}:{rnd.choice(['00', '15', '30'])} in room {rnd.randint(100, 999)}",
        lambda: f"it costs ${rnd.randint(10, 9999)}.{rnd.randint(10, 99)}",
        lambda: f"born in {rnd.randint(1950, 2005)}, joined in {rnd.randint(2010, 2025)}",
        lambda: f"ticket {rnd.randint(100, 999)}-{rnd.randint(10, 99)}-{rnd.randint(10, 99)} updated",
    ]
    templates = ["Please note {} for later.", "The customer said {} in the chat.", "Reminder: {} was mentioned."]
    tp: dict[str, int] = dict.fromkeys(gens, 0)
    fn = dict(tp)
    fp = dict(tp)
    for i in range(n):
        kind = list(gens)[i % len(gens)]
        value = gens[kind]()
        text = rnd.choice(templates).format(value)
        found = pii.find(text)
        kinds = [f.kind for f in found]
        if kind in kinds:
            tp[kind] += 1
        else:
            fn[kind] += 1
        for k2 in kinds:
            if k2 != kind:
                fp[k2] += 1
        neg = rnd.choice(templates).format(rnd.choice(lookalikes)())
        for f in pii.find(neg):
            fp[f.kind] += 1
    return {k: {"precision": tp[k] / ((tp[k] + fp[k]) or 1), "recall": tp[k] / ((tp[k] + fn[k]) or 1), "n": tp[k] + fn[k]} for k in gens}


async def scale(svc: MemoryService, tenant: str, total: int, namespaces: int, queries: int = 200, seed: int = 3) -> dict[str, Any]:
    """Load random unit vectors and measure filtered recall latency, the part of recall the database owns."""
    import numpy as np
    from psycopg.types.json import Jsonb

    rng = np.random.default_rng(seed)
    words = ["trip", "invoice", "dog", "flight", "budget", "error", "garden", "lease", "recipe", "meeting", "train", "visa"]
    # Row-level security rules out COPY, so rows go in as batched INSERTs, each batch its own transaction.
    batch = 2000
    for start in range(0, total, batch):
        vecs = rng.normal(size=(min(batch, total - start), 384)).astype("float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        rows = [
            (tenant, f"ns-{(start + j) % namespaces}", "episodic", " ".join(rng.choice(words, 6)), f"h{start + j}", v, Jsonb({}))
            for j, v in enumerate(vecs)
        ]
        async with svc.db.tenant(tenant) as cur:
            await cur.executemany(
                "insert into memories (tenant, namespace, kind, content, content_hash, embedding, pii) values (%s, %s, %s, %s, %s, %s, %s)",
                rows,
            )
    async with svc.db.tenant(tenant) as cur:
        await cur.execute("analyze memories")

    class Fixed:
        name, dim = "fixed", 384

        def embed(self, texts: list[str]) -> Any:
            v = rng.normal(size=(len(texts), 384)).astype("float32")
            return v / np.linalg.norm(v, axis=1, keepdims=True)

    real = svc.embedder
    svc.embedder = Fixed()
    out: dict[str, Any] = {"memories": total, "namespaces": namespaces}
    try:
        for mode in ("vector", "hybrid"):
            lat = []
            for q in range(queries):
                t = time.perf_counter()
                await svc.recall(tenant, f"ns-{q % namespaces}", f"{words[q % len(words)]} {words[(q * 5) % len(words)]}", k=10, mode=mode)
                lat.append((time.perf_counter() - t) * 1000)
            lat.sort()
            out[mode] = {"p50_ms": lat[len(lat) // 2], "p95_ms": lat[int(0.95 * (len(lat) - 1))], "p99_ms": lat[int(0.99 * (len(lat) - 1))]}
    finally:
        svc.embedder = real
    return out


def report(
    main: dict[str, Any], ablation: dict[str, Any], pii_scores: dict[str, dict[str, float]], scale_r: dict[str, Any], embedder: str
) -> str:
    L = [
        "# Evaluation",
        "",
        f"Generated by `python bench/evaluate.py`. Embedder `{embedder}`, PostgreSQL with pgvector (HNSW, cosine). "
        f"{main['memories']} memories for {main['users']} simulated users; {main['cases']} questions, "
        "each worded differently from the memory it should find.",
        "",
        "## Recall by retrieval mode",
        "",
        "Facts, episodes, order numbers and changed facts are hit@1 within each kind of question. Hybrid mixes 0.8 of the scaled vector similarity "
        "with 0.2 of the scaled text score.",
        "",
        "| Mode | Hit@1 | Recall@5 | MRR | Facts | Episodes | Order numbers | Changed facts | Stale answer first | Cross-user results | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for mode, m in main["modes"].items():
        L.append(
            f"| {mode} | {m['all']['hit@1']:.3f} | {m['all']['recall@5']:.3f} | {m['all']['mrr']:.3f} | {m['facts']['hit@1']:.3f} | "
            f"{m['episodes']['hit@1']:.3f} | {m['identifiers']['hit@1']:.3f} | {m['changed_facts']['hit@1']:.3f} | "
            f"{m['changed_facts']['stale_first']:.3f} | {m['leaks']} | {m['p95_ms']:.1f} |"
        )
    a = ablation["modes"]["hybrid"]
    h = main["modes"]["hybrid"]
    L += [
        "",
        "## Why supersession matters",
        "",
        f"The same data stored append-only (no `subject`, so a new value does not replace the old one): on the "
        f"{h['changed_facts']['n']} questions about facts that changed, hybrid recall puts the **outdated value first "
        f"{a['changed_facts']['stale_first']:.0%}** of the time and the current one first {a['changed_facts']['hit@1']:.0%}; "
        f"with supersession, stale first {h['changed_facts']['stale_first']:.0%}, current first {h['changed_facts']['hit@1']:.0%}.",
        "",
        "## Personal data",
        "",
        f"{main['pii_values_total']} email addresses and phone numbers were written inside memories; **{main['pii_values_stored']} reached the database** "
        f"(they are stored as keyed pseudonyms). {main['blocked']} of {main['attempted_blocked']} memories containing a card number were refused.",
        "",
        "Scrubber on generated sentences, half carrying a planted value and half lookalikes (order numbers, versions, times, prices, years, ticket ids):",
        "",
        "| Kind | Precision | Recall | Planted |",
        "|---|---|---|---|",
    ]
    L += [f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['n']:.0f} |" for k, v in pii_scores.items()]
    L += [
        "",
        "## Latency at scale",
        "",
        f"{scale_r['memories']:,} memories across {scale_r['namespaces']:,} namespaces in one tenant; 200 recalls of the top 10 in one namespace, "
        "with the embedding precomputed so the numbers are the database's. HNSW with iterative scans keeps the namespace filter from starving results.",
        "",
        "| Mode | p50 ms | p95 ms | p99 ms |",
        "|---|---|---|---|",
    ]
    L += [f"| {m} | {scale_r[m]['p50_ms']:.1f} | {scale_r[m]['p95_ms']:.1f} | {scale_r[m]['p99_ms']:.1f} |" for m in ("vector", "hybrid")]
    return "\n".join(L) + "\n"


def main_sync(url: str, embedder_name: str, users: int, scale_n: int, out: str | None) -> tuple[str, dict[str, Any]]:
    from memsvc.db import Database
    from memsvc.embed import get

    async def go() -> tuple[str, dict[str, Any]]:
        db = Database(url, max_size=4)
        await db.open()
        svc = MemoryService(db, get(embedder_name))
        stamp = int(time.time())
        try:
            data = generate(users)
            for t in (f"eval-{stamp}", f"eval-ablation-{stamp}", f"eval-scale-{stamp}"):
                await svc.create_tenant(t)
            m = await run(svc, f"eval-{stamp}", data, ["vector", "text", "hybrid"])
            a = await run(svc, f"eval-ablation-{stamp}", data, ["hybrid"], append_only=True)
            s = await scale(svc, f"eval-scale-{stamp}", scale_n, max(1, scale_n // 100))
            text = report(m, a, pii_benchmark(), s, svc.embedder.name)
        finally:
            for t in (f"eval-{stamp}", f"eval-ablation-{stamp}", f"eval-scale-{stamp}"):
                async with db.tenant(t) as cur:
                    await cur.execute("delete from memories")
            await db.close()
        if out:
            from pathlib import Path

            Path(out).write_text(text)
        return text, m

    return asyncio.run(go())
