from __future__ import annotations

import asyncio
import datetime as dt

import psycopg
import pytest

from memsvc.service import Blocked, content_hash, text_query


async def test_semantic_subject_supersedes_and_keeps_history(svc):
    a = await svc.remember("acme", "u1", "semantic", "The user lives in Lisbon.", subject="user.city")
    b = await svc.remember("acme", "u1", "semantic", "The user moved to Porto.", subject="user.city")
    assert b.superseded == [a.id]
    hits = await svc.recall("acme", "u1", "where does the user live", kinds=["semantic"])
    assert [h.content for h in hits] == ["The user moved to Porto."]
    hist = await svc.recall("acme", "u1", "where does the user live", kinds=["semantic"], include_history=True)
    assert {h.content for h in hist} == {"The user moved to Porto.", "The user lives in Lisbon."}
    assert next(h for h in hist if "Lisbon" in h.content).superseded
    assert [r["content"] for r in await svc.history("acme", "u1", "user.city")] == ["The user lives in Lisbon.", "The user moved to Porto."]


async def test_exact_repeat_is_a_duplicate_not_a_second_row(svc):
    a = await svc.remember("acme", "u1", "episodic", "Asked about the Japan trip budget.")
    b = await svc.remember("acme", "u1", "episodic", "  asked about the JAPAN trip budget. ")
    assert b.status == "duplicate" and b.id == a.id
    m = await svc.get("acme", a.id)
    assert m["seen_count"] == 2


async def test_near_duplicate_facts_merge(svc):
    svc.policy.merge_similarity = 0.8
    a = await svc.remember("acme", "u1", "semantic", "The user prefers window seats on long flights")
    b = await svc.remember("acme", "u1", "semantic", "The user prefers window seats on long flights always")
    assert b.status == "merged" and b.id == a.id
    c = await svc.remember("acme", "u1", "semantic", "The user is allergic to nothing and loves hiking")
    assert c.status == "created"


async def test_pii_is_scrubbed_or_blocked(svc):
    r = await svc.remember("acme", "u1", "episodic", "Send the itinerary to ana@example.com or call +351 912 345 678")
    assert "ana@" not in r.content and "[EMAIL:" in r.content and "[PHONE:" in r.content
    assert r.pii == {"EMAIL": 1, "PHONE": 1}
    m = await svc.get("acme", r.id)
    assert "912" not in m["content"] and m["pii"] == {"EMAIL": 1, "PHONE": 1}
    for text in ["my card is 4242 4242 4242 4242", "the api key is sk-live-abcdefghijklmnopqrstuvwxyz12", "password: hunter22"]:
        with pytest.raises(Blocked):
            await svc.remember("acme", "u1", "episodic", text)
    # The same address pseudonymises to the same token, so the agent can still link mentions.
    r2 = await svc.remember("acme", "u1", "episodic", "Ana's address ana@example.com again")
    assert r.content.split()[4] == r2.content.split()[2]


async def test_ttl_and_sweep(svc):
    r = await svc.remember("acme", "u1", "episodic", "Ephemeral note about lunch", ttl=dt.timedelta(seconds=1))
    keep = await svc.remember("acme", "u1", "semantic", "The user is vegetarian", ttl=None)
    assert keep.expires_at is None and r.expires_at is not None
    await asyncio.sleep(1.2)
    assert await svc.get("acme", r.id) is None
    assert not any(h.id == r.id for h in await svc.recall("acme", "u1", "lunch note", k=10))
    assert await svc.sweep() >= 1
    assert await svc.get("acme", keep.id) is not None


async def test_episodic_defaults_to_a_ttl_semantic_does_not(svc):
    e = await svc.remember("acme", "u1", "episodic", "Talked about the weather")
    s = await svc.remember("acme", "u1", "semantic", "The user's name is Ana")
    assert e.expires_at is not None and abs((e.expires_at - dt.datetime.now(dt.UTC)).total_seconds() - 90 * 86400) < 60
    assert s.expires_at is None


async def test_tenants_and_namespaces_are_isolated(svc):
    await svc.remember("acme", "u1", "semantic", "The user works at a bakery")
    assert await svc.recall("globex", "u1", "bakery work") == []
    assert await svc.recall("acme", "u2", "bakery work") == []
    assert len(await svc.recall("acme", "u1", "bakery work")) == 1


async def test_row_level_security_holds_without_a_where_clause(svc):
    await svc.remember("acme", "u1", "semantic", "Acme secret fact")
    await svc.remember("globex", "u1", "semantic", "Globex secret fact")
    async with svc.db.tenant("globex") as cur:
        await cur.execute("select content from memories")
        assert [r["content"] for r in await cur.fetchall()] == ["Globex secret fact"]
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            await cur.execute(
                "insert into memories (tenant, namespace, kind, content, content_hash, embedding) values ('acme', 'u1', 'semantic', 'x', 'x', %s)",
                ([0.1] * 384,),
            )
    async with svc.db.system() as cur:  # no tenant set: nothing is visible
        await cur.execute("select count(*) as n from memories")
        assert (await cur.fetchone())["n"] == 0


async def test_forget_and_export(svc):
    a = await svc.remember("acme", "u1", "semantic", "The user has a dog named Biscuit")
    await svc.remember("acme", "u1", "episodic", "Discussed dog food brands")
    await svc.remember("acme", "u2", "semantic", "Other user fact")
    assert len(await svc.export("acme", "u1")) == 2
    assert await svc.forget("acme", a.id) and not await svc.forget("acme", a.id)
    assert await svc.forget_namespace("acme", "u1") == 1
    assert await svc.export("acme", "u1") == [] and len(await svc.export("acme", "u2")) == 1
    async with svc.db.tenant("acme") as cur:
        await cur.execute("select reason, count from erasures where namespace = 'u1' order by id")
        assert [(r["reason"], r["count"]) for r in await cur.fetchall()] == [("memory", 1), ("namespace", 1)]


async def test_modes_and_recency(svc):
    old = await svc.remember("acme", "u1", "episodic", "Planned a trip to Kyoto", event_at=dt.datetime.now(dt.UTC) - dt.timedelta(days=120))
    new = await svc.remember("acme", "u1", "episodic", "Planned a trip to Osaka", event_at=dt.datetime.now(dt.UTC))
    hits = await svc.recall("acme", "u1", "planned trip", k=2, mode="hybrid")
    assert [h.id for h in hits] == [new.id, old.id]
    assert all(h.vector_rank and h.text_rank for h in hits)
    text_only = await svc.recall("acme", "u1", "planned trip", mode="text")
    assert all(h.vector_rank is None for h in text_only)
    flat = await svc.recall("acme", "u1", "planned trip", k=2, recency=False, mode="hybrid")
    assert abs(flat[0].score - flat[1].score) < abs(hits[0].score - hits[1].score)


def test_helpers():
    assert content_hash(" A  b ") == content_hash("a b")
    assert text_query("Where does the user live?") == "where | does | the | user | live"
    assert text_query("?!") == "zzzznothing"


async def test_episodic_subject_is_refused(svc):
    with pytest.raises(ValueError, match="subject"):
        await svc.remember("acme", "u1", "episodic", "x y z", subject="user.city")


async def test_bulk_erase_is_not_quadratic(svc):
    """Deleting memories checks the self-reference; without an index on superseded_by this took minutes at 100k rows."""
    import time

    import numpy as np

    rows = [("acme", "bulk", "episodic", f"note {i}", f"bulk{i}", np.full(384, 0.05, dtype="float32")) for i in range(3000)]
    async with svc.db.tenant("acme") as cur:
        await cur.executemany(
            "insert into memories (tenant, namespace, kind, content, content_hash, embedding) values (%s, %s, %s, %s, %s, %s)", rows
        )
    t = time.perf_counter()
    assert await svc.forget_namespace("acme", "bulk") == 3000
    assert time.perf_counter() - t < 5
    async with svc.db.tenant("acme") as cur:
        await cur.execute("select indexdef from pg_indexes where indexname = 'memories_superseded_by'")
        assert await cur.fetchone()
