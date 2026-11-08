"""Remember, recall and forget.

Write path: scrub PII, embed, then one transaction that deduplicates exact repeats, merges
near-duplicate facts, and supersedes the previous value of a subject (keeping it as history).

Read path: vector search (HNSW) and full-text search, fused by reciprocal rank, then weighted by
recency for episodic memories and by importance. Expired and superseded memories are never returned.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import math
import re
import secrets
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from memsvc import pii
from memsvc.db import Database
from memsvc.embed import Embedder

Kind = Literal["episodic", "semantic"]
Mode = Literal["hybrid", "vector", "text"]


class Blocked(ValueError):
    def __init__(self, kinds: list[str]):
        super().__init__(f"memory refused: contains {', '.join(kinds)}")
        self.kinds = kinds


@dataclass
class Policy:
    episodic_ttl: dt.timedelta | None = dt.timedelta(days=90)
    semantic_ttl: dt.timedelta | None = None
    merge_similarity: float = 0.92
    recency_half_life_days: float = 30.0
    candidates: int = 50
    vector_weight: float = 0.8
    pii: dict[str, pii.Action] = field(default_factory=lambda: dict(pii.DEFAULT_POLICY))


@dataclass
class Remembered:
    id: str
    content: str
    status: Literal["created", "duplicate", "merged"]
    superseded: list[str]
    pii: dict[str, int]
    expires_at: dt.datetime | None


@dataclass
class Hit:
    id: str
    kind: str
    namespace: str
    subject: str | None
    content: str
    score: float
    similarity: float
    vector_rank: int | None
    text_rank: int | None
    event_at: dt.datetime
    importance: float
    superseded: bool


def content_hash(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text.strip().lower()).encode()).hexdigest()


def text_query(text: str) -> str:
    """An OR query of the question's words, so a natural-language question can match a statement."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return " | ".join(dict.fromkeys(w for w in words if len(w) > 1)) or "zzzznothing"


class MemoryService:
    def __init__(self, db: Database, embedder: Embedder, policy: Policy | None = None):
        self.db, self.embedder, self.policy = db, embedder, policy or Policy()
        self._keys: dict[str, bytes] = {}

    async def _embed(self, texts: list[str]) -> np.ndarray:
        return await asyncio.to_thread(self.embedder.embed, texts)

    async def _key(self, tenant: str) -> bytes:
        if tenant not in self._keys:
            async with self.db.system() as cur:
                await cur.execute("select pseudonym_key from tenants where id = %s", (tenant,))
                row = await cur.fetchone()
            if row is None:
                raise KeyError(tenant)
            self._keys[tenant] = bytes(row["pseudonym_key"])
        return self._keys[tenant]

    # --- tenants and keys ---

    async def create_tenant(self, tenant: str) -> None:
        async with self.db.system() as cur:
            await cur.execute("reset role")
            await cur.execute("insert into tenants (id) values (%s) on conflict do nothing", (tenant,))
            await cur.execute("set role memsvc_app")

    async def create_key(self, tenant: str, label: str = "") -> str:
        key = "mk_" + secrets.token_urlsafe(32)
        async with self.db.system() as cur:
            await cur.execute("reset role")
            await cur.execute(
                "insert into api_keys (key_hash, tenant, label) values (%s, %s, %s)",
                (hashlib.sha256(key.encode()).hexdigest(), tenant, label),
            )
            await cur.execute("set role memsvc_app")
        return key

    async def authenticate(self, key: str) -> str | None:
        async with self.db.system() as cur:
            await cur.execute(
                "select tenant from api_keys where key_hash = %s and revoked_at is null", (hashlib.sha256(key.encode()).hexdigest(),)
            )
            row = await cur.fetchone()
        return str(row["tenant"]) if row else None

    # --- write ---

    async def remember(
        self,
        tenant: str,
        namespace: str,
        kind: Kind,
        content: str,
        *,
        subject: str | None = None,
        importance: float = 0.5,
        ttl: dt.timedelta | Literal["default"] | None = "default",
        event_at: dt.datetime | None = None,
        source: dict[str, Any] | None = None,
    ) -> Remembered:
        s = pii.scrub(content, await self._key(tenant), self.policy.pii)
        if s.blocked:
            raise Blocked(s.blocked)
        if kind == "episodic" and subject:
            raise ValueError("subject is for semantic memories")
        if ttl == "default":
            ttl = self.policy.episodic_ttl if kind == "episodic" else self.policy.semantic_ttl
        expires = dt.datetime.now(dt.UTC) + ttl if ttl else None
        vec = (await self._embed([s.text]))[0]
        h = content_hash(s.text)
        params = {
            "tenant": tenant,
            "ns": namespace,
            "kind": kind,
            "subject": subject,
            "content": s.text,
            "hash": h,
            "vec": vec,
            "imp": importance,
            "pii": pii_json(s.kinds),
            "source": pii_json(source or {}),
            "event_at": event_at or dt.datetime.now(dt.UTC),
            "expires": expires,
        }
        async with self.db.tenant(tenant) as cur:
            if kind == "semantic" and subject is None:
                await cur.execute(
                    """select id, content, 1 - (embedding <=> %(vec)s) as sim from memories
                       where tenant = %(tenant)s and namespace = %(ns)s and kind = 'semantic' and superseded_by is null
                         and (expires_at is null or expires_at > now())
                       order by embedding <=> %(vec)s limit 1""",
                    params,
                )
                near = await cur.fetchone()
                if near and float(near["sim"]) >= self.policy.merge_similarity and content_hash(str(near["content"])) != h:
                    await cur.execute(
                        """update memories set last_seen_at = now(), seen_count = seen_count + 1, importance = greatest(importance, %(imp)s)
                           where id = %(id)s returning expires_at""",
                        params | {"id": near["id"]},
                    )
                    row = await cur.fetchone()
                    return Remembered(str(near["id"]), str(near["content"]), "merged", [], s.kinds, row["expires_at"] if row else None)
            await cur.execute(
                """insert into memories (tenant, namespace, kind, subject, content, content_hash, embedding, importance, pii, source, event_at, expires_at)
                   values (%(tenant)s, %(ns)s, %(kind)s, %(subject)s, %(content)s, %(hash)s, %(vec)s, %(imp)s, %(pii)s, %(source)s, %(event_at)s, %(expires)s)
                   on conflict (tenant, namespace, kind, content_hash) where superseded_by is null do update
                     set last_seen_at = now(), seen_count = memories.seen_count + 1,
                         importance = greatest(memories.importance, excluded.importance),
                         expires_at = case when memories.expires_at is null or excluded.expires_at is null then null
                                           else greatest(memories.expires_at, excluded.expires_at) end
                   returning id, expires_at, (xmax = 0) as inserted""",
                params,
            )
            row = await cur.fetchone()
            assert row is not None
            new_id, inserted = row["id"], bool(row["inserted"])
            superseded: list[str] = []
            if subject and inserted:
                await cur.execute(
                    """update memories set superseded_by = %(new)s, superseded_at = now()
                       where tenant = %(tenant)s and namespace = %(ns)s and subject = %(subject)s and superseded_by is null and id <> %(new)s
                       returning id""",
                    params | {"new": new_id},
                )
                superseded = [str(r["id"]) for r in await cur.fetchall()]
        return Remembered(str(new_id), s.text, "created" if inserted else "duplicate", superseded, s.kinds, row["expires_at"])

    # --- read ---

    async def recall(
        self,
        tenant: str,
        namespace: str,
        query: str,
        *,
        k: int = 5,
        kinds: list[Kind] | None = None,
        mode: Mode = "vector",
        recency: bool = True,
        include_history: bool = False,
    ) -> list[Hit]:
        kinds = kinds or ["episodic", "semantic"]
        n = max(self.policy.candidates, k)
        vec = (await self._embed([pii.scrub(query, await self._key(tenant), self.policy.pii).text]))[0]
        params = {
            "tenant": tenant,
            "ns": namespace,
            "kinds": list(kinds),
            "vec": vec,
            "tq": text_query(query),
            "n": n,
            "hist": include_history,
        }
        active = """tenant = %(tenant)s and namespace = %(ns)s and kind = any(%(kinds)s)
                    and (%(hist)s or superseded_by is null) and (expires_at is null or expires_at > now())"""
        vector = f"""select id, row_number() over (order by d) as r from (
                       select id, embedding <=> %(vec)s as d from memories where {active} order by embedding <=> %(vec)s limit %(n)s) x"""
        text = f"""select id, row_number() over (order by rank desc, id) as r, rank from (
                     select id, ts_rank_cd(tsv, q) as rank from memories, to_tsquery('english', %(tq)s) q
                     where {active} and tsv @@ q order by rank desc limit %(n)s) y"""
        empty_v = "select null::uuid as id, null::bigint as r where false"
        empty_t = "select null::uuid as id, null::bigint as r, null::real as rank where false"
        sql = f"""with v as ({vector if mode != "text" else empty_v}), t as ({text if mode != "vector" else empty_t})
                  select m.id, m.kind, m.namespace, m.subject, m.content, m.event_at, m.importance, m.superseded_by,
                         1 - (m.embedding <=> %(vec)s) as sim, v.r as vr, t.r as tr, coalesce(t.rank, 0) as tscore
                  from (select id from v union select id from t) ids
                  join memories m on m.id = ids.id left join v on v.id = ids.id left join t on t.id = ids.id"""
        async with self.db.tenant(tenant) as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
        # Relative score fusion: scale each signal to 0..1 across the candidates, then mix. Unlike rank fusion,
        # the gaps between candidates survive, so recency and importance only settle close calls.
        alpha = {"hybrid": self.policy.vector_weight, "vector": 1.0, "text": 0.0}[mode]
        sims = [float(r["sim"]) for r in rows if r["vr"] is not None] or [0.0]
        lo, hi = min(sims), max(sims)
        top_t = max((float(r["tscore"]) for r in rows), default=0.0) or 1.0
        now = dt.datetime.now(dt.UTC)
        hits = []
        for r in rows:
            v = (float(r["sim"]) - lo) / (hi - lo) if r["vr"] is not None and hi > lo else (1.0 if r["vr"] is not None else 0.0)
            t = float(r["tscore"]) / top_t
            score = alpha * v + (1 - alpha) * t
            score *= 0.9 + 0.2 * float(r["importance"])
            if recency and r["kind"] == "episodic":
                age = (now - r["event_at"]).total_seconds() / 86400
                score *= 0.85 + 0.15 * math.pow(0.5, max(age, 0) / self.policy.recency_half_life_days)
            if r["superseded_by"] is not None:
                score *= 0.5
            hits.append(
                Hit(
                    str(r["id"]),
                    str(r["kind"]),
                    str(r["namespace"]),
                    r["subject"],
                    str(r["content"]),
                    score,
                    float(r["sim"]),
                    r["vr"],
                    r["tr"],
                    r["event_at"],
                    float(r["importance"]),
                    r["superseded_by"] is not None,
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    async def get(self, tenant: str, memory_id: str) -> dict[str, Any] | None:
        async with self.db.tenant(tenant) as cur:
            await cur.execute(
                """select id, namespace, kind, subject, content, importance, pii, source, event_at, created_at, last_seen_at, seen_count,
                          expires_at, superseded_by from memories where id = %s and (expires_at is null or expires_at > now())""",
                (memory_id,),
            )
            return await cur.fetchone()

    async def history(self, tenant: str, namespace: str, subject: str) -> list[dict[str, Any]]:
        async with self.db.tenant(tenant) as cur:
            await cur.execute(
                """select id, content, created_at, superseded_at from memories where namespace = %s and subject = %s
                   and (expires_at is null or expires_at > now()) order by created_at""",
                (namespace, subject),
            )
            return await cur.fetchall()

    # --- erase and expire ---

    async def forget(self, tenant: str, memory_id: str) -> bool:
        async with self.db.tenant(tenant) as cur:
            await cur.execute("delete from memories where id = %s returning namespace", (memory_id,))
            row = await cur.fetchone()
            if row:
                await cur.execute(
                    "insert into erasures (tenant, namespace, memory_id, count, reason) values (%s, %s, %s, 1, 'memory')",
                    (tenant, row["namespace"], memory_id),
                )
        return row is not None

    async def forget_namespace(self, tenant: str, namespace: str) -> int:
        async with self.db.tenant(tenant) as cur:
            await cur.execute("delete from memories where namespace = %s", (namespace,))
            n = cur.rowcount
            await cur.execute(
                "insert into erasures (tenant, namespace, count, reason) values (%s, %s, %s, 'namespace')", (tenant, namespace, n)
            )
        return n

    async def export(self, tenant: str, namespace: str) -> list[dict[str, Any]]:
        async with self.db.tenant(tenant) as cur:
            await cur.execute(
                """select id, kind, subject, content, importance, event_at, created_at, expires_at, superseded_by
                   from memories where namespace = %s and (expires_at is null or expires_at > now()) order by created_at""",
                (namespace,),
            )
            return await cur.fetchall()

    async def sweep(self, batch: int = 1000) -> int:
        total = 0
        while True:
            async with self.db.system() as cur:
                await cur.execute("select sweep_expired(%s) as n", (batch,))
                row = await cur.fetchone()
            n = int(row["n"]) if row else 0
            total += n
            if n < batch:
                return total


def pii_json(kinds: dict[str, Any]) -> Any:
    from psycopg.types.json import Jsonb

    return Jsonb(kinds)
