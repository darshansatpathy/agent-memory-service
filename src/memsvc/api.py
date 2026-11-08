"""HTTP API. Every request carries an API key; the key decides the tenant."""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import logging
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from memsvc import __version__, metrics
from memsvc.config import Settings
from memsvc.db import Database
from memsvc.embed import Embedder, get
from memsvc.service import Blocked, MemoryService, Policy

log = logging.getLogger("memsvc")
Namespace = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:@-]+$")]
Subject = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[a-z0-9_.-]+$", description="A key such as user.city")]


class MemoryIn(BaseModel):
    namespace: Namespace
    kind: Literal["episodic", "semantic"]
    content: Annotated[str, Field(min_length=1, max_length=4000)]
    subject: Subject | None = Field(None, description="Semantic only. A newer memory with the same subject supersedes the older one.")
    importance: Annotated[float, Field(ge=0, le=1)] = 0.5
    ttl_days: Annotated[float | None, Field(gt=0, le=3650)] = None
    no_expiry: bool = False
    event_at: dt.datetime | None = None
    source: dict[str, str | int] = Field(default_factory=dict, description="Provenance: session, turn, tool")


class MemoryOut(BaseModel):
    id: str
    status: Literal["created", "duplicate", "merged"]
    content: str
    superseded: list[str]
    pii: dict[str, int]
    expires_at: dt.datetime | None


class RecallIn(BaseModel):
    namespace: Namespace
    query: Annotated[str, Field(min_length=1, max_length=1000)]
    k: Annotated[int, Field(ge=1, le=50)] = 5
    kinds: list[Literal["episodic", "semantic"]] | None = None
    mode: Literal["hybrid", "vector", "text"] = Field(
        "vector", description="vector is best on the benchmark; hybrid adds exact-word matching"
    )
    recency: bool = True
    include_history: bool = False


class HitOut(BaseModel):
    id: str
    kind: str
    subject: str | None
    content: str
    score: float
    similarity: float
    vector_rank: int | None
    text_rank: int | None
    event_at: dt.datetime
    superseded: bool


def create_app(settings: Settings | None = None, *, embedder: Embedder | None = None, sweep: bool = True) -> FastAPI:
    s = settings or Settings()

    def days(v: float | None) -> dt.timedelta | None:
        return dt.timedelta(days=v) if v else None

    policy = Policy(
        episodic_ttl=days(s.episodic_ttl_days),
        semantic_ttl=days(s.semantic_ttl_days),
        merge_similarity=s.merge_similarity,
        recency_half_life_days=s.recency_half_life_days,
    )
    db = Database(s.database_url, max_size=s.pool_max)
    svc = MemoryService(db, embedder or get(s.embedder), policy)

    async def sweeper() -> None:
        while True:
            await asyncio.sleep(s.sweep_interval_seconds)
            try:
                n = await svc.sweep()
                metrics.EXPIRED.inc(n)
            except Exception:
                log.exception("sweep failed")

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await db.open()
        task = asyncio.create_task(sweeper()) if sweep else None
        yield
        if task:
            task.cancel()
        await db.close()

    app = FastAPI(title="agent-memory-service", version=__version__, lifespan=lifespan)
    app.state.service = svc

    async def tenant(authorization: Annotated[str | None, Header()] = None) -> str:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "missing API key", headers={"WWW-Authenticate": "Bearer"})
        t = await svc.authenticate(authorization[7:].strip())
        if t is None:
            raise HTTPException(401, "invalid API key", headers={"WWW-Authenticate": "Bearer"})
        return t

    # Declared as a default, not an Annotated alias: annotations are strings here and a local alias would not resolve.
    TENANT = Depends(tenant)

    @app.post("/v1/memories", response_model=MemoryOut, status_code=201)
    async def remember(body: MemoryIn, response: Response, t: str = TENANT) -> MemoryOut:
        if body.kind == "episodic" and body.subject:
            raise HTTPException(422, "subject is for semantic memories")
        ttl: Any = None if body.no_expiry else (dt.timedelta(days=body.ttl_days) if body.ttl_days else "default")
        started = time.perf_counter()
        try:
            r = await svc.remember(
                t,
                body.namespace,
                body.kind,
                body.content,
                subject=body.subject,
                importance=body.importance,
                ttl=ttl,
                event_at=body.event_at,
                source=dict(body.source),
            )
        except Blocked as e:
            for k in e.kinds:
                metrics.BLOCKED.labels(k).inc()
            raise HTTPException(422, {"error": "personal_data", "kinds": e.kinds, "message": str(e)}) from None
        finally:
            metrics.LATENCY.labels("remember").observe(time.perf_counter() - started)
        metrics.WRITES.labels(body.kind, r.status).inc()
        for k, n in r.pii.items():
            metrics.SCRUBBED.labels(k).inc(n)
        if r.status != "created":
            response.status_code = 200
        return MemoryOut(id=r.id, status=r.status, content=r.content, superseded=r.superseded, pii=r.pii, expires_at=r.expires_at)

    @app.post("/v1/recall", response_model=list[HitOut])
    async def recall(body: RecallIn, t: str = TENANT) -> list[HitOut]:
        started = time.perf_counter()
        hits = await svc.recall(
            t,
            body.namespace,
            body.query,
            k=body.k,
            kinds=body.kinds,
            mode=body.mode,
            recency=body.recency,
            include_history=body.include_history,
        )
        metrics.LATENCY.labels("recall").observe(time.perf_counter() - started)
        metrics.RECALLS.labels(body.mode).inc()
        return [
            HitOut(
                id=h.id,
                kind=h.kind,
                subject=h.subject,
                content=h.content,
                score=h.score,
                similarity=h.similarity,
                vector_rank=h.vector_rank,
                text_rank=h.text_rank,
                event_at=h.event_at,
                superseded=h.superseded,
            )
            for h in hits
        ]

    @app.get("/v1/memories/{memory_id}")
    async def get_memory(memory_id: Annotated[str, Path(pattern=r"^[0-9a-f-]{36}$")], t: str = TENANT) -> dict[str, Any]:
        m = await svc.get(t, memory_id)
        if m is None:
            raise HTTPException(404, "not found")
        return {k: (str(v) if k in ("id", "superseded_by") and v is not None else v) for k, v in m.items()}

    @app.get("/v1/namespaces/{namespace}/subjects/{subject}/history")
    async def history(namespace: str, subject: str, t: str = TENANT) -> list[dict[str, Any]]:
        return [{**r, "id": str(r["id"])} for r in await svc.history(t, namespace, subject)]

    @app.delete("/v1/memories/{memory_id}", status_code=204)
    async def forget(memory_id: Annotated[str, Path(pattern=r"^[0-9a-f-]{36}$")], t: str = TENANT) -> Response:
        if not await svc.forget(t, memory_id):
            raise HTTPException(404, "not found")
        metrics.ERASED.labels("memory").inc()
        return Response(status_code=204)

    @app.delete("/v1/namespaces/{namespace}")
    async def forget_namespace(namespace: str, t: str = TENANT) -> dict[str, int]:
        n = await svc.forget_namespace(t, namespace)
        metrics.ERASED.labels("namespace").inc(n)
        return {"erased": n}

    @app.get("/v1/namespaces/{namespace}/export")
    async def export(namespace: str, t: str = TENANT) -> list[dict[str, Any]]:
        return [
            {k: (str(v) if k in ("id", "superseded_by") and v is not None else v) for k, v in r.items()}
            for r in await svc.export(t, namespace)
        ]

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz() -> Response:
        try:
            async with db.system() as cur:
                await cur.execute("select 1")
            return Response("ready")
        except Exception:
            return Response("database unavailable", status_code=503)

    @app.get("/metrics", include_in_schema=False)
    async def prom() -> Response:
        return Response(generate_latest(metrics.REGISTRY), media_type=CONTENT_TYPE_LATEST)

    return app
