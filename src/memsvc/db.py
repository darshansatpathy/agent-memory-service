"""Connection pool, migrations and per-tenant transactions.

Every connection drops to the unprivileged `memsvc_app` role, so row-level security applies even if the
service logs in as an administrator. Every transaction sets `app.tenant`; the policies do the rest.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import resources
from typing import Any

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


def schema_sql() -> str:
    return resources.files("memsvc").joinpath("schema.sql").read_text()


def migrate(url: str) -> None:
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(schema_sql())


async def _configure(conn: psycopg.AsyncConnection) -> None:
    await conn.set_autocommit(True)
    await conn.execute("set role memsvc_app")
    await register_vector_async(conn)
    await conn.set_autocommit(False)


class Database:
    def __init__(self, url: str, min_size: int = 2, max_size: int = 10):
        self.pool = AsyncConnectionPool(
            url,
            min_size=min_size,
            max_size=max_size,
            configure=_configure,
            open=False,
            kwargs={"row_factory": dict_row, "options": "-c timezone=UTC"},
        )

    async def open(self) -> None:
        await self.pool.open(wait=True, timeout=30)

    async def close(self) -> None:
        await self.pool.close()

    @asynccontextmanager
    async def tenant(self, tenant: str, *, ef_search: int = 100) -> AsyncIterator[psycopg.AsyncCursor[dict[str, Any]]]:
        """A transaction scoped to one tenant. HNSW iterative scans keep filtered vector search from returning too few rows."""
        async with self.pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("select set_config('app.tenant', %s, true)", (tenant,))
            await cur.execute(
                "select set_config('hnsw.iterative_scan', 'relaxed_order', true), set_config('hnsw.ef_search', %s, true)", (str(ef_search),)
            )
            yield cur

    @asynccontextmanager
    async def system(self) -> AsyncIterator[psycopg.AsyncCursor[dict[str, Any]]]:
        """No tenant set: memories are invisible. For key lookup and the sweeper."""
        async with self.pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
            yield cur
