"""Tests run against a real PostgreSQL with pgvector: set MEMSVC_TEST_ADMIN_URL (CI uses the pgvector image).
Each session gets a fresh database, dropped afterwards."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Iterator

import psycopg
import pytest

from memsvc import db as dbm
from memsvc.embed import HashEmbedder
from memsvc.service import MemoryService, Policy

ADMIN = os.environ.get("MEMSVC_TEST_ADMIN_URL", "postgresql://postgres@localhost:5432/postgres")


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    name = f"memsvc_test_{uuid.uuid4().hex[:8]}"
    try:
        with psycopg.connect(ADMIN, autocommit=True) as c:
            c.execute(f"create database {name}")
    except psycopg.OperationalError as e:
        pytest.skip(f"no PostgreSQL at MEMSVC_TEST_ADMIN_URL: {e}")
    url = psycopg.conninfo.make_conninfo(ADMIN, dbname=name)
    dbm.migrate(url)
    yield url
    with psycopg.connect(ADMIN, autocommit=True) as c:
        c.execute(f"drop database {name} with (force)")


@pytest.fixture
async def svc(database_url: str) -> AsyncIterator[MemoryService]:
    d = dbm.Database(database_url, min_size=1, max_size=4)
    await d.open()
    s = MemoryService(d, HashEmbedder(), Policy())
    for t in ("acme", "globex"):
        await s.create_tenant(t)
    yield s
    for t in ("acme", "globex"):
        async with d.tenant(t) as cur:
            await cur.execute("delete from memories")
    await d.close()
