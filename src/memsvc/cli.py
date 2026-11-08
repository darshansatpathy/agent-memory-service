"""`memsvc`: migrate, create tenants and keys, sweep expired memories, serve, evaluate."""

from __future__ import annotations

import argparse
import asyncio
import sys

from memsvc import __version__
from memsvc.config import Settings


def _service(s: Settings):  # type: ignore[no-untyped-def]
    from memsvc.db import Database
    from memsvc.embed import HashEmbedder
    from memsvc.service import MemoryService

    return MemoryService(Database(s.database_url, min_size=1, max_size=2), HashEmbedder())


async def _admin(args: argparse.Namespace) -> int:
    s = Settings()
    svc = _service(s)
    await svc.db.open()
    try:
        if args.cmd == "create-tenant":
            await svc.create_tenant(args.tenant)
            print(f"tenant {args.tenant} ready")
        elif args.cmd == "create-key":
            print(await svc.create_key(args.tenant, args.label))
        elif args.cmd == "sweep":
            print(f"deleted {await svc.sweep(args.batch)} expired memories")
    finally:
        await svc.db.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="memsvc", description=__doc__)
    p.add_argument("--version", action="version", version=f"memsvc {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate", help="create or update the schema (run as the database owner)")
    s = sub.add_parser("create-tenant")
    s.add_argument("tenant")
    s = sub.add_parser("create-key", help="print a new API key for a tenant (shown once; only its hash is stored)")
    s.add_argument("tenant")
    s.add_argument("--label", default="")
    s = sub.add_parser("sweep", help="delete expired memories (for a CronJob; the server also sweeps)")
    s.add_argument("--batch", type=int, default=1000)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    args = p.parse_args(argv)

    if args.cmd == "migrate":
        from memsvc.db import migrate

        migrate(Settings().database_url)
        print("schema up to date")
        return 0
    if args.cmd == "serve":
        import uvicorn

        from memsvc.api import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning", server_header=False)
        return 0
    return asyncio.run(_admin(args))


if __name__ == "__main__":
    sys.exit(main())
