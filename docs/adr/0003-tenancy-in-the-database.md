# 3. Tenant isolation is enforced by PostgreSQL row-level security

Status: accepted

## Context

A memory service holds many customers' users' lives. One missing `where tenant = ...` would leak them. Application-level filtering is necessary but not sufficient.

## Decision

- Every API key maps to one tenant; only the key's SHA-256 is stored.
- Each transaction sets `app.tenant`. Row-level security policies on `memories` and `erasures` restrict reads and writes to that tenant, `force row level security` applies them to the table owner too, and every pooled connection drops to the unprivileged `memsvc_app` role, so they apply even if the service logs in as an administrator.
- With no tenant set, nothing is visible. The expiry sweeper uses a `security definer` function that deletes expired rows without reading them.
- Queries still filter by tenant explicitly, for the planner.

## Consequences

`COPY` is not available under row-level security; bulk loads use batched inserts. Tests check isolation with a query that has no tenant filter at all.
