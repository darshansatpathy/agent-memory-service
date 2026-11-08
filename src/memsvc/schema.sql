-- Schema for agent memory. Applied by `memsvc migrate`; idempotent.
create extension if not exists vector;
create extension if not exists pgcrypto;

create table if not exists tenants (
  id text primary key,
  pseudonym_key bytea not null default gen_random_bytes(32),
  created_at timestamptz not null default now()
);

create table if not exists api_keys (
  key_hash text primary key,              -- sha256 of the key; the key itself is shown once
  tenant text not null references tenants(id) on delete cascade,
  label text not null default '',
  created_at timestamptz not null default now(),
  revoked_at timestamptz
);

create table if not exists memories (
  id uuid primary key default gen_random_uuid(),
  tenant text not null references tenants(id) on delete cascade,
  namespace text not null,                -- whose memory: a user, an agent, a team
  kind text not null check (kind in ('episodic', 'semantic')),
  subject text,                           -- semantic only: a key such as user.city; a newer value supersedes the old
  content text not null,                  -- after PII scrubbing
  content_hash text not null,
  embedding vector(384) not null,
  tsv tsvector generated always as (to_tsvector('english', content)) stored,
  importance real not null default 0.5 check (importance between 0 and 1),
  pii jsonb not null default '{}',        -- kinds and counts found, never values
  source jsonb not null default '{}',     -- session, turn, tool: provenance
  event_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  last_seen_at timestamptz not null default now(),
  seen_count int not null default 1,
  expires_at timestamptz,
  superseded_by uuid references memories(id) on delete set null,
  superseded_at timestamptz
);

create index if not exists memories_embedding on memories using hnsw (embedding vector_cosine_ops) with (m = 16, ef_construction = 64);
create index if not exists memories_tsv on memories using gin (tsv);
create index if not exists memories_scope on memories (tenant, namespace, kind) where superseded_by is null;
create index if not exists memories_subject on memories (tenant, namespace, subject) where subject is not null and superseded_by is null;
create unique index if not exists memories_dedupe on memories (tenant, namespace, kind, content_hash) where superseded_by is null;
create index if not exists memories_expiry on memories (expires_at) where expires_at is not null;
-- Deleting a memory checks for rows it superseded; without this index every delete scans the table.
create index if not exists memories_superseded_by on memories (superseded_by) where superseded_by is not null;

-- Erasure leaves a record that it happened, never what was erased.
create table if not exists erasures (
  id bigserial primary key,
  tenant text not null,
  namespace text not null,
  memory_id uuid,
  count int not null,
  reason text not null,
  at timestamptz not null default now()
);

-- Row-level security: the application role sees only the tenant it set for the transaction.
alter table memories enable row level security;
alter table memories force row level security;
drop policy if exists tenant_isolation on memories;
create policy tenant_isolation on memories
  using (tenant = current_setting('app.tenant', true))
  with check (tenant = current_setting('app.tenant', true));

alter table erasures enable row level security;
alter table erasures force row level security;
drop policy if exists tenant_isolation on erasures;
create policy tenant_isolation on erasures
  using (tenant = current_setting('app.tenant', true))
  with check (tenant = current_setting('app.tenant', true));

-- The sweeper deletes expired rows across tenants without seeing their content.
create or replace function sweep_expired(batch int) returns int
language sql security definer set search_path = public as $$
  with gone as (
    delete from memories where id in (
      select id from memories where expires_at < now() order by expires_at limit batch for update skip locked
    ) returning 1
  ) select count(*)::int from gone;
$$;

do $$ begin
  if not exists (select 1 from pg_roles where rolname = 'memsvc_app') then
    create role memsvc_app nologin;
  end if;
end $$;
grant select, insert, update, delete on memories, erasures to memsvc_app;
grant usage on sequence erasures_id_seq to memsvc_app;
grant select on tenants, api_keys to memsvc_app;
grant execute on function sweep_expired(int) to memsvc_app;
