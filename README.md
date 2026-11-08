# agent-memory-service

[![CI](https://github.com/darshansatpathy/agent-memory-service/actions/workflows/ci.yml/badge.svg)](https://github.com/darshansatpathy/agent-memory-service/actions/workflows/ci.yml)
[![CodeQL](https://github.com/darshansatpathy/agent-memory-service/actions/workflows/codeql.yml/badge.svg)](https://github.com/darshansatpathy/agent-memory-service/actions/workflows/codeql.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%7C%2017%20%2B%20pgvector-336791)
![License](https://img.shields.io/badge/license-MIT-green)

Long-term memory for agents on PostgreSQL and pgvector: **episodes that fade, facts that change, personal data that never gets stored, and tenants that cannot see each other**.

Most agent memory is an append-only pile of text chunks ranked by similarity. That breaks in predictable ways: the user moved, and the agent still "remembers" the old city; a phone number said once is now in every backup and embedding; a missing `where` clause shows one customer's users to another. This service is built around those failures.

From [`docs/evaluation.md`](docs/evaluation.md): 30 simulated users, 481 memories, 420 questions each worded differently from the memory it should find, MiniLM embeddings, PostgreSQL 17 with pgvector.

| | |
|---|---|
| Right memory first (vector recall) | **96.9%**; in the top five, 100% |
| A fact that changed: outdated value returned first | **0%** with supersession; **29%** if memories are append-only |
| Email addresses and phone numbers written into memories that reached the database | **0 of 60** (stored as keyed pseudonyms) |
| Memories carrying a card number | 30 of 30 refused |
| Results from another user's memories | 0 |
| Recall at 100,000 memories across 1,000 namespaces | p95 **5.2 ms** (database time) |

The benchmark also caught two bugs before release: erasing memories was quadratic (a self-referencing foreign key without an index; 100,000 rows went from over 8 minutes to 0.8 s), and an early rank-fusion design let recency weights bury every past conversation.

## The model

| | Episodic | Semantic |
|---|---|---|
| What | Something that happened: "planned a trip to Peru in May" | Something true about someone: "lives in Porto" |
| Lifetime | Default TTL of 90 days; weighted by recency at recall | No default TTL |
| Change | Each is its own memory | A memory with a `subject` (`user.city`) supersedes the previous value for that subject, in the same transaction; the old one stays as history |
| Repeats | Exact repeats are counted, not duplicated | Near-duplicates (cosine ≥ 0.92) merge |

Recall is vector search (HNSW) over the caller's namespace by default; `mode: "hybrid"` adds full-text search, fused by relative score, for traffic full of exact tokens. Expired and superseded memories never come back unless history is asked for. Each hit says why it was returned: vector rank, text rank, similarity.

## Personal data

Everything is scrubbed **before** it is embedded or stored, on writes and on recall queries:

| Kind | Validated by | Default action |
|---|---|---|
| API keys, tokens, passwords | known prefixes, `password: ...` patterns | **refuse the memory** (422, kinds only) |
| Card numbers | Luhn and issuer prefix | refuse |
| IBANs | mod-97 checksum | refuse |
| US SSNs | area and group rules | refuse |
| Email addresses, phone numbers | shape, digit count | **pseudonymise**: `[EMAIL:3f9a1c2e]`, keyed per tenant, so two mentions still match |
| IP addresses | valid, not loopback | redact |

What was found is stored as kinds and counts, never values.

## Tenancy

API keys map to tenants (only SHA-256 hashes are stored). Every transaction sets `app.tenant`, and **PostgreSQL row-level security** enforces it: `force row level security`, and every pooled connection drops to an unprivileged role. A query with no tenant filter at all still returns only the caller's rows; with no tenant set, nothing is visible. The expiry sweeper deletes through a `security definer` function without reading content.

## API

```bash
# remember a fact that may change later
curl -s localhost:8080/v1/memories -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d '{"namespace": "user-42", "kind": "semantic", "subject": "user.city", "content": "The user moved to Porto."}'

# recall
curl -s localhost:8080/v1/recall -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d '{"namespace": "user-42", "query": "where is she based now?", "k": 5}'
```

| Endpoint | |
|---|---|
| `POST /v1/memories` | write (201 created, 200 duplicate or merged, 422 refused for personal data) |
| `POST /v1/recall` | `mode`: vector (default), hybrid or text; `kinds`; `include_history`; `recency` |
| `GET /v1/memories/{id}`, `GET /v1/namespaces/{ns}/subjects/{subject}/history` | read one, see how a fact changed |
| `DELETE /v1/memories/{id}`, `DELETE /v1/namespaces/{ns}` | erase; leaves a record that an erasure happened, not what was erased |
| `GET /v1/namespaces/{ns}/export` | everything held about a namespace, for access requests |

## Running it

```bash
docker compose up -d                                   # pgvector, migration, service on :8080
docker compose run --rm memsvc create-tenant acme
KEY=$(docker compose run --rm memsvc create-key acme)
```

The image bakes in the embedding model (all-MiniLM-L6-v2), so it starts without network access. Settings are `MEMSVC_*` environment variables ([`config.py`](src/memsvc/config.py)).

## Design decisions

[ADR 1](docs/adr/0001-two-kinds-of-memory.md) two kinds of memory · [ADR 2](docs/adr/0002-scrub-before-storage.md) scrub before storage · [ADR 3](docs/adr/0003-tenancy-in-the-database.md) tenancy in the database · [ADR 4](docs/adr/0004-hybrid-recall.md) hybrid recall

## License

MIT
