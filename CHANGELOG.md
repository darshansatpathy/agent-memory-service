# Changelog

## 0.1.0

- Episodic and semantic memories on PostgreSQL with pgvector; HNSW and full-text indexes.
- Semantic subjects: a new value supersedes the old in one transaction, with history kept. Near-duplicate facts merge; exact repeats are counted, not duplicated.
- TTLs per kind and per memory, enforced at read time and by a sweeper.
- PII scrubbing before embedding: block, pseudonymise with a per-tenant key, or redact, with validators against lookalikes.
- Row-level security per tenant, enforced for an unprivileged role on every connection; API keys stored as hashes.
- Hybrid recall fused by reciprocal rank, weighted by importance and recency, with per-hit signals.
- Erasure of a memory or a whole namespace with a content-free record, and export.
- FastAPI service, CLI, container with the model baked in, Compose stack, and a benchmark with a CI gate.
