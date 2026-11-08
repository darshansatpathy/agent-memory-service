## What changes

## Why

## How it was tested

- [ ] `make check` passes against PostgreSQL with pgvector
- [ ] Schema change: `schema.sql` stays idempotent, row-level security still covers every table with tenant data
- [ ] PII detector change: `tests/test_pii.py` and the scrubber benchmark updated
- [ ] Retrieval change: `bench/evaluate.py` rerun and `docs/evaluation.md` committed
- [ ] `CHANGELOG.md` updated
