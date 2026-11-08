# Contributing

You need PostgreSQL 16 or 17 with pgvector. `make db` starts one in Docker and prints the variable to export.

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # for the MiniLM embedder
make install
export MEMSVC_TEST_ADMIN_URL=postgresql://postgres:postgres@localhost:5432/postgres
make check
```

Tests create a throwaway database per session and use a hashing embedder, so they need no model. `pytest -m model` runs the MiniLM test.

## Changing retrieval or the schema

Rerun `make evaluate` and commit `docs/evaluation.md`; say in the pull request what moved. Keep `schema.sql` idempotent and keep row-level security on every table that holds tenant data.

## Changing PII detection

Add the positive and the lookalike cases to `tests/test_pii.py` and to the scrubber benchmark in `evaluation.py`.

## Commits

Small, focused commits with messages in the imperative. Update `CHANGELOG.md` for anything a user would notice.
