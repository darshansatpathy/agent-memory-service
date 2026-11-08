.PHONY: install check test db evaluate serve

# A throwaway PostgreSQL with pgvector for tests: make db, then export MEMSVC_TEST_ADMIN_URL as printed.
db:
	docker run -d --name memsvc-pg -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg17
	@echo 'export MEMSVC_TEST_ADMIN_URL=postgresql://postgres:postgres@localhost:5432/postgres'

install:
	pip install -e ".[embed,dev]"

check:
	ruff check .
	ruff format --check .
	mypy
	pytest --cov --cov-report=term-missing

evaluate:
	python bench/evaluate.py --users 30 --scale 100000 --out docs/evaluation.md

serve:
	memsvc serve
