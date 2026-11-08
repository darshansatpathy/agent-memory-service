from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

REGISTRY = CollectorRegistry()
WRITES = Counter("memory_writes_total", "Memories written, by kind and status", ["kind", "status"], registry=REGISTRY)
BLOCKED = Counter("memory_blocked_total", "Memories refused for personal data, by kind of data", ["pii"], registry=REGISTRY)
SCRUBBED = Counter("memory_pii_scrubbed_total", "Personal data replaced before storage, by kind", ["pii"], registry=REGISTRY)
RECALLS = Counter("memory_recalls_total", "Recall requests", ["mode"], registry=REGISTRY)
EXPIRED = Counter("memory_expired_total", "Memories deleted by the sweeper", registry=REGISTRY)
ERASED = Counter("memory_erased_total", "Memories erased on request", ["scope"], registry=REGISTRY)
LATENCY = Histogram(
    "memory_request_seconds", "Request latency", ["op"], buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5), registry=REGISTRY
)
