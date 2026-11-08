from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from memsvc.api import create_app
from memsvc.config import Settings
from memsvc.embed import HashEmbedder


@pytest.fixture
def client(database_url):
    app = create_app(Settings(database_url=database_url, embedder="hash"), embedder=HashEmbedder(), sweep=False)
    with TestClient(app) as c:
        svc = app.state.service
        c.portal.call(svc.create_tenant, "acme")
        c.portal.call(svc.create_tenant, "globex")
        c.headers["Authorization"] = "Bearer " + c.portal.call(svc.create_key, "acme", "test")
        c.globex = "Bearer " + c.portal.call(svc.create_key, "globex", "test")  # type: ignore[attr-defined]
        yield c
        c.delete("/v1/namespaces/api-u1")


def test_auth_required(client):
    assert client.post("/v1/recall", json={"namespace": "x", "query": "q"}, headers={"Authorization": ""}).status_code == 401
    assert client.post("/v1/recall", json={"namespace": "x", "query": "q"}, headers={"Authorization": "Bearer mk_wrong"}).status_code == 401


def test_remember_recall_supersede(client):
    r = client.post(
        "/v1/memories", json={"namespace": "api-u1", "kind": "semantic", "content": "The user lives in Lisbon", "subject": "user.city"}
    )
    assert r.status_code == 201
    r2 = client.post(
        "/v1/memories", json={"namespace": "api-u1", "kind": "semantic", "content": "The user lives in Porto", "subject": "user.city"}
    )
    assert r2.json()["superseded"] == [r.json()["id"]]
    again = client.post(
        "/v1/memories", json={"namespace": "api-u1", "kind": "semantic", "content": "The user lives in Porto", "subject": "user.city"}
    )
    assert again.status_code == 200 and again.json()["status"] == "duplicate"
    hits = client.post("/v1/recall", json={"namespace": "api-u1", "query": "which city does the user live in"}).json()
    assert hits[0]["content"] == "The user lives in Porto" and hits[0]["subject"] == "user.city"
    hist = client.get("/v1/namespaces/api-u1/subjects/user.city/history").json()
    assert [h["content"] for h in hist] == ["The user lives in Lisbon", "The user lives in Porto"]
    assert client.get(f"/v1/memories/{r2.json()['id']}").json()["content"] == "The user lives in Porto"


def test_pii_refusal_is_422_with_kinds_only(client):
    r = client.post("/v1/memories", json={"namespace": "api-u1", "kind": "episodic", "content": "card 4111 1111 1111 1111"})
    assert r.status_code == 422 and r.json()["detail"]["kinds"] == ["CARD"]
    assert "4111" not in r.text


def test_validation(client):
    assert client.post("/v1/memories", json={"namespace": "bad ns!", "kind": "semantic", "content": "x"}).status_code == 422
    assert (
        client.post("/v1/memories", json={"namespace": "u", "kind": "episodic", "content": "x", "subject": "user.city"}).status_code == 422
    )
    assert client.post("/v1/memories", json={"namespace": "u", "kind": "semantic", "content": "x" * 4001}).status_code == 422


def test_other_tenant_sees_nothing(client):
    r = client.post("/v1/memories", json={"namespace": "api-u1", "kind": "semantic", "content": "Acme private fact"})
    mid = r.json()["id"]
    other = {"Authorization": client.globex}
    assert client.get(f"/v1/memories/{mid}", headers=other).status_code == 404
    assert client.delete(f"/v1/memories/{mid}", headers=other).status_code == 404
    assert client.post("/v1/recall", json={"namespace": "api-u1", "query": "private fact"}, headers=other).json() == []


def test_erase_export_and_ops(client):
    client.post("/v1/memories", json={"namespace": "api-u1", "kind": "episodic", "content": "Talked about gardening", "ttl_days": 7})
    client.post("/v1/memories", json={"namespace": "api-u1", "kind": "semantic", "content": "Likes tulips", "no_expiry": True})
    exported = client.get("/v1/namespaces/api-u1/export").json()
    assert len(exported) == 2 and "embedding" not in exported[0]
    assert client.delete("/v1/namespaces/api-u1").json() == {"erased": 2}
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").status_code == 200
    assert "memory_writes_total" in client.get("/metrics").text
