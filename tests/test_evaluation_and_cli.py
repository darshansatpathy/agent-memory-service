from __future__ import annotations

import pytest

from memsvc import evaluation
from memsvc.cli import main


async def test_benchmark_end_to_end_with_the_hashing_embedder(svc):
    data = evaluation.generate(users=3, seed=1)
    await svc.create_tenant("bench")
    await svc.create_tenant("bench-ablation")
    await svc.create_tenant("bench-scale")
    m = await evaluation.run(svc, "bench", data, ["vector", "text", "hybrid"])
    a = await evaluation.run(svc, "bench-ablation", data, ["hybrid"], append_only=True)
    s = await evaluation.scale(svc, "bench-scale", 600, 6, queries=20)
    assert m["users"] == 3 and m["blocked"] == m["attempted_blocked"] == 3
    assert m["pii_values_stored"] == 0 and all(x["leaks"] == 0 for x in m["modes"].values())
    assert m["modes"]["hybrid"]["changed_facts"]["stale_first"] == 0
    assert a["modes"]["hybrid"]["changed_facts"]["n"] == m["modes"]["hybrid"]["changed_facts"]["n"] > 0
    text = evaluation.report(m, a, evaluation.pii_benchmark(n=70), s, "hash-384")
    assert "## Recall by retrieval mode" in text and "| hybrid |" in text and "600 memories" in text
    for t in ("bench", "bench-ablation", "bench-scale"):
        async with svc.db.tenant(t) as cur:
            await cur.execute("delete from memories")


def test_cli_admin_commands(database_url, monkeypatch, capsys):
    monkeypatch.setenv("MEMSVC_DATABASE_URL", database_url)
    assert main(["migrate"]) == 0
    assert main(["create-tenant", "cli-tenant"]) == 0
    capsys.readouterr()
    assert main(["create-key", "cli-tenant", "--label", "ci"]) == 0
    assert capsys.readouterr().out.strip().startswith("mk_")
    assert main(["sweep"]) == 0
    assert "expired memories" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--version"])


@pytest.mark.model
def test_minilm_understands_paraphrase():
    from memsvc.embed import MiniLM

    e = MiniLM().embed(["The user relocated to Porto last spring.", "Which city does the user live in?", "I like strawberry jam."])
    assert float(e[0] @ e[1]) > float(e[2] @ e[1]) + 0.1
