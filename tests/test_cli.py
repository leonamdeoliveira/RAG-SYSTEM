from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.zvec

MAIN_PY = str(Path(__file__).resolve().parent.parent / "main.py")
SKILL_DIR = str(Path(__file__).resolve().parent.parent)


def _run(args: list[str]) -> subprocess.CompletedProcess:
    env = {**__import__("os").environ, "PYTHONPATH": SKILL_DIR}
    return subprocess.run(
        [sys.executable, MAIN_PY, *args],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=SKILL_DIR,
        env=env,
    )


def test_cli_ingest_dummy(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "doc.md").write_text("# Manual\nContrato de servicos com pagamento em 30 dias.\n", encoding="utf-8")

    r = _run(["ingest", "--rag-only", "--markdown-dir", str(data),
              "--index-dir", str(tmp_path / "idx"),
              "--provider", "dummy", "--dimension", "64"])
    assert r.returncode == 0, r.stderr
    out = (r.stdout + r.stderr).lower()
    assert "chunks" in out or "scanned" in out


def test_cli_query_dummy_stub(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "doc.md").write_text("# Manual\nContrato de servicos com pagamento em 30 dias.\n", encoding="utf-8")

    r1 = _run(["ingest", "--rag-only", "--markdown-dir", str(data),
               "--index-dir", str(tmp_path / "idx"),
               "--provider", "dummy", "--dimension", "64"])
    assert r1.returncode == 0, r1.stderr

    r2 = _run(["query", "Qual o prazo?",
               "--markdown-dir", str(data), "--index-dir", str(tmp_path / "idx"),
               "--provider", "dummy", "--dimension", "64", "--llm-model", "stub"])
    assert r2.returncode == 0, r2.stderr
    out = (r2.stdout + r2.stderr).lower()
    # Stub LLM retorna mensagem de erro ou resposta com fontes
    assert "fontes" in out or "sources" in out or "evidence" in out or "stub" in out or "contexto" in out


def test_cli_reindex_purge(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.md").write_text("# A\nConteudo A.\n", encoding="utf-8")

    r1 = _run(["ingest", "--rag-only", "--markdown-dir", str(data),
               "--index-dir", str(tmp_path / "idx"),
               "--provider", "dummy", "--dimension", "64"])
    assert r1.returncode == 0

    r2 = _run(["reindex", "--purge", "--markdown-dir", str(data),
               "--index-dir", str(tmp_path / "idx"),
               "--provider", "dummy", "--dimension", "64"])
    assert r2.returncode == 0, r2.stderr
