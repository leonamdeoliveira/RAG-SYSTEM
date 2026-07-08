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


def test_cli_help():
    r = _run(["--help"])
    assert r.returncode == 0, r.stderr
    assert "ingest" in r.stdout.lower()
    assert "retrieve" in r.stdout.lower()
    assert "query" in r.stdout.lower()


def test_cli_ingest_help():
    r = _run(["ingest", "--help"])
    assert r.returncode == 0, r.stderr
    assert "--rag-only" in r.stdout
    assert "--ocr-only" in r.stdout


def test_cli_retrieve_help():
    r = _run(["retrieve", "--help"])
    assert r.returncode == 0, r.stderr
    assert "--retrieval-mode" in r.stdout
    assert "--top-k" in r.stdout


def test_cli_reindex_help():
    r = _run(["reindex", "--help"])
    assert r.returncode == 0, r.stderr
    assert "--purge" in r.stdout
