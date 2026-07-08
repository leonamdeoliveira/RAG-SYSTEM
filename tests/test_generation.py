from __future__ import annotations

import pytest

from pipeline.rag.generation.answerer import Answerer, StubLLMClient
from pipeline.rag.generation.prompt_builder import build_prompt, build_sources_block
from pipeline.rag.generation.prompts import VALID_MODES, system_prompt
from pipeline.rag.models import Evidence


def _ev(i: int, text: str = "chunk content", doc_id: str = "d1", score: float = 0.9) -> Evidence:
    return Evidence(
        chunk_id=f"c{i}",
        doc_id=doc_id,
        source_path=f"{doc_id}.md",
        file_name=f"{doc_id}.md",
        score=score,
        chunk_index=i,
        snippet=text,
        h1="Top",
        h2=f"Sec {i}",
    )


def test_system_prompt_contains_grounding_rules():
    s = system_prompt("answer")
    assert "Use ONLY the context" in s
    assert "could not find an answer" in s
    assert "Never invent" in s


def test_system_prompt_invalid_mode_raises():
    with pytest.raises(ValueError):
        system_prompt("bogus")


def test_build_prompt_with_evidence():
    evs = [_ev(1, "Contrato define servicos."), _ev(2, "Pagamento em 30 dias.")]
    system, user = build_prompt("Quais prazos?", evs, mode="answer_with_citations")
    assert "[1]" in user and "[2]" in user
    assert "Contrato define servicos" in user
    assert "QUESTION" in user
    assert "answer_with_citations" in system


def test_build_prompt_empty_evidence():
    system, user = build_prompt("X", [], mode="answer")
    assert "empty" in user.lower() or "no relevant" in user.lower()
    assert "X" in user


def test_build_prompt_truncates_long_snippet():
    long = "word " * 1000
    evs = [_ev(1, long)]
    _, user = build_prompt("q", evs, max_chars_per_chunk=100)
    assert "…" in user or "[…]" in user


def test_build_sources_block():
    evs = [_ev(1, "a"), _ev(2, "b", doc_id="d2")]
    s = build_sources_block(evs)
    assert "Sources:" in s
    assert "[1]" in s and "[2]" in s
    assert "d1.md" in s and "d2.md" in s


def test_valid_modes_set():
    assert set(VALID_MODES) == {"answer", "answer_with_citations", "extractive_summary", "study_mode"}


def test_answerer_stub_no_evidence_marks_insufficient():
    a = Answerer(llm=StubLLMClient(), default_mode="answer")
    ans = a.answer("Qualquer coisa", evidence=[])
    assert ans.insufficient_context
    assert ans.confidence == 0.0


def test_answerer_stub_with_evidence():
    evs = [_ev(1, "Resposta clara no chunk.", score=0.9)]
    a = Answerer(llm=StubLLMClient(), default_mode="answer")
    ans = a.answer("O que diz?", evidence=evs)
    assert ans.text.startswith("[STUB LLM]")
    assert ans.confidence == 0.9


def test_answerer_mode_passed_through():
    evs = [_ev(1, "x")]
    a = Answerer(llm=StubLLMClient(), default_mode="answer")
    ans = a.answer("q", evidence=evs, mode="study_mode")
    assert ans.mode == "study_mode"


def test_answerer_with_sources_appends_block():
    evs = [_ev(1, "x")]
    a = Answerer(llm=StubLLMClient())
    out = a.answer_with_sources("q", evs)
    assert "Sources:" in out


def test_answerer_llm_error_fallback_text():
    class Boom:
        def complete(self, system, user, temperature=0.2, max_tokens=700):
            raise RuntimeError("no model")

    a = Answerer(llm=Boom())
    ans = a.answer("q", evidence=[_ev(1, "x", score=0.5)])
    assert "could not find an answer" in ans.text.lower() or "LLM error" in ans.text