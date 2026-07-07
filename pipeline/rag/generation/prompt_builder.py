"""Montagem do prompt final (system + contexto + pergunta).

Cada chunk recuperado vira um bloco numerado [n] com:
  - file_name
  - hierarquia h1 / h2 / h3
  - chunk_id (rastreabilidade)
  - texto (snippet)
"""

from __future__ import annotations

from typing import Sequence

from pipeline.rag.generation.prompts import VALID_MODES, context_header, system_prompt
from pipeline.rag.models import Evidence


def _format_chunk(idx: int, e: Evidence, max_chars: int = 1200) -> str:
    path_parts = [p for p in (e.h1, e.h2, e.h3) if p]
    path = " / ".join(path_parts) if path_parts else "(sem secao)"
    snippet = e.snippet
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0] + " […]"
    return (
        f"[{idx}] FILE: {e.file_name} | SECTION: {path} | CHUNK: {e.chunk_id}\n"
        f"{snippet}"
    )


def build_prompt(
    question: str,
    evidence: Sequence[Evidence],
    mode: str = "answer",
    max_chars_per_chunk: int = 1200,
) -> tuple[str, str]:
    """Retorna (system, user) prontos para o LLM.

    `system` ja carrega as instrucoes de grounding do modo.
    `user` contem o contexto numerado + a pergunta.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"modo invalido: {mode}. Validos: {VALID_MODES}")
    if not evidence:
        # Sem contexto: ainda assim o system prompt instrui o LLM a dizer que
        # nao sabe, evitando alucinacao.
        user = f"{context_header()}\n(empty — no relevant chunks retrieved)\n\nQUESTION:\n{question}"
        return system_prompt(mode), user

    blocks = []
    for i, e in enumerate(evidence, start=1):
        blocks.append(_format_chunk(i, e, max_chars=max_chars_per_chunk))
    context = "\n\n".join(blocks)
    user = f"{context_header()}\n{context}\n\nQUESTION:\n{question}"
    return system_prompt(mode), user


def build_sources_block(evidence: Sequence[Evidence]) -> str:
    """Bloco de fontes em texto plano (util p/ CLI/auditoria)."""
    if not evidence:
        return "Sources: (none)"
    lines = ["Sources:"]
    for i, e in enumerate(evidence, start=1):
        path = " / ".join(p for p in (e.h1, e.h2, e.h3) if p) or "(sem secao)"
        lines.append(f"[{i}] {e.file_name} — {path} (chunk_id={e.chunk_id}, score={e.score:.3f})")
    return "\n".join(lines)
