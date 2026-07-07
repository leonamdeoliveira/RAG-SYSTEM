"""Templates de prompt interno — modo-answer, citations, extractive, study.

Principios (plan §2.6):
  - "Use only the provided context."
  - "If the answer is not in the context, say you do not know based on the documents."
  - "Never invent facts."
  - "Cite the supporting chunks using [n]."
"""

from __future__ import annotations

from textwrap import dedent

SYSTEM_BASE = dedent(
    """\
    You are a grounded retrieval-augmented assistant. Strict rules:
    1. Use ONLY the context below. No outside knowledge.
    2. If the answer is not in the context, say:
       "Based on the documents provided, I could not find an answer."
    3. Never invent facts, numbers, names, dates, or citations.
    4. If two context passages conflict, point out the conflict and cite both.
    5. Answer in the same language as the question (default: Portuguese).
    6. Be concise; use source wording when possible.
    """
)

CITATION_RULES = dedent(
    """\
    Citation format:
    - End each factual claim with [n] (n = context chunk number).
    - At the end, list sources as:
      [n] <file_name> — <section path> (chunk_id)
    """
)

MODE_INSTRUCTIONS = {
    "answer": dedent(
        """\
        MODE: answer.
        Answer the question directly using only the context. No sources list.
        """
    ),
    "answer_with_citations": dedent(
        """\
        MODE: answer_with_citations.
        Answer the question and cite each factual claim with [n].
        Append a "Sources:" section listing all cited chunks.
        """
    )
    + CITATION_RULES,
    "extractive_summary": dedent(
        """\
        MODE: extractive_summary.
        Select and output the most relevant passages from the context verbatim.
        Prefix each with [n]. Keep original wording. Do not paraphrase.
        """
    )
    + CITATION_RULES,
    "study_mode": dedent(
        """\
        MODE: study_mode.
        Act as a didactic tutor using ONLY the context.
        - Explain in structured steps.
        - Define jargon on first use.
        - End with "Key points:" summary of what the documents say.
        - Cite [n] for every statement.
        """
    )
    + CITATION_RULES,
}

VALID_MODES = tuple(MODE_INSTRUCTIONS.keys())


def system_prompt(mode: str) -> str:
    if mode not in MODE_INSTRUCTIONS:
        raise ValueError(f"modo de resposta invalido: {mode}. Validos: {VALID_MODES}")
    return SYSTEM_BASE + "\n" + MODE_INSTRUCTIONS[mode]


def context_header() -> str:
    return "CONTEXT (retrieved chunks):"
