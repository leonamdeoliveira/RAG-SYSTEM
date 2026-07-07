from __future__ import annotations

from pipeline.rag.chunker import Chunker
from pipeline.rag.models import Document


def _doc(body: str, doc_id: str = "d1") -> Document:
    return Document(
        doc_id=doc_id,
        source_path=f"{doc_id}.md",
        file_name=f"{doc_id}.md",
        file_hash="h",
        body=body,
    )


def test_short_doc_single_chunk():
    body = "Apenas uma frase curta sobre nada."
    chunks = Chunker().chunk(_doc(body))
    assert len(chunks) == 1
    assert chunks[0].token_count < 80  # below min -> 1 chunk


def test_headings_create_parents():
    body = (
        "# Capitulo Um\nTexto curto do cap 1.\n\n"
        "## Secao A\nConteudo da secao A com algum texto.\n\n"
        "## Secao B\nConteudo da secao B com algum texto aqui tambem.\n"
    )
    chunks = Chunker(target_tokens=20, min_tokens=5, max_tokens=40).chunk(_doc(body))
    parents_levels = [dict(c.parents).keys() for c in chunks]
    # Pelo menos um chunk tem h1 e h2 como ancestors
    flat = [lv for pl in parents_levels for lv in pl]
    assert 1 in flat
    assert 2 in flat


def test_long_paragraph_split():
    body = "word " * 4000  # ~ 1000 tokens
    chunks = Chunker(target_tokens=300, min_tokens=80, max_tokens=400, overlap_ratio=0.15).chunk(_doc(body))
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 410  # tolerancia: window por palavras


def test_fenced_code_atomic_when_small():
    body = "Intro.\n\n```python\nprint('hi')\n```\n"
    chunks = Chunker().chunk(_doc(body))
    # code preservado como bloco proprio (ou fundido com intro se acumular)
    joined = "\n".join(c.chunk_text for c in chunks)
    assert "```python" in joined
    assert "print('hi')" in joined


def test_fenced_code_split_when_huge():
    code_lines = "\n".join(f"print({i})" for i in range(2000))
    body = f"```python\n{code_lines}\n```\n"
    chunks = Chunker(target_tokens=100, min_tokens=10, max_tokens=200).chunk(_doc(body))
    assert len(chunks) > 1
    for c in chunks:
        assert "```python" in c.chunk_text  # fence reaberta em cada subchunk


def test_gfm_table_preserved():
    body = (
        "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |\n"
    )
    chunks = Chunker().chunk(_doc(body))
    joined = "\n".join(c.chunk_text for c in chunks)
    assert "| A | B |" in joined


def test_appendix_flagged():
    body = "# Topo\nTexto.\n\n## Notas\nRodape extra.\n"
    chunks = Chunker(target_tokens=20, min_tokens=5, max_tokens=40).chunk(_doc(body))
    assert any(c.is_appendix for c in chunks)
    assert not all(c.is_appendix for c in chunks)


def test_chunk_id_stable_for_same_text():
    body = "# T\nParagrafo estavel.\n"
    a = Chunker().chunk(_doc(body, "d1"))
    b = Chunker().chunk(_doc(body, "d1"))
    assert a[0].chunk_id == b[0].chunk_id


def test_chunk_id_changes_when_text_changes():
    body1 = "# T\nTexto A.\n"
    body2 = "# T\nTexto B.\n"
    a = Chunker(min_tokens=2, target_tokens=10, max_tokens=20).chunk(_doc(body1, "d1"))
    b = Chunker(min_tokens=2, target_tokens=10, max_tokens=20).chunk(_doc(body2, "d1"))
    assert {c.chunk_id for c in a} != {c.chunk_id for c in b}


def test_chunk_index_monotonic():
    body = "\n\n".join(f"## Secao {i}\n{'palavra ' * 50}" for i in range(6))
    chunks = Chunker(target_tokens=80, min_tokens=20, max_tokens=120).chunk(_doc(body))
    idx = [c.chunk_index for c in chunks]
    assert idx == sorted(idx)
    assert idx[0] == 0


def test_long_list_split_preserves_lead_in():
    lead = "Principais recomendacoes do time:"
    items = "\n".join(f"- item numero {i} " + "palavra " * 15 for i in range(60))
    body = f"{lead}\n{items}\n"
    chunks = Chunker(target_tokens=150, min_tokens=30, max_tokens=200).chunk(_doc(body))
    assert len(chunks) > 1
    # cada subchunk deve carregar o lead-in (preservacao de contexto)
    for c in chunks:
        if "item" in c.chunk_text:
            assert lead in c.chunk_text


def test_short_list_kept_together():
    body = "Requisitos:\n- foo\n- bar\n- baz\n"
    chunks = Chunker().chunk(_doc(body))
    joined = "\n".join(c.chunk_text for c in chunks)
    assert joined.count("foo") == 1
    assert "- bar" in joined