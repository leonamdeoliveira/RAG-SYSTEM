from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.rag.manifest import Manifest
from pipeline.rag.markdown_loader import MarkdownLoader


def _write(p: Path, content: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _loader(tmp_path: Path) -> tuple[MarkdownLoader, Manifest]:
    root = tmp_path / "data"
    root.mkdir()
    manifest_path = tmp_path / "index" / "manifest.json"
    manifest = Manifest(manifest_path)
    return MarkdownLoader(root, manifest), manifest


def test_scan_finds_md_recursively(tmp_path):
    loader, _ = _loader(tmp_path)
    _write(loader.root / "a.md", "# A\n")
    _write(loader.root / "sub" / "b.md", "# B\n")
    _write(loader.root / "sub" / "c.txt", "ignore")
    paths, hashes, doc_ids = loader.scan()
    rels = sorted(hashes.keys())
    assert rels == ["a.md", "sub/b.md"]
    assert len(hashes) == 2
    assert len(doc_ids) == 2


def test_diff_detects_new_changed_removed(tmp_path):
    loader, manifest = _loader(tmp_path)
    a = _write(loader.root / "a.md", "# A v1\n")
    b = _write(loader.root / "b.md", "# B\n")

    # primeira passagem: tudo novo
    diff, hashes, doc_ids = loader.compute_diff()
    assert diff.new == sorted(diff.new)  # deterministico
    assert set(diff.new) == {"a.md", "b.md"}
    assert not diff.changed and not diff.removed

    # registra no manifesto
    for rel in diff.new:
        doc = loader.load(loader.root / rel, hashes[rel], doc_ids[rel])
        from pipeline.rag.manifest import ManifestEntry

        manifest.upsert(
            ManifestEntry(
                doc_id=doc.doc_id,
                source_path=doc.source_path,
                file_name=doc.file_name,
                file_hash=doc.file_hash,
                ingested_at=doc.ingested_at,
                language=doc.language,
                doc_type=doc.doc_type,
                tags=doc.tags,
            )
        )
    manifest.save()

    # segunda passagem: a mudou, b igual, remover c (nao existe)
    _write(loader.root / "a.md", "# A v2\n")
    diff2, _, _ = loader.compute_diff()
    assert "a.md" in diff2.changed
    assert "b.md" in diff2.unchanged

    # remover b
    (loader.root / "b.md").unlink()
    diff3, _, _ = loader.compute_diff()
    assert diff3.removed  # algum doc_id
    assert "a.md" not in diff3.removed


def test_front_matter_extraction(tmp_path):
    loader, _ = _loader(tmp_path)
    p = _write(
        loader.root / "doc.md",
        "---\ntitle: Meu Doc\ntype: manual\ntags: [a, b]\nlang: pt\n---\n# Meu Doc\nCorpo.\n",
    )
    doc = loader.load(p, "hash", "did")
    assert doc.title == "Meu Doc"
    assert doc.doc_type == "manual"
    assert doc.language == "pt"
    assert tuple(doc.tags) == ("a", "b")
    assert doc.body.startswith("# Meu Doc")


def test_title_from_first_heading_when_no_frontmatter(tmp_path):
    loader, _ = _loader(tmp_path)
    p = _write(loader.root / "x.md", "# Titulo Do Topo\nTexto.\n")
    doc = loader.load(p, "h", "did")
    assert doc.title == "Titulo Do Topo"


def test_doc_id_stable_across_content_changes(tmp_path):
    loader, _ = _loader(tmp_path)
    p = _write(loader.root / "p.md", "v1")
    _, hashes, doc_ids = loader.scan()
    did1 = doc_ids["p.md"]
    _write(p, "v2 conteudo diferente")
    _, _, doc_ids2 = loader.scan()
    assert doc_ids2["p.md"] == did1  # doc_id nao muda com conteudo


def test_manifest_save_load_roundtrip(tmp_path):
    from pipeline.rag.manifest import ManifestEntry

    path = tmp_path / "manifest.json"
    m = Manifest(path)
    m.upsert(
        ManifestEntry(
            doc_id="d1",
            source_path="a.md",
            file_name="a.md",
            file_hash="h1",
            ingested_at=42,
            language="pt",
            doc_type="manual",
            tags=("t1", "t2"),
        )
    )
    m.save()
    m2 = Manifest(path)
    e = m2.get("d1")
    assert e is not None
    assert e.file_hash == "h1"
    assert e.tags == ("t1", "t2")
    assert e.language == "pt"


def test_language_heuristic_pt_vs_en(tmp_path):
    loader, _ = _loader(tmp_path)
    pt = _write(loader.root / "pt.md", "Este é um texto sobre o uso de dados para a empresa.")
    en = _write(loader.root / "en.md", "This is a text about the use of data for the company.")
    dpt = loader.load(pt, "h", "d1")
    den = loader.load(en, "h", "d2")
    assert dpt.language == "pt"
    assert den.language == "en"


def test_doc_type_inferred_from_content(tmp_path):
    loader, _ = _loader(tmp_path)
    p = _write(loader.root / "x.md", "# Manual do Usuario\nComo usar o sistema.")
    doc = loader.load(p, "h", "d1")
    assert doc.doc_type == "manual"