"""Camada 1 — Ingestão de Markdown.

Responsabilidade (ARCHITECTURE.md §2.1):
  - varrer diretório raiz recursivamente por `*.md`
  - detectar novos/alterados/removidos comparando file_hash com o manifesto
  - parse de front-matter YAML (sem dep obrigatória de pyyaml — fallback)
  - extrair metadados: doc_id, source_path, file_name, file_hash, ingested_at,
    language (langdetect opcional, fallback), doc_type (best-effort), tags, title
  - emitir `Document` pronto para o chunker

`doc_id` = sha1(source_path relativo) — estável entre runs, independe do conteúdo
(decisão D9).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from pipeline.rag.manifest import CorpusDiff, Manifest
from pipeline.rag.models import Document
from pipeline.rag.text_cleaner import clean_text, clean_markdown
from pipeline.rag.utils.hashing import sha256_file, stable_doc_id
from pipeline.rag.utils.logging import get_logger

log = get_logger("app.loader")

_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---(?:\s*\n|\s*$)", re.DOTALL)
_ATX_TITLE = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.MULTILINE)

_DOC_TYPE_HINTS = {
    "manual": ("manual", "guia", "guide", "handbook"),
    "relatorio": ("relatorio", "report", "relatório"),
    "estudo": ("estudo", "study", "research", "pesquisa"),
    "codigo": ("codigo", "code", "api", "reference"),
    "politica": ("politica", "policy", "termos", "terms"),
}


class MarkdownLoader:
    def __init__(self, root: str | Path, manifest: Manifest) -> None:
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"diretório raiz não existe: {self.root}")
        self.manifest = manifest

    # -------------------------------------------------------- scan

    def scan(self) -> tuple[list[Path], dict[str, str], dict[str, str]]:
        """Varre `*.md` recursivamente.

        Retorna:
          - lista de Paths absolutos
          - dict rel_path -> file_hash
          - dict rel_path -> doc_id
        """
        paths: list[Path] = []
        for p in sorted(self.root.rglob("*.md")):
            if p.is_dir() or p.name.startswith("."):
                continue
            paths.append(p)
        hashes: dict[str, str] = {}
        doc_ids: dict[str, str] = {}
        for p in paths:
            rel = self._rel(p)
            hashes[rel] = sha256_file(str(p))
            doc_ids[rel] = stable_doc_id(rel)
        log.info("scan: %d arquivos .md em %s", len(paths), self.root)
        return paths, hashes, doc_ids

    def compute_diff(self) -> tuple[CorpusDiff, dict[str, str], dict[str, str]]:
        """Computa diff (new/changed/removed/unchanged) vs manifesto."""
        paths, hashes, doc_ids = self.scan()
        rel_paths = list(hashes.keys())
        diff = self.manifest.diff(rel_paths, hashes)
        log.info(
            "diff: %d novos, %d alterados, %d removidos, %d inalterados",
            len(diff.new),
            len(diff.changed),
            len(diff.removed),
            len(diff.unchanged),
        )
        return diff, hashes, doc_ids

    # -------------------------------------------------------- load

    def load(self, abs_path: Path, file_hash: str, doc_id: str) -> Document:
        """Lê um arquivo .md e produz um `Document` com metadados."""
        raw = abs_path.read_text(encoding="utf-8", errors="replace")
        body, front = self._strip_front_matter(raw)
        # Aplicar text preprocessing para melhorar qualidade dos chunks
        body = clean_text(body)
        body = clean_markdown(body)
        rel = self._rel(abs_path)

        title = front.get("title") or self._extract_title(body) or abs_path.stem
        language = self._detect_language(front, body)
        doc_type = front.get("type") or front.get("doc_type") or self._infer_doc_type(title, body, abs_path)
        tags = self._parse_tags(front.get("tags"))

        return Document(
            doc_id=doc_id,
            source_path=rel,
            file_name=abs_path.name,
            file_hash=file_hash,
            body=body,
            title=title,
            language=language,
            doc_type=doc_type,
            tags=tuple(tags),
            ingested_at=self._now_ms(),
        )

    # -------------------------------------------------------- helpers

    def _rel(self, abs_path: Path) -> str:
        return abs_path.relative_to(self.root).as_posix()

    def _strip_front_matter(self, raw: str) -> tuple[str, dict]:
        m = _FRONT_MATTER.match(raw)
        if not m:
            return raw, {}
        fm = m.group(1)
        body = raw[m.end():]
        return body, self._parse_front_matter(fm)

    def _parse_front_matter(self, fm: str) -> dict:
        """Parser YAML mínimo (chave: valor). Sem pyyaml -> apenas pares simples."""
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(fm)
            return data if isinstance(data, dict) else {}
        except ImportError:
            pass
        out: dict = {}
        for line in fm.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip().strip('"').strip("'")
        return out

    def _extract_title(self, body: str) -> Optional[str]:
        m = _ATX_TITLE.search(body)
        return m.group(1).strip() if m else None

    def _detect_language(self, front: dict, body: str) -> str:
        if "lang" in front or "language" in front:
            lang = front.get("lang") or front.get("language")
            if lang is not None and str(lang).strip():
                return str(lang).strip()[:8]
            return "und"
        try:
            from langdetect import detect  # type: ignore

            sample = body[:2000]
            if sample.strip():
                return detect(sample)
        except ImportError:
            pass
        except Exception:
            pass
        # heurística mínima PT/EN
        pt_hints = (" o ", " a ", " de ", " que ", " para ", " com ", " não ")
        en_hints = (" the ", " and ", " of ", " to ", " for ", " with ", " not ")
        s = " " + body[:3000].lower() + " "
        pt = sum(s.count(h) for h in pt_hints)
        en = sum(s.count(h) for h in en_hints)
        if pt == 0 and en == 0:
            return "und"
        return "pt" if pt > en else "en"

    def _infer_doc_type(self, title: str, body: str, path: Path) -> Optional[str]:
        hay = (title + " " + body[:1000] + " " + path.name).lower()
        for dtype, hints in _DOC_TYPE_HINTS.items():
            if any(h in hay for h in hints):
                return dtype
        return None

    def _parse_tags(self, tags) -> list[str]:
        if not tags:
            return []
        if isinstance(tags, (list, tuple)):
            return [str(t).strip() for t in tags if str(t).strip()]
        s = str(tags)
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        return [t.strip().strip('"').strip("'") for t in s.split(",") if t.strip()]

    def _now_ms(self) -> int:
        import time

        return int(time.time() * 1000)
