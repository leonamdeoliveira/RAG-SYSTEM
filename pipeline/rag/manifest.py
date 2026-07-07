"""Manifesto do corpus — fonte da verdade para ingestão incremental.

O Zvec armazena chunks, mas não sabe quais *arquivos* existem no corpus.
O manifesto (JSON em disco) mapeia `doc_id -> {source_path, file_hash, ingested_at,
language, doc_type, tags}`, permitindo detectar:
  - novos arquivos (não estão no manifesto)
  - alterados   (file_hash diferente)
  - removidos   (no manifesto, ausentes no filesystem)

Decisão D10 (ARCHITECTURE.md): manifesto fora do Zvec, simples e auditável.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from pipeline.rag.utils.logging import get_logger

log = get_logger("app.manifest")

MANIFEST_FILENAME = "manifest.json"


@dataclass
class ManifestEntry:
    doc_id: str
    source_path: str
    file_name: str
    file_hash: str
    ingested_at: int
    language: str = "und"
    doc_type: Optional[str] = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ManifestEntry":
        return cls(
            doc_id=d["doc_id"],
            source_path=d["source_path"],
            file_name=d.get("file_name", Path(d["source_path"]).name),
            file_hash=d["file_hash"],
            ingested_at=int(d.get("ingested_at", 0)),
            language=d.get("language", "und"),
            doc_type=d.get("doc_type"),
            tags=tuple(d.get("tags", [])),
        )


@dataclass
class CorpusDiff:
    """Resultado da comparação filesystem vs manifesto.

    - `new`     : arquivos presentes no FS e ausentes no manifesto
    - `changed` : presentes em ambos, hash diferente
    - `removed` : presentes no manifesto, ausentes no FS
    - `unchanged`: presentes em ambos, hash igual (pular reindexação)
    """

    new: list[str]  # source_path relativos
    changed: list[str]
    removed: list[str]  # doc_id
    unchanged: list[str]  # source_path

    def has_changes(self) -> bool:
        return bool(self.new or self.changed or self.removed)


class Manifest:
    """Persistência do estado do corpus em `index/manifest.json`."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._entries: dict[str, ManifestEntry] = {}  # doc_id -> entry
        self._by_path: dict[str, str] = {}  # source_path -> doc_id
        self._load()

    # -------------------------------------------------------- IO

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("manifesto corrompido, ignorando (%s)", e)
            return
        for d in data.get("entries", []):
            try:
                e = ManifestEntry.from_dict(d)
                self._entries[e.doc_id] = e
                self._by_path[e.source_path] = e.doc_id
            except (KeyError, TypeError) as e:
                log.warning("entrada de manifesto invalida ignorada: %s", e)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"entries": [e.to_dict() for e in self._entries.values()]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # -------------------------------------------------------- acesso

    def get(self, doc_id: str) -> Optional[ManifestEntry]:
        return self._entries.get(doc_id)

    def get_by_path(self, source_path: str) -> Optional[ManifestEntry]:
        did = self._by_path.get(source_path)
        return self._entries.get(did) if did else None

    def upsert(self, entry: ManifestEntry) -> None:
        self._entries[entry.doc_id] = entry
        self._by_path[entry.source_path] = entry.doc_id

    def remove(self, doc_id: str) -> None:
        e = self._entries.pop(doc_id, None)
        if e:
            self._by_path.pop(e.source_path, None)

    def all_doc_ids(self) -> list[str]:
        return list(self._entries.keys())

    def diff(self, current_paths: list[str], current_hashes: dict[str, str]) -> CorpusDiff:
        new, changed, unchanged = [], [], []
        for p in current_paths:
            e = self.get_by_path(p)
            if e is None:
                new.append(p)
            elif e.file_hash != current_hashes[p]:
                changed.append(p)
            else:
                unchanged.append(p)
        current_set = set(current_paths)
        removed = [e.doc_id for e in self._entries.values() if e.source_path not in current_set]
        return CorpusDiff(new=new, changed=changed, removed=removed, unchanged=unchanged)
