from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Document:
    """Documento de entrada já parseado, pronto para chunking."""

    doc_id: str
    source_path: str
    file_name: str
    file_hash: str
    body: str
    title: Optional[str] = None
    language: str = "und"
    doc_type: Optional[str] = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    ingested_at: int = 0


@dataclass(frozen=True)
class Chunk:
    """Unidade indexável. `chunk_id` estável (só muda quando o texto muda)."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    chunk_text: str
    char_start: int
    char_end: int
    token_count: int
    char_count: int
    parents: tuple[tuple[int, str], ...] = field(default_factory=tuple)
    is_appendix: bool = False


@dataclass(frozen=True)
class Evidence:
    """Resultado de retrieval com rastreabilidade."""

    chunk_id: str
    doc_id: str
    source_path: str
    file_name: str
    score: float
    chunk_index: int
    snippet: str
    h1: Optional[str] = None
    h2: Optional[str] = None
    h3: Optional[str] = None
    title: Optional[str] = None
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Answer:
    text: str
    mode: str
    evidence: tuple[Evidence, ...]
    confidence: float
    insufficient_context: bool = False
    conflict: bool = False
