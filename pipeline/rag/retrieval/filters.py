"""Builder de filtros SQL-like do Zvec (sintaxe v0.5.1 confirmada empiricamente).

Sintaxe suportada (mapeada em testes — ver ARCHITECTURE.md §3 "suposições"):
  - Igualdade:     field = 'value'        (NAO usar ==)
  - Comparacao:    field < N | <= | > | >= | !=
  - Strings:       'single' ou "double" quoted
  - Booleanos:     TRUE | FALSE  (maiusculo)
  - Combinadores:  AND | OR | NOT (maiusculo)
  - LIKE:          field LIKE 'prefix%'
  - IN (escalar):  field IN ('a', 'b')    -> OK para campos STRING/INT

NAO suportado nesta versao (limitacao real constatada):
  - Filtro sobre ARRAY_STRING (ex.: tags).
    Sintaxes testadas rejeitadas: tags = 'a', tags IN (..), CONTAIN/CONTAINS,
    ARRAY_CONTAINS, OVERLAPS, @>, &&.
  -> Filtragem por tags fica EM MEMORIA pos-recuperacao (ver `post_filter_tags`).

API:
  FilterBuilder().eq("doc_id","d1").eq("language","pt").build()
    -> "doc_id = 'd1' AND language = 'pt'"
  FilterBuilder().in_("doc_id", ["d1","d2"]).build()
    -> "doc_id IN ('d1', 'd2')"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional


def _quote_string(v: str) -> str:
    """Single-quote + escape interno (sintaxe SQUOTA_STRING do Zvec)."""
    return "'" + v.replace("'", "\\'") + "'"


def _format_value(v) -> str:
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return str(v)
    return _quote_string(str(v))


@dataclass
class FilterBuilder:
    """Constroi expressao SQL-like do Zvec encadeando condicoes AND.

    Uso:
        f = FilterBuilder().eq("doc_id","d1").gte("chunk_index", 2)
        f.build() -> "doc_id = 'd1' AND chunk_index >= 2"
        f.is_empty() -> False
    """

    parts: list[str] = field(default_factory=list)
    tags_filter: Optional[tuple[str, ...]] = None  # post-filtro em memoria

    # --- condicoes que viram string SQL ---

    def eq(self, name: str, value) -> "FilterBuilder":
        self.parts.append(f"{name} = {_format_value(value)}")
        return self

    def ne(self, name: str, value) -> "FilterBuilder":
        self.parts.append(f"{name} != {_format_value(value)}")
        return self

    def lt(self, name: str, value) -> "FilterBuilder":
        self.parts.append(f"{name} < {_format_value(value)}")
        return self

    def lte(self, name: str, value) -> "FilterBuilder":
        self.parts.append(f"{name} <= {_format_value(value)}")
        return self

    def gt(self, name: str, value) -> "FilterBuilder":
        self.parts.append(f"{name} > {_format_value(value)}")
        return self

    def gte(self, name: str, value) -> "FilterBuilder":
        self.parts.append(f"{name} >= {_format_value(value)}")
        return self

    def like(self, name: str, pattern: str) -> "FilterBuilder":
        self.parts.append(f"{name} LIKE {_format_value(pattern)}")
        return self

    def in_(self, name: str, values: Iterable) -> "FilterBuilder":
        vals = ", ".join(_format_value(v) for v in values)
        if not vals:
            # IN vazio nao e valido; forca condicao sempre-falsa
            self.parts.append("1 = 0")
            return self
        self.parts.append(f"{name} IN ({vals})")
        return self

    def raw(self, expr: str) -> "FilterBuilder":
        """Permite compor expressao arbitrary (caller responsavel pela sintaxe)."""
        self.parts.append(expr)
        return self

    # --- tags: post-filtro em memoria ---

    def with_tags(self, tags: Iterable[str]) -> "FilterBuilder":
        """Marca tags para filtragem pos-recuperacao (Zvec nao filtra ARRAY_STRING)."""
        self.tags_filter = tuple(t for t in tags if t)
        return self

    # --- saida ---

    def is_empty(self) -> bool:
        return not self.parts and not self.tags_filter

    def build(self) -> Optional[str]:
        """Retorna string SQL-like ou None se nao houver condicoes escalares."""
        if not self.parts:
            return None
        return " AND ".join(self.parts)

    def build_or(self) -> Optional[str]:
        if not self.parts:
            return None
        return " OR ".join(self.parts)


def post_filter_tags(evidence, tags: Optional[tuple[str, ...]]) -> bool:
    """Retorna True se `evidence` satisfaz o filtro de tags (None = passa)."""
    if not tags:
        return True
    # Evidence nao carrega tags diretamente; esperamos campo `tags` em fields.
    # Se ausente, conservador: nao passa (caller pode escolher dropar ou manter).
    ev_tags = getattr(evidence, "tags", None)
    if ev_tags is None:
        # Tentar ler de fields (dict) se existir
        fields = getattr(evidence, "fields", None)
        if isinstance(fields, dict):
            ev_tags = fields.get("tags")
    if ev_tags is None:
        return False
    return any(t in ev_tags for t in tags)
