"""Chunking semântico orientado à estrutura do Markdown.

Estratégia (ver ARCHITECTURE.md §2.2):
  1. Parse em blocos preservando a árvore de headings.
  2. Cada bloco folha vira *seed chunk* carregando o caminho de headings (parents).
  3. Acumulação: une blocos irmãos pequenos (< min_tokens) até atingir o alvo.
  4. Subdivisão: se um bloco > max_tokens, divide por parágrafos e depois por
     janela deslizante por tokens com overlap. Trata tabelas, listas, código.
  5. chunk_id estável: doc_id + "_" + sha1(texto normalizado)[:12].
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from pipeline.rag.models import Chunk, Document
from pipeline.rag.utils.hashing import sha1_text
from pipeline.rag.utils.logging import get_logger
from pipeline.rag.utils.tokenization import count_tokens, normalize_ws

log = get_logger("app.chunker")

# Defaults — alvo 300–600, máx 800, overlap 15%.
DEFAULT_MIN_TOKENS = 80
DEFAULT_TARGET_TOKENS = 600
DEFAULT_MAX_TOKENS = 800
DEFAULT_OVERLAP_RATIO = 0.15

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)(?:\s+#+\s*)?$", re.MULTILINE)
_FENCED = re.compile(r"^(\s*```)(.*)$", re.MULTILINE)
_GFM_TABLE_ROW = re.compile(r"^\|.+\|\s*$")
_GFM_TABLE_SEP = re.compile(r"^\|?[\s:-]+\|[\s:-|]+\|?\s*$")
_LIST_ITEM = re.compile(r"^(\s*)([-*+]\s+|\d+[.)]\s+)(.*)$")

_APPENDIX_HEADINGS = {"notas", "footnotes", "footnote", "apêndice", "apendice", "anexo", "appendix"}


@dataclass
class _Block:
    """Bloco de Markdown com seus headings ancestrais."""

    text: str
    char_start: int
    char_end: int
    parents: tuple[tuple[int, str], ...] = field(default_factory=tuple)
    kind: str = "text"
    is_appendix: bool = False


@dataclass
class _Heading:
    level: int
    text: str
    pos: int


class Chunker:
    def __init__(
        self,
        target_tokens: int = DEFAULT_TARGET_TOKENS,
        min_tokens: int = DEFAULT_MIN_TOKENS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    ) -> None:
        if min_tokens > target_tokens:
            raise ValueError("min_tokens deve ser <= target_tokens")
        if target_tokens > max_tokens:
            raise ValueError("target_tokens deve ser <= max_tokens")
        if not 0.0 <= overlap_ratio < 0.5:
            raise ValueError("overlap_ratio deve estar em [0, 0.5)")
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_ratio = overlap_ratio

    # ------------------------------------------------------------------ public

    def chunk(self, doc: Document) -> list[Chunk]:
        blocks = self._parse_blocks(doc.body)
        seeds: list[_Block] = []
        for b in blocks:
            seeds.extend(self._split_block(b))
        seeds = self._accumulate(seeds)
        chunks: list[Chunk] = []
        chunk_idx = 0
        for b in seeds:
            text = b.text.strip("\n")
            if not text.strip():
                continue
            tc = count_tokens(text)
            if self._is_noise(text, tc):
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_{sha1_text(normalize_ws(text))[:12]}",
                    doc_id=doc.doc_id,
                    chunk_index=chunk_idx,
                    chunk_text=text,
                    char_start=b.char_start,
                    char_end=b.char_end,
                    token_count=tc,
                    char_count=len(text),
                    parents=b.parents,
                    is_appendix=b.is_appendix,
                )
            )
            chunk_idx += 1
        if chunks and len(chunks) == 1 and chunks[0].token_count < self.min_tokens:
            log.debug("doc curto virou 1 chunk (ok): %s", doc.doc_id)
        return chunks

    # ------------------------------------------------------------------ parsing

    def _parse_blocks(self, body: str) -> list[_Block]:
        """Divide o Markdown em blocos respeitando headings, fenced code e tabelas."""
        headings = [_Heading(m.group(1).count("#"), m.group(2).strip(), m.start()) for m in _ATX_HEADING.finditer(body)]
        sections = self._sections_by_headings(body, headings)

        blocks: list[_Block] = []
        for parents, start, end, is_appendix in sections:
            section_text = body[start:end]
            if not section_text.strip():
                continue
            blocks.extend(self._leaf_blocks(section_text, start, parents, is_appendix))
        return blocks

    def _sections_by_headings(self, body: str, headings: list[_Heading]):
        """Retorna lista de (parents, start, end, is_appendix)."""
        if not headings:
            return [((), 0, len(body), False)]

        # Se há texto antes do primeiro heading, vira seção sem parents.
        sections = []
        first = headings[0].pos
        if first > 0 and body[:first].strip():
            sections.append(((), 0, first, False))

        for i, h in enumerate(headings):
            level = h.level
            parents_list: list[tuple[int, str]] = []
            for prev in headings[:i]:
                if prev.level < level:
                    parents_list.append((prev.level, prev.text))
            parents_list.sort(key=lambda x: x[0])
            parents = tuple(parents_list)

            end = headings[i + 1].pos if i + 1 < len(headings) else len(body)
            is_appendix = h.text.strip().lower().rstrip(":") in _APPENDIX_HEADINGS
            sections.append((parents, h.pos, end, is_appendix))
        return sections

    def _leaf_blocks(self, section_text: str, base_offset: int, parents, is_appendix: bool) -> list[_Block]:
        """Quebra uma seção em blocos atômicos: fenced code, tabelas, listas, paragrafos."""
        lines = section_text.splitlines(keepends=True)
        blocks: list[_Block] = []
        i = 0
        offset = base_offset
        buf: list[str] = []
        buf_start = offset

        def flush(kind: str = "text"):
            nonlocal buf, buf_start
            if buf:
                txt = "".join(buf).strip("\n")
                if txt.strip():
                    blocks.append(_Block(txt, buf_start, buf_start + len("".join(buf)), parents, kind, is_appendix))
                buf = []
                buf_start = offset

        while i < len(lines):
            line = lines[i]
            stripped = line.lstrip()

            # Fenced code block
            if stripped.startswith("```"):
                block_lines = [line]
                block_start = offset
                i += 1
                offset += len(line)
                while i < len(lines):
                    bline = lines[i]
                    block_lines.append(bline)
                    offset += len(bline)
                    if bline.lstrip().startswith("```"):
                        i += 1
                        break
                    i += 1
                flush()
                blocks.append(
                    _Block("".join(block_lines).strip("\n"), block_start, offset, parents, "code", is_appendix)
                )
                continue

            # GFM table
            if _GFM_TABLE_ROW.match(line) and i + 1 < len(lines) and _GFM_TABLE_SEP.match(lines[i + 1].strip()):
                block_lines = [line]
                block_start = offset
                i += 1
                offset += len(line)
                while i < len(lines) and _GFM_TABLE_ROW.match(lines[i]):
                    block_lines.append(lines[i])
                    offset += len(lines[i])
                    i += 1
                flush()
                blocks.append(
                    _Block("".join(block_lines).strip("\n"), block_start, offset, parents, "table", is_appendix)
                )
                continue

            # Listas (bullet/ordered) — bloco proprio preservando lead-in
            if _LIST_ITEM.match(line):
                list_lines, consumed, new_offset = self._collect_list(lines, i, offset)
                lead_in = self._pop_lead_in(buf)
                if lead_in:
                    flush()
                block_text = (lead_in + "\n" if lead_in else "") + "".join(list_lines)
                blocks.append(
                    _Block(block_text.strip("\n"), offset, new_offset, parents, "list", is_appendix)
                )
                offset = new_offset
                i += consumed
                buf_start = offset
                continue

            # Headings internos (sub-headings dentro da seção) — tornam-se seus próprios blocos
            line_stripped = line.strip()
            m = re.match(r"^(#{1,6})\s+(.*?)(?:\s+#+\s*)?$", line_stripped)
            if m:
                flush()
                level = m.group(1).count("#")
                text = m.group(2).strip()
                new_parents = tuple(p for p in parents if p[0] < level) + ((level, text),)
                blocks.append(_Block(line.strip("\n"), offset, offset + len(line), new_parents, "heading", is_appendix))
                offset += len(line)
                i += 1
                continue

            buf.append(line)
            offset += len(line)
            i += 1
        flush()
        return blocks

    # ------------------------------------------------------------------ splitting

    def _split_block(self, b: _Block) -> list[_Block]:
        tokens = count_tokens(b.text)
        if tokens <= self.max_tokens:
            return [b]

        if b.kind == "code":
            return self._split_code(b)
        if b.kind == "table":
            return self._split_table(b)
        if b.kind == "list":
            return self._split_list(b)
        return self._split_paragraphs(b)

    def _split_code(self, b: _Block) -> list[_Block]:
        lines = b.text.splitlines(keepends=True)
        if not lines:
            return [b]
        fence_open = lines[0]
        body_lines = lines[1:]
        fence_close = ""
        if body_lines and body_lines[-1].lstrip().startswith("```"):
            fence_close = body_lines[-1]
            body_lines = body_lines[:-1]

        out: list[_Block] = []
        buf: list[str] = []
        buf_tokens = 0
        for ln in body_lines:
            t = count_tokens(ln)
            if buf and buf_tokens + t > self.max_tokens:
                out.append(self._make_code_chunk(b, fence_open, buf, fence_close))
                buf = []
                buf_tokens = 0
            buf.append(ln)
            buf_tokens += t
        if buf:
            out.append(self._make_code_chunk(b, fence_open, buf, fence_close))
        return out

    def _make_code_chunk(self, parent: _Block, fence_open: str, body_lines: list[str], fence_close: str) -> _Block:
        text = fence_open + "".join(body_lines) + (fence_close or "```\n")
        return _Block(text.strip("\n"), parent.char_start, parent.char_end, parent.parents, "code", parent.is_appendix)

    def _collect_list(self, lines: list[str], start: int, offset: int) -> tuple[list[str], int, int]:
        """Coleta linhas consecutivas de uma lista (itens + continuações indentadas + blanks entre itens)."""
        out: list[str] = []
        i = start
        cur_offset = offset
        in_list = True
        while i < len(lines):
            ln = lines[i]
            if _LIST_ITEM.match(ln):
                out.append(ln)
                cur_offset += len(ln)
                i += 1
                in_list = True
                continue
            stripped = ln.strip()
            if stripped == "":
                # blank entre itens: tolera 1 linha em branco (lista solta)
                if i + 1 < len(lines) and _LIST_ITEM.match(lines[i + 1]):
                    out.append(ln)
                    cur_offset += len(ln)
                    i += 1
                    continue
                break
            # continuação indentada (sub-item ou paragrafo do item)
            if ln[:1].isspace() and in_list:
                out.append(ln)
                cur_offset += len(ln)
                i += 1
                continue
            break
        return out, i - start, cur_offset

    def _pop_lead_in(self, buf: list[str]) -> str:
        """Extrai e remove do buffer o ultimo paragrafo (lead-in) imediatamente antes da lista."""
        if not buf:
            return ""
        last_blank = -1
        for idx, ln in enumerate(buf):
            if ln.strip() == "":
                last_blank = idx
        lead = buf[last_blank + 1 :]
        del buf[last_blank + 1 :]
        # remove trailing blank de buf se ficou solto
        while buf and buf[-1].strip() == "":
            buf.pop()
        txt = "".join(lead).strip("\n")
        return txt

    def _split_table(self, b: _Block) -> list[_Block]:
        lines = b.text.splitlines(keepends=True)
        if len(lines) < 3:
            return [b]
        header = lines[0]
        sep = lines[1]
        rows = lines[2:]
        out: list[_Block] = []
        buf_rows: list[str] = []
        buf_tokens = count_tokens(header) + count_tokens(sep)
        for r in rows:
            t = count_tokens(r)
            if buf_rows and buf_tokens + t > self.max_tokens:
                text = (header + sep + "".join(buf_rows)).strip("\n")
                out.append(_Block(text, b.char_start, b.char_end, b.parents, "table", b.is_appendix))
                buf_rows = []
                buf_tokens = count_tokens(header) + count_tokens(sep)
            buf_rows.append(r)
            buf_tokens += t
        if buf_rows:
            text = (header + sep + "".join(buf_rows)).strip("\n")
            out.append(_Block(text, b.char_start, b.char_end, b.parents, "table", b.is_appendix))
        return out

    def _split_list(self, b: _Block) -> list[_Block]:
        """Divide listas grandes por itens, preservando o lead-in em cada subchunk."""
        # separa lead-in (paragrafo antes da primeira lista)
        lines = b.text.splitlines(keepends=True)
        lead_in = ""
        first_item_idx = None
        for idx, ln in enumerate(lines):
            if _LIST_ITEM.match(ln):
                first_item_idx = idx
                break
        if first_item_idx is None:
            return [b]
        if first_item_idx > 0:
            lead_in = "".join(lines[:first_item_idx]).strip("\n")
        item_lines = lines[first_item_idx:]

        # agrupa itens individuais
        items: list[str] = []
        cur: list[str] = []
        for ln in item_lines:
            if _LIST_ITEM.match(ln):
                if cur:
                    items.append("".join(cur))
                cur = [ln]
            else:
                cur.append(ln)
        if cur:
            items.append("".join(cur))

        out: list[_Block] = []
        group: list[str] = []
        group_tokens = count_tokens(lead_in) if lead_in else 0
        for it in items:
            it_t = count_tokens(it)
            if group and group_tokens + it_t > self.max_tokens:
                out.append(self._make_list_chunk(b, lead_in, group))
                group = []
                group_tokens = count_tokens(lead_in) if lead_in else 0
            group.append(it)
            group_tokens += it_t
        if group:
            out.append(self._make_list_chunk(b, lead_in, group))
        return out

    def _make_list_chunk(self, parent: _Block, lead_in: str, items: list[str]) -> _Block:
        text = (lead_in + "\n" if lead_in else "") + "".join(items)
        return _Block(text.strip("\n"), parent.char_start, parent.char_end, parent.parents, "list", parent.is_appendix)

    def _split_paragraphs(self, b: _Block) -> list[_Block]:
        # Texto longo genérico: divide por paragrafos (blank line) e depois por janela.
        paras = re.split(r"\n\s*\n", b.text)
        out: list[_Block] = []
        for p in paras:
            p = p.strip()
            if not p:
                continue
            t = count_tokens(p)
            if t <= self.max_tokens:
                out.append(_Block(p, b.char_start, b.char_end, b.parents, b.kind, b.is_appendix))
            else:
                out.extend(self._window_split(p, b))
        return out

    def _window_split(self, text: str, parent: _Block) -> list[_Block]:
        """Janela deslizante por tokens (palavras) com overlap."""
        words = text.split()
        out: list[_Block] = []
        step = max(1, int(self.max_tokens * (1 - self.overlap_ratio)))
        total = len(words)
        i = 0
        while i < total:
            window = words[i : i + self.max_tokens]
            if not window:
                break
            chunk_text = " ".join(window)
            out.append(_Block(chunk_text, parent.char_start, parent.char_end, parent.parents, parent.kind, parent.is_appendix))
            if i + len(window) >= total:
                break
            i += step
        return out

    # ------------------------------------------------------------------ accumulate

    def _accumulate(self, seeds: list[_Block]) -> list[_Block]:
        """Une blocos pequenos mesmos parents para evitar chunks sub-mínimos."""
        if not seeds:
            return []
        out: list[_Block] = []
        cur: Optional[_Block] = None
        cur_tokens = 0
        for b in seeds:
            bt = count_tokens(b.text)
            if cur is None:
                cur, cur_tokens = b, bt
                continue
            same_lineage = cur.parents == b.parents and cur.is_appendix == b.is_appendix
            if (cur_tokens < self.min_tokens or bt < self.min_tokens) and same_lineage and cur_tokens + bt <= self.max_tokens:
                merged_text = cur.text + "\n\n" + b.text
                merged_kind = cur.kind if cur.kind == b.kind else f"{cur.kind}+{b.kind}"
                cur = _Block(
                    merged_text,
                    cur.char_start,
                    b.char_end,
                    cur.parents,
                    merged_kind,
                    cur.is_appendix,
                )
                cur_tokens += bt
            else:
                out.append(cur)
                cur, cur_tokens = b, bt
        if cur is not None:
            out.append(cur)
        return out

    # ------------------------------------------------------------------ noise filter

    def _is_noise(self, text: str, token_count: int) -> bool:
        """Filtra chunks que sao apenas numeros, pontuacao ou whitespace estrutural."""
        stripped = re.sub(r"[\s\d.,;:()\[\]{}#*_\-+|/\\%$\^&=<>@!?\"]+", "", text)
        if len(stripped) < 5:
            return True
        if token_count < self.min_tokens and len(stripped) < 15:
            return True
        digit_ratio = len(re.sub(r"[^\d]", "", text)) / max(len(text), 1)
        if digit_ratio > 0.4 and len(stripped) < 30:
            return True
        return False
