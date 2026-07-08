import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

from pipeline.ocr.lmstudio_client import LMStudioClient, LMStudioClientError
from pipeline.ocr.model_loader import load_model_config, load_model_prompts
from pipeline.ocr.pdf_utils import PDFPage
from pipeline.ocr.document_model import (
    Document, Page, TextItem, HeadingItem,
    TableItem, BoundingBox,
)
from pipeline.ocr.ocr_engine.base import OCREngineError, EngineNotAvailableError
from pipeline.ocr.ocr_engine.router import OCRRouter
from pipeline.ocr.ocr_engine.config import HybridOCRConfig
from pipeline.ocr.ocr_engine.text_stats import tokenize, extract_numbers
from pipeline.ocr.ocr_engine.quality import QualityScorer, ItemQualityReport
from pipeline.ocr.ocr_engine.layout_engine import LayoutEngine

logger = logging.getLogger(__name__)


class PageResult:
    def __init__(
        self,
        page_num: int,
        method: str,
        processing_time: float,
        image_size: tuple[int, int],
        native_chars: int = 0,
        output_chars: int = 0,
        status: str = "ok",
    ):
        self.page_num = page_num
        self.method = method
        self.processing_time = processing_time
        self.image_size = image_size
        self.native_chars = native_chars
        self.output_chars = output_chars
        self.status = status

    def to_dict(self) -> dict:
        return {
            "page": self.page_num,
            "method": self.method,
            "processing_time_s": round(self.processing_time, 2),
            "image_size": list(self.image_size),
            "native_chars": self.native_chars,
            "output_chars": self.output_chars,
            "status": self.status,
        }


class OCRPipeline:
    def __init__(
        self,
        client: LMStudioClient,
        output_dir: Path,
        formats: list[str],
        model_name: str = "chandra-ocr-2",
        mode: str = "text-first",
        dpi: int = 200,
        resume: bool = False,
        basename: str = "documento",
        hybrid_config: Optional[HybridOCRConfig] = None,
        router: Optional[OCRRouter] = None,
        use_layout: bool = True,
        image_output_dir: Optional[Path] = None,
    ):
        self.client = client
        self.output_dir = output_dir
        self.formats = formats
        self.model_name = model_name
        self.model_config = load_model_config(model_name)
        self.prompts = load_model_prompts(model_name)
        self._output_is_html = self.model_config.get("default_prompt_format", "html") == "html"
        self._use_system_prompt = self.model_config.get("use_system_prompt", True)
        self._use_native_ref = self.model_config.get("use_native_text_reference", True)
        self._use_image_only = self.model_config.get("use_image_only", False)
        self._use_multi_image = self.model_config.get("use_multi_image", False)
        self._batch_size = self.model_config.get("batch_size", 0)
        self._max_tokens = self.model_config.get("max_tokens", 48000)
        self.mode = mode
        self.dpi = dpi
        self.resume = resume
        self.basename = basename
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hybrid_config = hybrid_config
        self.router = router
        self.use_layout = use_layout
        self.image_output_dir = image_output_dir or (output_dir / "images")
        self.scorer = QualityScorer()
        self.layout_engine: Optional[LayoutEngine] = None
        if self.use_layout:
            self.layout_engine = LayoutEngine(image_output_dir=self.image_output_dir)
        self._document: Optional[Document] = None

    def _partial_path(self, page_num: int, fmt: str) -> Path:
        return self.output_dir / f"page_{page_num:04d}.{fmt}.partial"

    EXT_MAP = {"markdown": "md", "html": "html", "json": "json"}

    def _output_path(self, fmt: str) -> Path:
        ext = self.EXT_MAP.get(fmt, fmt)
        return self.output_dir / f"{self.basename}.{ext}"

    def _metadata_path(self) -> Path:
        return self.output_dir / f"{self.basename}.metadata.json"

    def _document_path(self) -> Path:
        return self.output_dir / f"{self.basename}.document.json"

    def _load_partial(self, page_num: int, fmt: str) -> Optional[str]:
        path = self._partial_path(page_num, fmt)
        return path.read_text(encoding="utf-8") if path.exists() else None

    def _save_partial(self, page_num: int, fmt: str, content: str):
        self._partial_path(page_num, fmt).write_text(content, encoding="utf-8")

    def _save_metadata(self, results: list[PageResult]):
        data = [r.to_dict() for r in results]
        self._metadata_path().write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _save_document_model(self, doc: Document):
        self._document_path().write_text(doc.to_json(), encoding="utf-8")
        logger.info("Saved document model: %s", self._document_path())

    def _build_document_from_layout(self, fitz_doc) -> Document:
        if not self.layout_engine or not self.use_layout:
            return Document(filename="")
        import fitz
        from pipeline.ocr.pdf_utils import extract_native_text
        doc = Document(filename="")
        for i in range(len(fitz_doc)):
            fz_page = fitz_doc[i]
            page_num = i + 1
            rect = fz_page.rect
            pw, ph = rect.width, rect.height
            native_text = extract_native_text(fz_page)
            items, tables, pics = self.layout_engine.analyze_page(page_num, fz_page, pw, ph)
            page_items = items + tables + pics
            page = Page(
                page_num=page_num,
                width=int(pw),
                height=int(ph),
                items=page_items,
                native_text=native_text,
            )
            doc.pages.append(page)
            for item in page_items:
                doc.add_item(item)
                doc.add_to_body(item.id)
        doc.metadata["source"] = "layout"
        doc.metadata["pages"] = len(doc.pages)
        doc.metadata["total_items"] = len(doc.items)
        return doc

    def _enrich_document_with_ocr(
        self, doc: Document, fitz_doc,
    ) -> list[PageResult]:
        from pipeline.ocr.pdf_utils import render_page, extract_native_text

        results: list[PageResult] = []
        total = len(fitz_doc)

        for i in range(total):
            fz_page = fitz_doc[i]
            page_num = i + 1
            start = time.time()
            status = "ok"
            output_chars = 0
            method = "layout"

            doc_page = next((p for p in doc.pages if p.page_num == page_num), None)
            text_items = [
                it for it in (doc_page.items if doc_page else [])
                if isinstance(it, (TextItem, HeadingItem))
            ]

            force_ocr = (
                self.hybrid_config is None
                or self.hybrid_config.mode in ("legacy", "ai_only")
            )

            if not force_ocr and any(it.text.strip() for it in text_items):
                total_chars = sum(len(it.text) for it in text_items)
                logger.info(
                    "Page %d/%d: layout extracted %d items (%d chars), skipping OCR",
                    page_num, total, len(text_items), total_chars,
                )
                results.append(PageResult(
                    page_num=page_num,
                    method="layout",
                    processing_time=time.time() - start,
                    image_size=(int(fz_page.rect.width), int(fz_page.rect.height)),
                    native_chars=len(doc_page.native_text) if doc_page else 0,
                    output_chars=total_chars,
                    status="ok",
                ))
                continue

            native_text = extract_native_text(fz_page)
            image = render_page(fz_page, dpi=self.dpi)
            page = PDFPage(page_num=page_num, image=image, native_text=native_text)

            if self.router is not None:
                try:
                    result = self.router.process_with_fallback(page, fmt="markdown")
                    ocr_text = result.text
                    method = result.engine_name
                    conf = result.confidence
                except EngineNotAvailableError as e:
                    logger.error("Page %d failed: %s", page_num, e)
                    status = "error: lm_studio_needed"
                    results.append(PageResult(
                        page_num=page_num, method="ocr_fallback",
                        processing_time=time.time() - start,
                        image_size=image.size,
                        native_chars=len(native_text),
                        output_chars=0, status=status,
                    ))
                    continue
            else:
                try:
                    ocr_text = self.process_page(page, "markdown")
                    method = "ocr"
                    conf = 1.0
                except KeyboardInterrupt:
                    logger.warning("OCR interrompido pelo usuário na página %d", page_num)
                    raise
                except Exception as e:
                    logger.error("Page %d OCR failed: %s", page_num, e)
                    status = f"error: {e}"
                    results.append(PageResult(
                        page_num=page_num, method="ocr",
                        processing_time=time.time() - start,
                        image_size=image.size,
                        native_chars=len(native_text),
                        output_chars=0, status=status,
                    ))
                    continue

            if not doc_page:
                doc_page = Page(
                    page_num=page_num,
                    width=image.width,
                    height=image.height,
                    native_text=native_text,
                )
                doc.pages.append(doc_page)

            ocr_item = TextItem(
                text=ocr_text,
                page_num=page_num,
                confidence=conf,
                bbox=BoundingBox(x0=0, y0=0, x1=float(image.width), y1=float(image.height)),
            )
            doc_page.items.append(ocr_item)
            doc.add_item(ocr_item)
            doc.add_to_body(ocr_item.id)

            elapsed = time.time() - start
            output_chars = len(ocr_text)
            results.append(PageResult(
                page_num=page_num,
                method=method,
                processing_time=elapsed,
                image_size=image.size,
                native_chars=len(native_text),
                output_chars=output_chars,
                status=status,
            ))
            logger.info("Page %d/%d done in %.1fs [%s]", page_num, total, elapsed, method)

        return results

    def _score_and_mark(self, doc: Document, threshold: float = 0.70):
        all_items = list(doc.items.values())
        reports = self.scorer.score_document_items(all_items, threshold)

        low_items = [r for r in reports if not r.acceptable]
        ok_items = [r for r in reports if r.acceptable]

        logger.info(
            "Quality: %d/%d items OK (>=%.2f), %d LOW",
            len(ok_items), len(reports), threshold, len(low_items),
        )

        for r in low_items:
            logger.warning(
                "  Page %d [%s] score=%.3f (chars=%d)",
                r.page_num, r.item_id[:8], r.score, r.char_count,
            )

        doc.metadata["item_quality"] = {
            "total_items": len(reports),
            "ok_items": len(ok_items),
            "low_items": len(low_items),
            "threshold": threshold,
            "reports": [r.to_dict() for r in reports],
        }
        return reports

    @staticmethod
    def _strip_html_attrs(html: str) -> str:
        html = re.sub(r'\sdata-[a-zA-Z_-]+="[^"]*"', '', html)
        html = re.sub(r"\sdata-[a-zA-Z_-]+='[^']*'", '', html)
        html = re.sub(r'\s(?:class|style|id|data)=(?:"[^"]*"|\'[^\']*\'|[^\s>]+)', '', html)
        return html

    @staticmethod
    def _html_to_markdown(html: str) -> str:
        html = OCRPipeline._strip_html_attrs(html)
        for tag in ('table', 'div', 'ol', 'ul', 'p', 'pre'):
            if f'<{tag}' in html and f'</{tag}>' not in html:
                html += f'</{tag}>'
        # GLM-OCR pode retornar <tr> soltas sem <table>
        if '<tr' in html and '<table' not in html:
            html = '<table>' + html + '</table>'
        html = re.sub(r'<h([1-6])[^>]*>(.*?)</h\1>', lambda m: '#' * int(m.group(1)) + ' ' + m.group(2), html, flags=re.DOTALL)
        html = re.sub(r'<table[^>]*>(.*?)</table>', lambda m: OCRPipeline._table_to_markdown(m.group(1)), html, flags=re.DOTALL)
        html = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1', html, flags=re.DOTALL)
        html = re.sub(r'</?(?:ul|ol)[^>]*>', '', html)
        html = re.sub(r'<br\s*/?>\s*<br\s*/?>', '\n\n', html)
        html = re.sub(r'<br\s*/?>', '\n', html)

        def _replace_p(m: re.Match) -> str:
            content = m.group(1)
            content = re.sub(r'\s+', ' ', content).strip()
            return content + '\n' if content else ''
        html = re.sub(r'<p[^>]*>(.*?)</p>', _replace_p, html, flags=re.DOTALL)
        html = re.sub(r'<div[^>]*>(.*?)</div>', r'\1', html, flags=re.DOTALL)
        html = re.sub(r'<span[^>]*>(.*?)</span>', r'\1', html, flags=re.DOTALL)
        html = re.sub(r'<a[^>]*>(.*?)</a>', r'\1', html)
        html = re.sub(r'<img[^>]*alt="([^"]*)"[^>]*/>', r'[Imagem: \1]', html)
        html = re.sub(r'<img[^>]*/>', '', html)
        html = re.sub(r'<strong[^>]*>(.*?)</strong>', r'**\1**', html, flags=re.DOTALL)
        html = re.sub(r'<b[^>]*>(.*?)</b>', r'**\1**', html, flags=re.DOTALL)
        html = re.sub(r'<em[^>]*>(.*?)</em>', r'*\1*', html, flags=re.DOTALL)
        html = re.sub(r'<i[^>]*>(.*?)</i>', r'*\1*', html, flags=re.DOTALL)
        html = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', html, flags=re.DOTALL)
        html = re.sub(r'<pre[^>]*>(.*?)</pre>', r'```\n\1\n```', html, flags=re.DOTALL)
        html = re.sub(r'<th[^>]*>(.*?)</th>', r'\1', html, flags=re.DOTALL)
        html = re.sub(r'<td[^>]*>(.*?)</td>', r'\1', html, flags=re.DOTALL)
        html = re.sub(r'</?tr[^>]*>', '\n', html)
        html = re.sub(r'</?thead[^>]*>', '', html)
        html = re.sub(r'</?tbody[^>]*>', '', html)
        html = re.sub(r'</?caption[^>]*>', '', html)
        html = re.sub(r'</?h[1-6][^>]*>', '', html)
        html = re.sub(r'<[^>]+>', '', html)
        html = re.sub(r'<![^>]+>', '', html)
        html = re.sub(r'  +', ' ', html)
        html = re.sub(r'\n{3,}', '\n\n', html)
        html = re.sub(r'\n[ \t]+\n', '\n\n', html)
        html = re.sub(r'[ \t]+\n', '\n', html)
        return html.strip()

    @staticmethod
    def _table_to_markdown(table_html: str) -> str:
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, flags=re.DOTALL)
        if not rows:
            return table_html
        md_rows = []
        for row in rows:
            cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, flags=re.DOTALL)
            cell_texts = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            md_rows.append("| " + " | ".join(cell_texts) + " |")
        if not md_rows:
            return ""
        header_idx = None
        for i, row in enumerate(rows):
            if re.search(r'<th', row, re.IGNORECASE):
                header_idx = i
                break
        result = "\n".join(md_rows) + "\n"
        if header_idx is not None:
            header_cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', rows[header_idx], flags=re.DOTALL)
            sep = "| " + " | ".join(["---"] * len(header_cells)) + " |"
            lines = list(md_rows)
            lines.insert(header_idx + 1, sep)
            result = "\n".join(lines) + "\n"
        return result

    _REASONING_PREFIXES = (
        "the user wants", "the user asks", "the user provided",
        "the image is", "the image shows", "the document contains",
        "the screenshot", "the picture",
        "i can see", "i need to", "i will", "i think",
        "let me", "looking at", "this is a",
        "here is", "here's",
    )

    def _is_html_content(self, content: str) -> bool:
        stripped = content.strip()
        return bool(
            stripped.lower().startswith("<table")
            or stripped.lower().startswith("<html")
            or stripped.lower().startswith("<div")
            or stripped.lower().startswith("<h")
            or "<table" in stripped.lower()[:500]
        )

    def _is_doctags_content(self, content: str) -> bool:
        stripped = content.strip()
        return stripped.lower().startswith("<doctag") or "<doctag" in stripped.lower()[:200]

    @staticmethod
    def _doctags_to_markdown(content: str) -> str:
        content = re.sub(r'<loc_\d+>', '', content)
        content = re.sub(r'</?doctag>', '', content, flags=re.IGNORECASE)

        content = re.sub(
            r'<section_header[^>]*>(.*?)</section_header>',
            r'\n## \1\n', content, flags=re.DOTALL | re.IGNORECASE,
        )

        content = re.sub(
            r'<unordered_list[^>]*>(.*?)</unordered_list>',
            lambda m: OCRPipeline._list_to_md(m.group(1), ordered=False),
            content, flags=re.DOTALL | re.IGNORECASE,
        )
        content = re.sub(
            r'<ordered_list[^>]*>(.*?)</ordered_list>',
            lambda m: OCRPipeline._list_to_md(m.group(1), ordered=True),
            content, flags=re.DOTALL | re.IGNORECASE,
        )

        content = re.sub(
            r'<table[^>]*>(.*?)</table>',
            lambda m: OCRPipeline._doctags_table_to_md(m.group(1)),
            content, flags=re.DOTALL | re.IGNORECASE,
        )

        content = re.sub(
            r'<text[^>]*>(.*?)</text>',
            r'\1\n', content, flags=re.DOTALL | re.IGNORECASE,
        )

        content = re.sub(
            r'<page_header[^>]*>(.*?)</page_header>',
            r'', content, flags=re.DOTALL | re.IGNORECASE,
        )
        content = re.sub(
            r'<page_footer[^>]*>(.*?)</page_footer>',
            r'\n---\n*\1*\n', content, flags=re.DOTALL | re.IGNORECASE,
        )

        content = re.sub(r'<[^>]+>', '', content)
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r'[ \t]+\n', '\n', content)
        return content.strip()

    @staticmethod
    def _list_to_md(content: str, ordered: bool = False) -> str:
        items = re.findall(r'<list_item[^>]*>(.*?)</list_item>', content, flags=re.DOTALL | re.IGNORECASE)
        if not items:
            return content
        lines = []
        for i, item in enumerate(items):
            item = re.sub(r'<loc_\d+>', '', item)
            item = re.sub(r'<[^>]+>', '', item).strip()
            prefix = f"{i + 1}. " if ordered else "- "
            lines.append(f"{prefix}{item}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _doctags_table_to_md(content: str) -> str:
        rows = re.findall(r'<table_row[^>]*>(.*?)</table_row>', content, flags=re.DOTALL | re.IGNORECASE)
        if not rows:
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', content, flags=re.DOTALL | re.IGNORECASE)
        if not rows:
            return content
        md_rows = []
        for row in rows:
            cells = re.findall(r'<table_cell[^>]*>(.*?)</table_cell>', row, flags=re.DOTALL | re.IGNORECASE)
            if not cells:
                cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, flags=re.DOTALL | re.IGNORECASE)
            cell_texts = []
            for c in cells:
                c = re.sub(r'<loc_\d+>', '', c)
                c = re.sub(r'<[^>]+>', '', c).strip()
                cell_texts.append(c)
            md_rows.append("| " + " | ".join(cell_texts) + " |")
        if not md_rows:
            return ""
        header = md_rows[0]
        sep = "| " + " | ".join("---" for _ in range(header.count("|") - 1)) + " |"
        lines = [header, sep] + md_rows[1:]
        return "\n".join(lines) + "\n"

    def _strip_reasoning(self, content: str) -> str:
        lines = content.split("\n")
        cleaned = []
        found_content = False
        for line in lines:
            lower = line.strip().lower()
            if not found_content:
                if any(lower.startswith(p) for p in self._REASONING_PREFIXES):
                    continue
                if not line.strip():
                    continue
                found_content = True
            cleaned.append(line)
        return "\n".join(cleaned) if cleaned else content

    def _extract_html(self, content: str) -> str:
        start = re.search(r'<[a-zA-Z]', content)
        if not start:
            return ""
        html_part = content[start.start():]
        html_part = re.sub(r'\n\s*\[.*?(\[.*?\])\s*\]\s*$', '', html_part, flags=re.DOTALL)
        html_part = re.sub(r'\n\s*\{.*\}\s*$', '', html_part, flags=re.DOTALL)
        html_part = re.sub(r'\n\s*\[', '', html_part)
        return html_part.strip()

    def _strip_metadata_prefix(self, content: str) -> str:
        content = re.sub(
            r'^\[\s*\{[^}]*"label"[^}]*"bbox"[^}]*\}\s*(,\s*\{[^}]*"label"[^}]*"bbox"[^}]*\})*\s*\]\s*\n*',
            '', content, flags=re.DOTALL
        )
        return content

    def _strip_trailing_commentary(self, content: str) -> str:
        matches = list(re.finditer(r'</([a-zA-Z]+)>', content))
        if not matches:
            return content
        content = content[:matches[-1].end()]
        return content.strip()

    @staticmethod
    def _strip_det_format(content: str) -> str:
        content = re.sub(r'<\|det\|>[^<]*?(<\|/det\|>)?', '', content)
        content = re.sub(r'<\|/det\|>', '', content)
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content.strip()

    @staticmethod
    def _strip_repetition(content: str, max_repeat: int = 5, min_ngram: int = 10) -> str:
        lines = content.split("\n")
        exact: dict[str, int] = {}
        prefix: dict[str, int] = {}
        cutoff = len(lines)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) < min_ngram:
                continue
            exact[stripped] = exact.get(stripped, 0) + 1
            if exact[stripped] > max_repeat:
                cutoff = i
                break
            pref = stripped[:40]
            prefix[pref] = prefix.get(pref, 0) + 1
            if prefix[pref] > max_repeat * 2:
                cutoff = i
                break
        return "\n".join(lines[:cutoff]).strip() if cutoff > 0 else content

    def _clean_output(self, content: str, fmt: str = "markdown") -> str:
        content = self._strip_reasoning(content)
        content = self._strip_det_format(content)
        content = self._strip_metadata_prefix(content)
        content = self._strip_repetition(content)
        if self._is_doctags_content(content):
            content = self._doctags_to_markdown(content)
        if fmt == "markdown" and ("<table" in content[:1000] or self._output_is_html or self._is_html_content(content)):
            if not self._use_image_only:
                content = self._strip_trailing_commentary(content)
            content = self._html_to_markdown(content)
        if fmt == "html":
            content = self._extract_html(content)
        return content

    def _append_saida(self, fmt: str, content: str):
        if fmt == "json":
            return
        content = self._clean_output(content, fmt)
        if fmt == "html" and not content:
            logger.warning("Model returned no HTML content for format=html")
            return
        path = self._output_path(fmt)
        try:
            with path.open("a", encoding="utf-8") as f:
                if path.stat().st_size > 0:
                    f.write("\n\n")
                f.write(content)
        except OSError as e:
            logger.error("Failed to write to %s: %s", path, e)
            raise
        logger.info("Incremental %s.%s updated (page appended)", self.basename, fmt)

    def _export_document(self, doc: Document, fmt: str):
        if fmt == "markdown":
            content = doc.to_markdown()
        elif fmt == "html":
            content = doc.to_html()
        elif fmt == "json":
            content = doc.to_json()
        else:
            content = doc.to_markdown()

        if fmt == "markdown" and ("<table" in content[:2000] or "<tr" in content[:2000] or "<td" in content[:2000]):
            content = self._html_to_markdown(content)

        path = self._output_path(fmt)
        path.write_text(content, encoding="utf-8")
        logger.info("Saved %s (%d chars)", path, len(content))

    def _consolidate_json(self):
        if "json" not in self.formats:
            return
        parts = []
        for partial in sorted(self.output_dir.glob("page_*.json.partial")):
            parts.append(partial.read_text(encoding="utf-8"))
        if not parts:
            return
        combined = "[\n" + ",\n".join(parts) + "\n]"
        self._output_path("json").write_text(combined, encoding="utf-8")
        logger.info("Saved %s", self._output_path("json"))

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return tokenize(text)

    @staticmethod
    def _extract_numbers(text: str) -> set[str]:
        return extract_numbers(text)

    @staticmethod
    def _extract_headings(text: str) -> list[str]:
        return re.findall(r"^#{1,6}\s.+", text, re.MULTILINE)

    def _content_overlap(self, native: str, output: str) -> dict:
        native_words = self._tokenize(native)
        output_words = self._tokenize(output)
        shared = native_words & output_words
        native_nums = self._extract_numbers(native)
        output_nums = self._extract_numbers(output)
        num_shared = native_nums & output_nums
        word_recall = (len(shared) / max(len(native_words), 1)) * 100
        num_recall = (len(num_shared) / max(len(native_nums), 1)) * 100
        out_headings = self._extract_headings(output)
        return {
            "word_recall_pct": round(word_recall, 1),
            "num_recall_pct": round(num_recall, 1),
            "native_words": len(native_words),
            "output_words": len(output_words),
            "shared_words": len(shared),
            "native_numbers": len(native_nums),
            "output_numbers": len(output_nums),
            "shared_numbers": len(num_shared),
            "output_headings": len(out_headings),
        }

    def process_page(self, page: PDFPage, fmt: str) -> str:
        if self.router is not None:
            result = self.router.process_with_fallback(page, fmt=fmt)
            return self._clean_output(result.text, fmt)

        prompt_fmt = self.model_config.get("default_prompt_format", "html")
        actual_fmt = prompt_fmt if fmt == "markdown" else fmt
        prompt = self.prompts[actual_fmt]
        system_prompt = "" if not self._use_system_prompt else None

        if self.mode == "text-first" and page.native_text and self._use_native_ref:
            try:
                enhanced_prompt = (
                    f"{prompt}\n\n"
                    f"As reference, here is the raw extracted text from this page "
                    f"(may have OCR errors or wrong order - use the image as source of truth):\n"
                    f"---\n{page.native_text}\n---"
                )
                content = self.client.ocr_image(enhanced_prompt, page.image, system_prompt=system_prompt)
            except (LMStudioClientError, Exception):
                logger.warning("LM Studio indisponivel — usando texto nativo puro para pagina %d", page.page_num)
                content = page.native_text
        elif self._use_image_only:
            content = self.client.ocr_image("", page.image, system_prompt=system_prompt)
        else:
            content = self.client.ocr_image(prompt, page.image, system_prompt=system_prompt)

        return self._clean_output(content, fmt)

    def _process_batch(self, pages: list[PDFPage], fmt: str) -> str:
        """Processa múltiplas páginas em batch usando o prompt correto."""
        images = [p.image for p in pages]
        
        # Usar o prompt correto baseado no formato
        prompt_fmt = self.model_config.get("default_prompt_format", "html")
        actual_fmt = prompt_fmt if fmt == "markdown" else fmt
        prompt = self.prompts[actual_fmt]
        
        # System prompt baseado na config do modelo
        system_prompt = "" if not self._use_system_prompt else None
        
        raw = self.client.ocr_images(prompt, images, system_prompt=system_prompt)
        return self._clean_output(raw, fmt)

    def _prepare_outputs(self):
        for fmt in self.formats:
            path = self._output_path(fmt)
            if fmt == "json":
                path.write_text("[\n", encoding="utf-8")
            else:
                path.write_text("", encoding="utf-8")

    def run(self, pdf_path: Path) -> list[PageResult]:
        import fitz
        fitz_doc = fitz.open(str(pdf_path))
        total = len(fitz_doc)
        logger.info("Processing %d pages with mode=%s, formats=%s", total, self.mode, self.formats)

        if not self.resume:
            self._prepare_outputs()

        doc = self._build_document_from_layout(fitz_doc)
        doc.filename = pdf_path.name
        self._document = doc

        results = self._enrich_document_with_ocr(doc, fitz_doc)
        fitz_doc.close()

        threshold = self.hybrid_config.quality_threshold_accept if self.hybrid_config else 0.70
        self._score_and_mark(doc, threshold)

        self._save_document_model(doc)
        self._save_metadata(results)

        for fmt in self.formats:
            self._export_document(doc, fmt)

        self._verify_completeness_after_run(doc, results, total)

        ok_count = sum(1 for r in results if r.status in ("ok", "resumed"))
        lmstudio_needed = sum(1 for r in results if r.status == "error: lm_studio_needed")
        all_ok = ok_count == total

        if lmstudio_needed > 0:
            print(
                f"\n{'='*60}\n"
                f"  ATENCAO: {lmstudio_needed} pagina(s) precisam de IA (GLM-OCR)\n"
                f"  mas o LM Studio nao estava rodando.\n"
                f"\n"
                f"  Para processar essas paginas com qualidade:\n"
                f"  1. Abra o LM Studio\n"
                f"  2. Carregue o modelo de OCR (ex: glm-ocr)\n"
                f"  3. Inicie o servidor em http://localhost:1234/v1\n"
                f"  4. Execute novamente com --ocr-mode hybrid --resume\n"
                f"\n"
                f"  As demais paginas foram processadas com OCR classico\n"
                f"  e estao disponiveis no arquivo de saida.\n"
                f"{'='*60}\n",
                flush=True,
            )

        if all_ok:
            for partial in self.output_dir.glob("page_*.partial"):
                partial.unlink()
            metadata = self._metadata_path()
            if metadata.exists():
                metadata.unlink()
            logger.info("Arquivos temporarios limpos")

        logger.info("Pipeline finished: %d/%d pages OK", ok_count, total)
        return results

    def _verify_completeness_after_run(self, doc: Document, results: list[PageResult], total: int):
        fmt = self.formats[0]
        path = self._output_path(fmt)

        if doc and doc.items:
            total_items = len(doc.items)
            low_items = sum(1 for it in doc.items.values() if it.confidence < 0.70)
            avg_conf = doc.overall_confidence()
            logger.info("Document: %d items, avg confidence=%.4f, %d low-confidence items", total_items, avg_conf, low_items)
            if avg_conf >= 0.80 and low_items == 0:
                logger.info("Fidelity: PASS (all checks OK)")
            else:
                logger.warning("Fidelity: PARTIAL - avg confidence=%.4f, %d low items", avg_conf, low_items)
            return

        ok_results = [r for r in results if r.status in ("ok", "resumed")]
        if not ok_results:
            return

        if not path.exists() or not path.stat().st_size:
            logger.warning("No output files found")
            return

        combined_native = "\n".join(
            doc.pages[r.page_num - 1].native_text
            for r in ok_results
            if r.page_num - 1 < len(doc.pages)
        )
        if not combined_native.strip():
            logger.info("No native text available for content comparison")
            return

        stats = self._content_overlap(combined_native, path.read_text(encoding="utf-8"))
        scores = []
        if stats["native_words"] > 0:
            scores.append(("vocabulary recall", stats["word_recall_pct"], 80.0))
        if stats["native_numbers"] > 0:
            scores.append(("number preservation", stats["num_recall_pct"], 90.0))

        all_ok = True
        for label, value, threshold in scores:
            status = "OK" if value >= threshold else "LOW"
            if value < threshold:
                all_ok = False
            logger.info("  %s: %.1f%% (%s, threshold: %.0f%%)", label, value, status, threshold)

        if all_ok:
            logger.info("Fidelity: PASS (all checks OK)")
        else:
            issues = []
            for label, value, threshold in scores:
                if value < threshold:
                    issues.append(f"{label} {value:.0f}% < {threshold:.0f}%")
            logger.warning("Fidelity: PARTIAL - %s", "; ".join(issues))

    def run_with_document(self, pdf_path: Path) -> tuple[list[PageResult], Document]:
        results = self.run(pdf_path)
        return results, self._document
