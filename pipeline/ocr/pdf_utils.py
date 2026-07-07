import logging
from pathlib import Path
from typing import Optional

import fitz
from PIL import Image

from pipeline.ocr.document_model import (
    BoundingBox, DocItem, TextItem, HeadingItem,
    TableItem, PictureItem, ListItem, Page,
)

logger = logging.getLogger(__name__)


class PDFPage:
    def __init__(self, page_num: int, image: Image.Image, native_text: str = ""):
        self.page_num = page_num
        self.image = image
        self.native_text = native_text.strip()


def render_page(page: fitz.Page, dpi: int = 200) -> Image.Image:
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def extract_native_text(page: fitz.Page) -> str:
    return page.get_text("text", sort=True)


def has_meaningful_text(text: str, min_chars: int = 50) -> bool:
    stripped = text.strip()
    return len(stripped) >= min_chars


def load_pdf(
    path: Path, dpi: int = 200, mode: str = "text-first"
) -> list[PDFPage]:
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(str(path))
    try:
        pages: list[PDFPage] = []
        logger.info("PDF has %d pages", len(doc))

        for i in range(len(doc)):
            page = doc[i]
            page_num = i + 1

            native_text = extract_native_text(page)
            image = render_page(page, dpi=dpi)

            if mode == "text-first" and has_meaningful_text(native_text):
                logger.info("Page %d: using native text", page_num)
            else:
                logger.info("Page %d: will use OCR", page_num)

            pages.append(PDFPage(page_num=page_num, image=image, native_text=native_text))

        return pages
    finally:
        doc.close()


def _block_is_bold(block: dict) -> bool:
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            font_name = (span.get("font", "") or "").lower()
            flags = span.get("flags", 0)
            if "bold" in font_name or (flags & 2):
                return True
    return False


def _block_font_info(block: dict) -> tuple[Optional[str], Optional[float]]:
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            return span.get("font", ""), span.get("size")
    return None, None


def _block_text(block: dict) -> str:
    texts = []
    for line in block.get("lines", []):
        line_text = "".join(span.get("text", "") for span in line.get("spans", []))
        texts.append(line_text)
    return "\n".join(texts).strip()


def _block_bbox(page_num: int, block: dict) -> BoundingBox:
    b = block.get("bbox")
    if b and len(b) == 4:
        return BoundingBox(x0=b[0], y0=b[1], x1=b[2], y1=b[3])
    return BoundingBox(x0=0, y0=0, x1=0, y1=0)


def _is_list_item(text: str) -> bool:
    t = text.strip()
    return bool(
        t and (
            t[0] in "•-–—*▪▸▹▶►"
            or (len(t) > 2 and t[0].isdigit() and t[1] in ".):")
        )
    )


def _detect_heading(text: str, font_size: Optional[float], is_bold: bool) -> Optional[int]:
    if not text or not font_size:
        return None
    clean = text.strip()
    if not clean:
        return None
    if is_bold and font_size >= 16:
        return 1
    if is_bold and font_size >= 14:
        return 2
    if is_bold and font_size >= 12:
        return 3
    if is_bold:
        return 4
    if len(clean) < 100 and clean.endswith((".", ":", "?")) is False and font_size >= 12:
        for keyword in ("introdu", "conclu", "resumo", "sumári", "capítul",
                        "seção", "abstract", "introduction", "conclusion",
                        "summary", "chapter", "section", "referênc",
                        "bibliograf", "anexo", "apêndic"):
            if keyword in clean.lower():
                return 2 if font_size >= 14 else 3
    return None


def extract_layout(
    fitz_page: fitz.Page,
    page_num: int,
) -> list[DocItem]:
    items: list[DocItem] = []
    data = fitz_page.get_text("dict")
    blocks = data.get("blocks", [])

    font_sizes = []
    for b in blocks:
        if b.get("type") == 0:
            _, fs = _block_font_info(b)
            if fs:
                font_sizes.append(fs)
    median_fs = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 10.0

    last_list = None
    for block in blocks:
        if block.get("type") != 0:
            continue

        text = _block_text(block)
        if not text:
            continue

        bbox = _block_bbox(page_num, block)
        font_name, font_size = _block_font_info(block)
        is_bold = _block_is_bold(block)

        heading_level = _detect_heading(text, font_size, is_bold)
        if heading_level:
            item = HeadingItem(
                text=text,
                heading_level=heading_level,
                bbox=bbox,
                page_num=page_num,
                font_name=font_name,
                font_size=font_size,
                is_bold=is_bold,
            )
            items.append(item)
            last_list = None
            continue

        if _is_list_item(text):
            if last_list is None or not isinstance(last_list, ListItem):
                last_list = ListItem(bbox=bbox, page_num=page_num)
                items.append(last_list)
            list_text_item = TextItem(
                text=text,
                bbox=bbox,
                page_num=page_num,
                font_name=font_name,
                font_size=font_size,
                is_bold=is_bold,
            )
            lid = list_text_item.id
            items.append(list_text_item)
            last_list.children.append(lid)
            continue
        else:
            last_list = None

        item = TextItem(
            text=text,
            bbox=bbox,
            page_num=page_num,
            font_name=font_name,
            font_size=font_size,
            is_bold=is_bold,
        )
        items.append(item)

    return items


def extract_tables(
    fitz_page: fitz.Page,
    page_num: int,
) -> list[TableItem]:
    tables: list[TableItem] = []
    try:
        found = fitz_page.find_tables()
    except Exception:
        return tables

    if found is None:
        return tables

    for table in found.tables:
        try:
            data = table.extract()
            if not data:
                continue

            rows = []
            headers = []
            for ri, row in enumerate(data):
                clean_row = [(cell.strip() if cell else "") for cell in row]
                if any(cell for cell in clean_row):
                    rows.append(clean_row)

            if rows:
                headers = rows[0]
                data_rows = rows[1:] if len(rows) > 1 else []

                t_bbox = table.bbox or (0, 0, 0, 0)
                bbox = BoundingBox(
                    x0=t_bbox[0], y0=t_bbox[1],
                    x1=t_bbox[2], y1=t_bbox[3],
                )
                item = TableItem(
                    rows=data_rows or rows,
                    headers=headers if data_rows else [],
                    bbox=bbox,
                    page_num=page_num,
                )
                tables.append(item)
        except Exception:
            continue

    return tables


def extract_images(
    fitz_page: fitz.Page,
    page_num: int,
    output_dir: Optional[Path] = None,
) -> list[PictureItem]:
    pictures: list[PictureItem] = []
    image_list = fitz_page.get_images(full=True)
    if not image_list:
        return pictures

    for img_idx, img_info in enumerate(image_list):
        xref = img_info[0]
        try:
            base_image = fitz_page.parent.extract_image(xref)
            if not base_image:
                continue
            image_bytes = base_image.get("image")
            if not image_bytes:
                continue

            ext = base_image.get("ext", "png")
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)

            image_path = None
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                fname = f"page_{page_num:04d}_img_{img_idx:02d}.{ext}"
                image_path = str(output_dir / fname)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

            rects = fitz_page.get_image_rects(xref)
            if rects:
                r = fitz_page.get_image_bbox(rects[0])
                if r:
                    bbox = BoundingBox(x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1)
                else:
                    bbox = BoundingBox(x0=0, y0=0, x1=float(width), y1=float(height))
            else:
                bbox = BoundingBox(x0=0, y0=0, x1=float(width), y1=float(height))

            item = PictureItem(
                image_path=image_path,
                bbox=bbox,
                page_num=page_num,
            )
            pictures.append(item)
        except Exception:
            continue

    return pictures
