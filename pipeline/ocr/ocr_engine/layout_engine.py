import logging
from pathlib import Path
from typing import Optional

from pipeline.ocr.ocr_engine.base import OCREngineBase, EngineResult
from pipeline.ocr.pdf_utils import PDFPage

logger = logging.getLogger(__name__)


class LayoutEngine(OCREngineBase):

    def __init__(self, image_output_dir: Optional[Path] = None):
        self._image_output_dir = image_output_dir

    @property
    def name(self) -> str:
        return "layout"

    def is_available(self) -> bool:
        return True

    def supports(self, page: PDFPage) -> bool:
        return True

    def extract_text(self, page: PDFPage, **kwargs) -> EngineResult:
        raise NotImplementedError("LayoutEngine does single-page text extraction")

    def analyze_page(self, page_num: int, fz_page, pw: int, ph: int) -> tuple[list, list, list]:
        from pipeline.ocr.pdf_utils import extract_layout, extract_tables, extract_images
        try:
            items = extract_layout(fz_page, page_num)
        except Exception as e:
            logger.warning("Layout extraction failed page %d: %s", page_num, e)
            items = []
        try:
            tables = extract_tables(fz_page, page_num)
        except Exception as e:
            logger.warning("Table extraction failed page %d: %s", page_num, e)
            tables = []
        try:
            pics = extract_images(fz_page, page_num, self._image_output_dir)
        except Exception as e:
            logger.warning("Image extraction failed page %d: %s", page_num, e)
            pics = []
        return items, tables, pics
