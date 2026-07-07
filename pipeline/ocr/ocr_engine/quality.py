import re
from dataclasses import dataclass, field
from typing import ClassVar, Optional

from pipeline.ocr.document_model import TextItem, TableItem


@dataclass
class QualityReport:
    score: float
    char_count: int
    alphanumeric_ratio: float
    symbol_ratio: float
    avg_line_length: float
    empty_line_ratio: float
    has_repeated_chars: bool
    engine_confidence: Optional[float]
    diacritic_ratio: float = 0.0

    def acceptable(self, threshold: float = 0.7) -> bool:
        return self.score >= threshold

    def retryable(self, threshold: float = 0.4) -> bool:
        return self.score >= threshold

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "char_count": self.char_count,
            "alphanumeric_ratio": round(self.alphanumeric_ratio, 4),
            "symbol_ratio": round(self.symbol_ratio, 4),
            "avg_line_length": round(self.avg_line_length, 2),
            "empty_line_ratio": round(self.empty_line_ratio, 4),
            "has_repeated_chars": self.has_repeated_chars,
            "engine_confidence": self.engine_confidence,
            "diacritic_ratio": round(self.diacritic_ratio, 4),
        }


@dataclass
class ItemQualityReport:
    item_id: str
    label: str
    page_num: int
    score: float
    char_count: int
    alphanumeric_ratio: float
    symbol_ratio: float
    avg_line_length: float
    engine_confidence: Optional[float]
    acceptable: bool

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "label": self.label,
            "page": self.page_num,
            "score": round(self.score, 4),
            "char_count": self.char_count,
            "alphanumeric_ratio": round(self.alphanumeric_ratio, 4),
            "symbol_ratio": round(self.symbol_ratio, 4),
            "avg_line_length": round(self.avg_line_length, 2),
            "engine_confidence": self.engine_confidence,
            "acceptable": self.acceptable,
        }


_CHAR_REPEAT_PATTERN = re.compile(r"(.)\1{7,}")


class QualityScorer:

    WEIGHTS: ClassVar[dict[str, float]] = {
        "alphanumeric_ratio": 0.35,
        "symbol_ratio": 0.20,
        "avg_line_length": 0.15,
        "empty_line_ratio": 0.10,
        "repeated_chars": 0.10,
        "engine_confidence": 0.10,
    }

    MIN_CHARS: ClassVar[int] = 20
    IDEAL_LINE_LENGTH: ClassVar[int] = 60
    MAX_SYMBOL_RATIO: ClassVar[float] = 0.10
    MAX_EMPTY_LINE_RATIO: ClassVar[float] = 0.50
    MIN_DIACRITIC_RATIO_PT: ClassVar[float] = 0.008
    MIN_TABLE_PIPE_RATIO: ClassVar[float] = 0.30
    MIN_TABLE_CONSISTENCY: ClassVar[float] = 0.70  # 0.8% — pt tem ~2-5%

    _PT_HINTS: ClassVar[tuple[str, ...]] = (
        " o ", " a ", " de ", " que ", " para ", " com ", " não ", " é ",
    )
    _DIACRITICS: ClassVar[set[str]] = {
        "á", "à", "ã", "â", "ä", "é", "ê", "ë", "í", "î", "ï",
        "ó", "ô", "õ", "ö", "ú", "û", "ü", "ç", "ñ",
    }

    @staticmethod
    def _is_portuguese(text: str) -> bool:
        s = " " + text[:2000].lower() + " "
        return any(h in s for h in QualityScorer._PT_HINTS)

    @staticmethod
    def _diacritic_ratio(text: str) -> float:
        letters = [c for c in text.lower() if c.isalpha()]
        if not letters:
            return 1.0
        diacritic_count = sum(1 for c in letters if c in QualityScorer._DIACRITICS)
        return diacritic_count / len(letters)

    @staticmethod
    def _looks_like_table(text: str) -> bool:
        """Detecta se o texto parece ser uma tabela (tem pipes | organizados)."""
        lines = [l for l in text.split("\n") if l.strip() and not l.strip().startswith("#")]
        if len(lines) < 3:
            return False
        pipe_lines = [l for l in lines if "|" in l]
        return len(pipe_lines) >= 2 and len(pipe_lines) / len(lines) >= QualityScorer.MIN_TABLE_PIPE_RATIO

    @staticmethod
    def _numeric_density(text: str) -> float:
        """Densidade de caracteres numericos + simbolos financeiros. Alto = provavel tabela."""
        chars = list(text)
        if not chars:
            return 0.0
        numeric = sum(1 for c in chars if c.isdigit() or c in ".%,;()[]")
        return numeric / len(chars)

    @staticmethod
    def _table_quality(text: str) -> float:
        """Score de qualidade para tabela. 1.0 = tabela bem formatada, 0.0 = falha."""
        lines = [l for l in text.split("\n") if l.strip()]
        pipe_lines = [l for l in lines if "|" in l]
        if not pipe_lines:
            return 0.0

        pipe_ratio = len(pipe_lines) / max(len(lines), 1)
        if pipe_ratio < QualityScorer.MIN_TABLE_PIPE_RATIO:
            return pipe_ratio / QualityScorer.MIN_TABLE_PIPE_RATIO

        col_counts = []
        for l in pipe_lines:
            cols = [c.strip() for c in l.strip("|").split("|")]
            col_counts.append(len(cols))
        if len(col_counts) < 3:
            return pipe_ratio

        mode_count = max(set(col_counts), key=col_counts.count)
        consistency = sum(1 for c in col_counts if c == mode_count) / len(col_counts)

        return pipe_ratio * 0.3 + consistency * 0.7

    def score(self, text: str, engine_confidence: Optional[float] = None, min_chars: int = 20) -> QualityReport:
        if not text or len(text.strip()) < min_chars:
            return QualityReport(
                score=0.0,
                char_count=len(text),
                alphanumeric_ratio=0.0,
                symbol_ratio=1.0,
                avg_line_length=0.0,
                empty_line_ratio=1.0,
                has_repeated_chars=False,
                engine_confidence=engine_confidence,
                diacritic_ratio=0.0,
            )

        lines = text.split("\n")
        non_empty = [l for l in lines if l.strip()]

        char_count = len(text)
        total_chars = max(len(text), 1)
        letter_digit_chars = sum(1 for c in text if c.isalnum() or c.isspace())
        alphanumeric_ratio = letter_digit_chars / total_chars

        symbol_chars = sum(1 for c in text if c in "|\\/<>{}[]~`^_=+@#$%&*!")
        symbol_ratio = symbol_chars / total_chars
        symbol_score = max(0.0, 1.0 - (symbol_ratio / self.MAX_SYMBOL_RATIO))

        line_lengths = [len(l) for l in non_empty] if non_empty else [0]
        avg_line_length = sum(line_lengths) / len(line_lengths)
        length_score = min(1.0, avg_line_length / self.IDEAL_LINE_LENGTH)

        empty_line_ratio = (len(lines) - len(non_empty)) / max(len(lines), 1)
        empty_score = max(0.0, 1.0 - (empty_line_ratio / self.MAX_EMPTY_LINE_RATIO))

        has_repeated = bool(_CHAR_REPEAT_PATTERN.search(text))
        repeat_score = 0.0 if has_repeated else 1.0

        conf_score = engine_confidence if engine_confidence is not None else 1.0

        score = (
            self.WEIGHTS["alphanumeric_ratio"] * alphanumeric_ratio
            + self.WEIGHTS["symbol_ratio"] * symbol_score
            + self.WEIGHTS["avg_line_length"] * length_score
            + self.WEIGHTS["empty_line_ratio"] * empty_score
            + self.WEIGHTS["repeated_chars"] * repeat_score
            + self.WEIGHTS["engine_confidence"] * conf_score
        )

        diacritic_ratio = self._diacritic_ratio(text)
        if self._is_portuguese(text) and diacritic_ratio < self.MIN_DIACRITIC_RATIO_PT:
            score *= 0.4 + 0.6 * (diacritic_ratio / self.MIN_DIACRITIC_RATIO_PT)

        if self._looks_like_table(text):
            table_q = self._table_quality(text)
            score *= 0.5 + 0.5 * table_q
        elif self._numeric_density(text) > 0.20:
            score *= 0.5

        return QualityReport(
            score=score,
            char_count=char_count,
            alphanumeric_ratio=alphanumeric_ratio,
            symbol_ratio=symbol_ratio,
            avg_line_length=avg_line_length,
            empty_line_ratio=empty_line_ratio,
            has_repeated_chars=has_repeated,
            engine_confidence=engine_confidence,
            diacritic_ratio=diacritic_ratio,
        )

    def score_item(self, item, threshold: float = 0.70) -> ItemQualityReport:
        if isinstance(item, TextItem):
            text = item.text
            label = "heading" if hasattr(item, "heading_level") else "text"
            page_num = item.page_num
            item_id = item.id
        elif isinstance(item, TableItem):
            text = "\n".join(
                " | ".join(str(c) for c in row)
                for row in item.rows
            )
            label = "table"
            page_num = item.page_num
            item_id = item.id
        else:
            return ItemQualityReport(
                item_id=getattr(item, "id", "unknown"),
                label=getattr(item, "label", "unknown"),
                page_num=getattr(item, "page_num", 0),
                score=1.0,
                char_count=0,
                alphanumeric_ratio=0.0,
                symbol_ratio=0.0,
                avg_line_length=0.0,
                engine_confidence=None,
                acceptable=True,
            )

        if label == "heading":
            item_min_chars = 5
        elif label == "table":
            item_min_chars = 10
        else:
            item_min_chars = 20
        report = self.score(text, engine_confidence=item.confidence, min_chars=item_min_chars)
        return ItemQualityReport(
            item_id=item_id,
            label=label,
            page_num=page_num,
            score=report.score,
            char_count=report.char_count,
            alphanumeric_ratio=report.alphanumeric_ratio,
            symbol_ratio=report.symbol_ratio,
            avg_line_length=report.avg_line_length,
            engine_confidence=report.engine_confidence,
            acceptable=report.acceptable(threshold),
        )

    def score_document_items(self, items: list, threshold: float = 0.70) -> list[ItemQualityReport]:
        reports = []
        for item in items:
            report = self.score_item(item, threshold)
            reports.append(report)
            item.confidence = report.score
        return reports
