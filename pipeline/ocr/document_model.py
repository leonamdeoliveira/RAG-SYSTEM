from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Optional


def _uid() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def to_dict(self) -> dict:
        return {"x0": round(self.x0, 1), "y0": round(self.y0, 1),
                "x1": round(self.x1, 1), "y1": round(self.y1, 1)}


@dataclass
class DocItem:
    id: str = field(default_factory=_uid)
    level: int = 0
    label: str = "text"
    bbox: Optional[BoundingBox] = None
    page_num: int = 1
    confidence: float = 1.0
    children: list[str] = field(default_factory=list)
    parent: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "level": self.level,
            "label": self.label,
            "page": self.page_num,
            "confidence": round(self.confidence, 4),
            "children": self.children,
            "parent": self.parent,
        }
        if self.bbox:
            d["bbox"] = self.bbox.to_dict()
        return d


@dataclass
class TextItem(DocItem):
    text: str = ""
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    is_bold: bool = False

    def __post_init__(self):
        self.label = "text"

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["text"] = self.text
        if self.font_name:
            d["font_name"] = self.font_name
        if self.font_size:
            d["font_size"] = round(self.font_size, 1)
        if self.is_bold:
            d["is_bold"] = True
        return d


@dataclass
class HeadingItem(TextItem):
    heading_level: int = 1

    def __post_init__(self):
        self.label = "heading"


@dataclass
class TableItem(DocItem):
    rows: list[list[str]] = field(default_factory=list)
    headers: list[str] = field(default_factory=list)
    caption: str = ""

    def __post_init__(self):
        self.label = "table"

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["rows"] = self.rows
        d["headers"] = self.headers
        if self.caption:
            d["caption"] = self.caption
        return d

    def to_markdown(self) -> str:
        if not self.rows:
            return ""
        lines = []
        if self.caption:
            lines.append(f"**{self.caption}**\n")
        if self.headers:
            lines.append("| " + " | ".join(str(h) for h in self.headers) + " |")
            lines.append("| " + " | ".join("---" for _ in self.headers) + " |")
        else:
            lines.append("| " + " | ".join("---" for _ in self.rows[0]) + " |")
        for row in self.rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)


@dataclass
class PictureItem(DocItem):
    image_path: Optional[str] = None
    caption: str = ""
    description: str = ""

    def __post_init__(self):
        self.label = "picture"

    def to_dict(self) -> dict:
        d = super().to_dict()
        if self.image_path:
            d["image_path"] = self.image_path
        if self.caption:
            d["caption"] = self.caption
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class GroupItem(DocItem):
    group_type: str = "generic"

    def __post_init__(self):
        self.label = "group"

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["group_type"] = self.group_type
        return d


@dataclass
class ListItem(GroupItem):
    ordered: bool = False

    def __post_init__(self):
        self.group_type = "list_ol" if self.ordered else "list_ul"


@dataclass
class SectionItem(GroupItem):
    title: str = ""

    def __post_init__(self):
        self.group_type = "section"


@dataclass
class Page:
    page_num: int
    width: int = 0
    height: int = 0
    items: list[DocItem] = field(default_factory=list)
    native_text: str = ""

    @property
    def item_count(self) -> int:
        return len(self.items)

    def to_dict(self) -> dict:
        return {
            "page": self.page_num,
            "width": self.width,
            "height": self.height,
            "items": [it.to_dict() for it in self.items],
            "native_chars": len(self.native_text),
        }


@dataclass
class Document:
    filename: str = ""
    pages: list[Page] = field(default_factory=list)
    items: dict[str, DocItem] = field(default_factory=dict)
    body: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_item(self, item: DocItem) -> str:
        self.items[item.id] = item
        return item.id

    def add_to_body(self, item_id: str):
        if item_id not in self.body:
            self.body.append(item_id)

    def get_items_by_page(self, page_num: int) -> list[DocItem]:
        return [it for it in self.items.values() if it.page_num == page_num]

    def get_text_items(self) -> list[TextItem]:
        return [it for it in self.items.values() if isinstance(it, TextItem)]

    def get_table_items(self) -> list[TableItem]:
        return [it for it in self.items.values() if isinstance(it, TableItem)]

    def get_picture_items(self) -> list[PictureItem]:
        return [it for it in self.items.values() if isinstance(it, PictureItem)]

    def get_body_items(self) -> list[DocItem]:
        return [self.items[rid] for rid in self.body if rid in self.items]

    def overall_confidence(self) -> float:
        confidences = [it.confidence for it in self.items.values()]
        return sum(confidences) / len(confidences) if confidences else 0.0

    def stats(self) -> dict:
        text_items = self.get_text_items()
        total_chars = sum(len(it.text) for it in text_items)
        return {
            "pages": len(self.pages),
            "total_items": len(self.items),
            "text_items": len(text_items),
            "table_items": len(self.get_table_items()),
            "picture_items": len(self.get_picture_items()),
            "total_chars": total_chars,
            "overall_confidence": round(self.overall_confidence(), 4),
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "pages": [p.to_dict() for p in self.pages],
            "items": {k: v.to_dict() for k, v in self.items.items()},
            "body": self.body,
            "stats": self.stats(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_markdown(self) -> str:
        lines = []
        for item_id in self.body:
            item = self.items.get(item_id)
            if item is None:
                continue
            if isinstance(item, HeadingItem):
                prefix = "#" * min(item.heading_level, 6)
                lines.append(f"\n{prefix} {item.text}\n")
            elif isinstance(item, TextItem):
                if item.text.strip():
                    lines.append(item.text)
            elif isinstance(item, TableItem):
                md = item.to_markdown()
                if md:
                    lines.append(f"\n{md}\n")
            elif isinstance(item, PictureItem):
                alt = item.caption or item.description or "Image"
                lines.append(f"\n[Imagem: {alt}]\n")
            elif isinstance(item, ListItem):
                for child_id in item.children:
                    child = self.items.get(child_id)
                    if isinstance(child, TextItem):
                        prefix = "1. " if item.ordered else "- "
                        lines.append(f"{prefix}{child.text}")
        return "\n".join(lines)

    def to_html(self) -> str:
        parts = ['<!DOCTYPE html><html><head><meta charset="utf-8">']
        if self.filename:
            parts.append(f'<title>{self.filename}</title>')
        parts.append('</head><body>')
        for item_id in self.body:
            item = self.items.get(item_id)
            if item is None:
                continue
            if isinstance(item, HeadingItem):
                h = min(item.heading_level, 6)
                parts.append(f'<h{h}>{item.text}</h{h}>')
            elif isinstance(item, TextItem):
                if item.text.strip():
                    parts.append(f'<p>{item.text}</p>')
            elif isinstance(item, TableItem):
                parts.append('<table>')
                if item.caption:
                    parts.append(f'<caption>{item.caption}</caption>')
                if item.headers:
                    parts.append('<thead><tr>')
                    for h in item.headers:
                        parts.append(f'<th>{h}</th>')
                    parts.append('</tr></thead>')
                parts.append('<tbody>')
                for row in item.rows:
                    parts.append('<tr>')
                    for cell in row:
                        parts.append(f'<td>{cell}</td>')
                    parts.append('</tr>')
                parts.append('</tbody></table>')
            elif isinstance(item, PictureItem):
                alt = item.caption or item.description or "Image"
                parts.append(f'<figure><figcaption>{alt}</figcaption></figure>')
            elif isinstance(item, ListItem):
                tag = 'ol' if item.ordered else 'ul'
                parts.append(f'<{tag}>')
                for child_id in item.children:
                    child = self.items.get(child_id)
                    if isinstance(child, TextItem):
                        parts.append(f'<li>{child.text}</li>')
                parts.append(f'</{tag}>')
        parts.append('</body></html>')
        return '\n'.join(parts)
