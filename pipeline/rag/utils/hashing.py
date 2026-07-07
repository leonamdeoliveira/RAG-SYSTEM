from __future__ import annotations

import hashlib


def sha256_file(abs_path: str) -> str:
    h = hashlib.sha256()
    with open(abs_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stable_doc_id(rel_path: str) -> str:
    return hashlib.sha1(rel_path.replace("\\", "/").encode("utf-8")).hexdigest()[:16]
