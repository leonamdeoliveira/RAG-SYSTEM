from __future__ import annotations

from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=1)
def _tiktoken_encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Contagem aproximada de tokens. Usa tiktoken se disponível; fallback char/4."""
    enc = _tiktoken_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


def normalize_ws(text: str) -> str:
    """Colapsa espaços consecutivos (não newlines) para hashing estável de chunks."""
    out = []
    for line in text.splitlines():
        out.append(" ".join(line.split()))
    return "\n".join(out).strip()
