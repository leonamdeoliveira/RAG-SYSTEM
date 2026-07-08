"""Configuração centralizada — Settings via pydantic-settings.

Tudo é local-first e pluggável por env vars (prefixo RAG_) ou arquivo .env.
Defaults alinhados ao ARCHITECTURE.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # fallback mínimo se pydantic-settings ausente
    class BaseSettings:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        model_config = {}

    def SettingsConfigDict(**kw):
        return kw


class Settings(BaseSettings):
    # --- paths ---
    data_dir: Path = Path("data")  # diretorio com .md para o RAG (recebe overrides de markdown_dir)
    index_dir: Path = Path("index")
    zvec_path: Path = Path("index/zvec_collection")

    # --- chunker ---
    chunker_target_tokens: int = 600
    chunker_min_tokens: int = 80
    chunker_max_tokens: int = 800
    chunker_overlap_ratio: float = 0.15

    # --- embeddings ---
    embedding_provider: str = "local"  # local | dummy
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    embedding_device: str = "cpu"
    embedding_use_fp16: bool = False
    embedding_enable_sparse: bool = True
    embedding_batch_size: int = 32
    embedding_backend: str = "onnx"  # onnx | torch
    embedding_model_onnx: str = "gpahal/bge-m3-onnx-int8"
    embedding_onnx_device: str = "auto"  # auto | cpu | dml (DirectML para GPU AMD/Intel/NVIDIA)

    # --- zvec store ---
    zvec_enable_sparse: bool = True  # schema com sparse_embedding

    # --- retrieval ---
    retrieval_top_k: int = 20
    retrieval_score_threshold: float = 0.0
    retrieval_max_context_chunks: int = 8
    retrieval_max_per_doc: Optional[int] = 3
    retrieval_mode: str = "hybrid"  # dense | fts | hybrid | sparse
    retrieval_rerank: bool = False  # desativado — chat IA faz reranking

    # --- reranker (cross-encoder; desativado por padrao — chat IA faz o reranking) ---
    rerank_enabled: bool = False
    rerank_model: str = "suhaan7988/bge-reranker-v2-m3-int8-onnx"
    rerank_device: str = "cpu"
    rerank_max_length: int = 1024
    rerank_top_n_candidates: int = 20  # quantos candidatos entram no reranker

    # --- generation / LLM ---
    llm_base_url: str = "http://localhost:1234/v1"
    llm_api_key: str = ""
    llm_model: str = "local"
    llm_timeout: float = 60.0
    llm_temperature: float = 0.2
    llm_max_tokens: int = 700
    answer_mode: str = "answer"  # answer | answer_with_citations | extractive_summary | study_mode
    prompt_max_chars_per_chunk: int = 1600  # limite de chars por chunk no prompt do LLM

    # --- reliability ---
    reliability_low_confidence_threshold: float = 0.3
    reliability_expand_factor: int = 2
    reliability_max_retries: int = 1

    # --- logging ---
    log_level: str = "INFO"

    try:
        model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")
    except Exception:
        model_config = {}

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


def get_settings(**overrides) -> Settings:
    """Factory com overrides opcionais (util em testes)."""
    s = Settings()
    path_fields = {"data_dir", "index_dir", "zvec_path"}
    for k, v in overrides.items():
        if hasattr(s, k):
            if k in path_fields and isinstance(v, str):
                v = Path(v)
            setattr(s, k, v)
    return s
