"""Provider local: BGE-M3 (dense 1024d + sparse) com backends torch e onnx.

Backends:
  - "torch" (padrao): FlagEmbedding (dense+sparse) com fallback sentence-transformers
  - "onnx" : gpahal/bge-m3-onnx-int8 via optimum.onnxruntime, saida dense+sparse

Modelo: BAAI/bge-m3 (dense 1024d + sparse lexical)
Normalizacao L2 + metrica COSINE no Zvec: ver decisao D5.
"""

from __future__ import annotations

import multiprocessing
import threading
from typing import Optional

import numpy as np

from pipeline.rag.embeddings.base import (
    EmbeddingResult,
    SparseVector,
    l2_normalize,
    validate_dimension,
)
from pipeline.rag.utils.logging import get_logger

log = get_logger("app.embeddings.local")

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_MODEL_ONNX = "gpahal/bge-m3-onnx-int8"
DEFAULT_DIMENSION = 1024


def _detect_dml() -> bool:
    """Detecta se onnxruntime tem suporte a DirectML (GPU AMD/Intel/NVIDIA no Windows)."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return "DmlExecutionProvider" in providers
    except (ImportError, RuntimeError):
        return False


def _detect_cpu_features() -> str:
    """Detecta suporte a AVX-512 vs AVX2. Retorna 'avx512', 'avx2', ou 'unknown'."""
    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
        flags = set(info.get("flags", []))
        if "avx512f" in flags:
            return "avx512"
        if "avx2" in flags:
            return "avx2"
    except ImportError:
        pass
    try:
        import platform

        if platform.system() == "Windows":
            import subprocess

            r = subprocess.run(
                ["powershell", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=5,
            )
            name = r.stdout.lower()
            if "avx512" in name:
                return "avx512"
            if "avx2" in name:
                return "avx2"
    except (OSError, subprocess.TimeoutExpired) as e:
        log.debug("Erro ao detectar CPU features via PowerShell: %s", e)
        pass
    try:
        import platform

        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                flags_line = [l for l in f if l.startswith("flags")]
            if flags_line:
                flags = flags_line[0].lower()
                if "avx512f" in flags:
                    return "avx512"
                if "avx2" in flags:
                    return "avx2"
    except (IOError, OSError) as e:
        log.debug("Erro ao ler /proc/cpuinfo: %s", e)
    return "unknown"


class BGEM3LocalProvider:
    """Provider local BGE-M3. Suporta backends torch e onnx."""

    name: str = "bge-m3-local"
    dimension: int
    supports_sparse: bool

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        dimension: int = DEFAULT_DIMENSION,
        device: str = "cpu",
        use_fp16: bool = False,
        enable_sparse: bool = True,
        backend: str = "torch",
        model_name_onnx: str = DEFAULT_MODEL_ONNX,
        onnx_device: str = "auto",
    ) -> None:
        self._model_name = model_name
        self._model_name_onnx = model_name_onnx
        self._device = device
        self._use_fp16 = use_fp16
        self._enable_sparse = enable_sparse
        self._backend = backend
        self._onnx_device = onnx_device
        self.dimension = dimension
        self._model = None
        self._onnx_tokenizer = None
        self._onnx_output_names: Optional[list[str]] = None
        self._internal_backend: Optional[str] = None
        self._model_lock = threading.Lock()
        self.supports_sparse = enable_sparse  # Inicializar com o valor configurado

    # ---------------------------------------------------------------- init

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return
            if self._backend == "onnx":
                self._ensure_onnx()
                if self._internal_backend is not None:
                    return
                log.warning("fallback onnx -> torch devido a erro de carregamento")
            self._ensure_torch()

    def _ensure_onnx(self) -> None:
        cpu = _detect_cpu_features()
        if cpu == "avx2" and cpu != "avx512":
            log.warning(
                "CPU detectada com AVX2 (sem AVX-512). "
                "Modelos ONNX INT8 podem ter perda de precisao. "
                "Considere usar RAG_EMBEDDING_BACKEND=torch."
            )
        try:
            from tokenizers import Tokenizer

            self._onnx_tokenizer = Tokenizer.from_pretrained("BAAI/bge-m3")
            self._onnx_tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
            self._onnx_tokenizer.enable_truncation(max_length=8192)
            self._cls_id = 0
            self._eos_id = 2
            self._pad_id = 0
            self._unk_id = 100
        except ImportError:
            log.warning("tokenizers nao disponivel para ONNX. Instale: pip install tokenizers")
            return
        except Exception as e:
            log.warning("falha ao carregar tokenizer ONNX (%s) -> fallback torch", e)
            return

        try:
            model_path = _resolve_onnx_model_path(self._model_name_onnx)
            if model_path is None:
                log.warning("modelo ONNX nao encontrado no cache -> fallback torch")
                self._onnx_tokenizer = None
                return
            import onnxruntime as ort

            cpu_count = multiprocessing.cpu_count()
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = max(2, cpu_count // 2)
            so.inter_op_num_threads = 1
            so.enable_mem_pattern = True

            device = self._onnx_device
            if device == "dml" or (device == "auto" and _detect_dml()):
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
                log.info("ONNX usando DirectML (GPU AMD/Intel/NVIDIA)")
            else:
                providers = ["CPUExecutionProvider"]
                if device == "dml":
                    log.warning("DirectML nao disponivel -> fallback CPU")
                else:
                    log.info("ONNX usando CPU (threads=%d)", max(2, cpu_count // 2))
            self._model = ort.InferenceSession(
                model_path, sess_options=so, providers=providers
            )
            output_names = [o.name for o in self._model.get_outputs()]
            log.info("ONNX outputs: %s", output_names)
            self._onnx_output_names = output_names

            self._warmup_onnx()
        except ImportError:
            log.warning(
                "onnxruntime nao disponivel -> fallback torch. "
                "Instale com: pip install onnxruntime"
            )
            self._onnx_tokenizer = None
            return
        except Exception as e:
            log.warning("falha ao carregar modelo ONNX '%s' (%s) -> fallback torch", self._model_name_onnx, e)
            self._onnx_tokenizer = None
            return

        self._internal_backend = "onnx"
        self.supports_sparse = self._enable_sparse
        log.info("BGE-M3 carregado via ONNX (quantizado int8, threads=%d)", max(2, cpu_count // 2))

    def _warmup_onnx(self) -> None:
        """Warmup sincrono que tambem valida o provider e dispara fallback DML->CPU se necessario."""
        if self._onnx_tokenizer is None or self._model is None:
            return
        try:
            self._embed_onnx(["warmup"], batch_size=1)
            log.debug("Warmup ONNX concluido")
        except Exception as e:
            log.warning("Warmup ONNX falhou: %s", e)

    def _warmup_torch(self) -> None:
        if self._model is None:
            return
        try:
            self._model.encode(["warmup"], batch_size=1, return_dense=True, return_sparse=False, return_colbert=False)
        except (AttributeError, TypeError) as e:
            log.debug("Warmup FlagEmbedding falhou: %s", e)

    def _warmup_st(self) -> None:
        if self._model is None:
            return
        try:
            self._model.encode(["warmup"], batch_size=1, normalize_embeddings=True, convert_to_numpy=True)
        except (AttributeError, TypeError) as e:
            log.debug("Warmup SentenceTransformers falhou: %s", e)

    def _ensure_torch(self) -> None:
        if self._model is not None:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                self._model_name,
                use_fp16=self._use_fp16,
                device=self._device,
            )
            self._internal_backend = "flagembedding"
            self.supports_sparse = self._enable_sparse
            log.info("BGEM3 carregado via FlagEmbedding (dense+sparse, device=%s)", self._device)
            self._warmup_torch()
            return
        except ImportError:
            pass
        except Exception as e:
            log.warning("FlagEmbedding falhou (%s) -> fallback sentence-transformers", e)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "Nenhum backend de embedding encontrado. Instale um dos:\n"
                "  pip install FlagEmbedding        (dense+sparse, recomendado)\n"
                "  pip install sentence-transformers (dense-only, fallback)"
            ) from e
        self._model = SentenceTransformer(self._model_name, device=self._device)
        self._internal_backend = "sentence_transformers"
        self.supports_sparse = False
        log.info("BGEM3 carregado via sentence-transformers (dense-only, device=%s)", self._device)
        self._warmup_st()

    # ---------------------------------------------------------------- public

    def embed(self, texts: list[str], batch_size: int = 32) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(dense=np.zeros((0, self.dimension), dtype=np.float32))
        self._ensure_model()
        assert self._internal_backend is not None

        if self._internal_backend == "onnx":
            return self._embed_onnx(texts, batch_size)
        if self._internal_backend == "flagembedding":
            return self._embed_flag(texts, batch_size)
        return self._embed_st(texts, batch_size)

    def embed_query(self, text: str) -> EmbeddingResult:
        return self.embed([text])

    def embed_query_dense(self, text: str) -> np.ndarray:
        res = self.embed([text])
        return res.dense[0]

    # ---------------------------------------------------------------- torch backends

    def _embed_flag(self, texts: list[str], batch_size: int) -> EmbeddingResult:
        out = self._model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=self._enable_sparse,
            return_colbert=False,
        )
        dense = np.asarray(out["dense_vecs"], dtype=np.float32)
        if dense.ndim == 1:
            dense = dense.reshape(1, -1)
        dense = l2_normalize(dense)
        validate_dimension(dense, self.dimension)

        sparse: Optional[list[SparseVector]] = None
        if self._enable_sparse and "lexical_weights" in out:
            sparse = [
                SparseVector(values={int(k): float(v) for k, v in d.items()})
                for d in out["lexical_weights"]
            ]
        return EmbeddingResult(dense=dense, sparse=sparse)

    def _embed_st(self, texts: list[str], batch_size: int) -> EmbeddingResult:
        dense = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        dense = np.asarray(dense, dtype=np.float32)
        if dense.ndim == 1:
            dense = dense.reshape(1, -1)
        dense = l2_normalize(dense)
        validate_dimension(dense, self.dimension)
        return EmbeddingResult(dense=dense, sparse=None)

    # ---------------------------------------------------------------- onnx backend

    def _run_onnx_inference(self, output_names: list[str], feed_dict: dict) -> list[np.ndarray]:
        """Roda inferencia ONNX com fallback automatico DML -> CPU se falhar."""
        try:
            return self._model.run(output_names, feed_dict)
        except (RuntimeError, ValueError) as e:
            providers = getattr(self._model, "get_providers", lambda: [])()
            if "DmlExecutionProvider" not in providers:
                raise
            log.warning("DirectML falhou na inferencia (%s). Recriando sessao com CPU...", e)
            try:
                import onnxruntime as ort
                model_path = _resolve_onnx_model_path(self._model_name_onnx)
                if model_path is None:
                    raise RuntimeError("modelo ONNX nao encontrado para recriar sessao CPU")
                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                so.intra_op_num_threads = max(2, multiprocessing.cpu_count() // 2)
                so.inter_op_num_threads = 1
                so.enable_mem_pattern = True
                self._model = ort.InferenceSession(
                    model_path, sess_options=so, providers=["CPUExecutionProvider"]
                )
                log.info("ONNX recriado com CPU apos falha do DirectML")
                return self._model.run(output_names, feed_dict)
            except Exception as e2:
                log.error("Falha ao recriar sessao ONNX com CPU: %s", e2)
                raise RuntimeError(
                    f"DirectML falhou e fallback CPU tambem falhou. "
                    f"Use RAG_EMBEDDING_ONNX_DEVICE=cpu ou RAG_EMBEDDING_BACKEND=torch. "
                    f"Erro original: {e}"
                ) from e2

    def _embed_onnx(self, texts: list[str], batch_size: int = 16) -> EmbeddingResult:
        if self._onnx_tokenizer is None or self._model is None:
            raise RuntimeError("ONNX model/tokenizer not loaded")

        output_names = self._onnx_output_names or ["dense_vecs", "sparse_vecs"]
        all_dense: list[np.ndarray] = []
        all_sparse: list[Optional[SparseVector]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encodings = [self._onnx_tokenizer.encode(t) for t in batch]
            max_len = max(len(e.ids) for e in encodings)
            input_ids = np.array([e.ids + [0] * (max_len - len(e.ids)) for e in encodings], dtype=np.int64)
            attn_mask = np.array([e.attention_mask + [0] * (max_len - len(e.attention_mask)) for e in encodings], dtype=np.int64)

            outputs = self._run_onnx_inference(
                output_names,
                {"input_ids": input_ids, "attention_mask": attn_mask},
            )
            output_map = dict(zip(output_names, outputs))

            dense = np.asarray(output_map["dense_vecs"], dtype=np.float32)
            if dense.ndim == 3:
                dense = dense[:, 0, :]
            dense = l2_normalize(dense)
            validate_dimension(dense, self.dimension)
            all_dense.append(dense)

            if self._enable_sparse and "sparse_vecs" in output_map:
                token_weights = np.asarray(output_map["sparse_vecs"], dtype=np.float32)
                if token_weights.ndim == 3:
                    token_weights = token_weights.squeeze(-1)
                for j in range(len(batch)):
                    sd = _process_token_weights(
                        token_weights[j], encodings[j].ids,
                        self._cls_id, self._eos_id, self._pad_id, self._unk_id,
                    )
                    all_sparse.append(SparseVector(values=sd))

        dense = np.concatenate(all_dense, axis=0) if all_dense else np.zeros((0, self.dimension), dtype=np.float32)
        sparse = all_sparse if self._enable_sparse else None
        return EmbeddingResult(dense=dense, sparse=sparse)


# ---------------------------------------------------------------- helpers


def _process_token_weights(
    token_weights: np.ndarray,
    input_ids: list,
    cls_id: int,
    eos_id: int,
    pad_id: int,
    unk_id: int,
) -> dict[int, float]:
    unused = {cls_id, eos_id, pad_id, unk_id}
    result: dict[int, float] = {}
    for w, idx in zip(token_weights, input_ids):
        idx_int = int(idx)
        w_float = float(w)
        if idx_int not in unused and w_float > 0:
            if w_float > result.get(idx_int, 0.0):
                result[idx_int] = w_float
    return result


def _resolve_onnx_model_path(model_name: str) -> Optional[str]:
    """Encontra o caminho local do modelo ONNX no cache do HuggingFace."""
    from pathlib import Path
    import os

    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(model_name, "model_quantized.onnx")
    except ImportError:
        log.debug("huggingface_hub não instalado, tentando cache local")
        pass
    except (OSError, ValueError) as e:
        log.debug("Erro ao baixar modelo ONNX do HF Hub: %s", e)
        pass

    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
    model_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    if not model_dir.exists():
        return None
    for onnx_file in model_dir.rglob("model_quantized.onnx"):
        return str(onnx_file)
    return None
