from __future__ import annotations

import numpy as np
from unittest.mock import MagicMock, patch

import pytest

from pipeline.rag.embeddings.base import EmbeddingResult, SparseVector
from pipeline.rag.embeddings.local_provider import BGEM3LocalProvider


class TestONNXBackend:
    """Valida o backend ONNX com mock do modelo e tokenizer."""

    def _make_mock_model(self):
        """Cria um mock de onnxruntime.InferenceSession."""
        model = MagicMock()
        dense = np.random.randn(1024).astype(np.float32)
        dense = dense / np.linalg.norm(dense)
        sparse_weights = np.array([[0.5, 0.3, 0.0, 0.8, 0.0]], dtype=np.float32)
        model.run.return_value = [dense.reshape(1, 1024), sparse_weights.reshape(1, 5, 1)]
        return model

    def _make_mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.cls_token_id = 0
        tokenizer.eos_token_id = 1
        tokenizer.pad_token_id = 2
        tokenizer.unk_token_id = 3
        
        # Criar um objeto de encoding mock
        encoding = MagicMock()
        encoding.ids = [0, 101, 102, 103, 1]
        encoding.attention_mask = [1, 1, 1, 1, 1]
        tokenizer.encode.return_value = encoding
        
        return tokenizer

    def test_onnx_produces_dense_and_sparse(self):
        provider = BGEM3LocalProvider(backend="torch")
        provider._backend = "onnx"
        provider._internal_backend = None
        provider._enable_sparse = True
        provider._model = self._make_mock_model()
        provider._onnx_tokenizer = self._make_mock_tokenizer()
        provider._cls_id = 0
        provider._eos_id = 1
        provider._pad_id = 2
        provider._unk_id = 3

        with patch.object(provider, "_ensure_model"):
            provider._internal_backend = "onnx"
            result = provider.embed(["test query"])

        assert result.dense.shape == (1, 1024)
        assert result.sparse is not None
        assert len(result.sparse) == 1
        assert isinstance(result.sparse[0], SparseVector)
        assert len(result.sparse[0].values) >= 1

    def test_onnx_fallback_to_torch_when_onnx_fails(self):
        """_ensure_model chama _ensure_torch quando _ensure_onnx nao seta o backend."""
        provider = BGEM3LocalProvider(backend="onnx", enable_sparse=False)
        provider._internal_backend = None
        provider._model = None

        with patch.object(provider, "_ensure_onnx", return_value=None):
            with patch.object(provider, "_ensure_torch") as mock_torch:
                provider._ensure_model()
                mock_torch.assert_called_once()

    def test_detect_cpu_features_returns_string(self):
        from pipeline.rag.embeddings.local_provider import _detect_cpu_features

        result = _detect_cpu_features()
        assert result in ("avx512", "avx2", "unknown")

    def test_onnx_sparse_format_is_valid_svector(self):
        provider = BGEM3LocalProvider(backend="torch")
        provider._internal_backend = "onnx"
        provider._model = self._make_mock_model()
        provider._onnx_tokenizer = self._make_mock_tokenizer()
        provider._cls_id = 0
        provider._eos_id = 1
        provider._pad_id = 2
        provider._unk_id = 3

        with patch.object(provider, "_ensure_model"):
            result = provider.embed(["test"])

        assert result.sparse is not None
        sv = result.sparse[0]
        assert isinstance(sv.values, dict)
        for k, v in sv.values.items():
            assert isinstance(k, int)
            assert isinstance(v, float)
            assert v > 0


class TestTorchBackendRegression:
    """Garante que backend=torch continua funcionando (retrocompatibilidade)."""

    def test_torch_backend_accepts_extra_kwargs(self):
        """_get_provider local deve aceitar backend e model_name_onnx sem quebrar."""
        from pipeline.rag.embeddings import get_provider

        p = get_provider(
            "local",
            backend="torch",
            model_name_onnx="gpahal/bge-m3-onnx-int8",
            enable_sparse=False,
        )
        assert p.name == "bge-m3-local"
        assert p.dimension == 1024


class TestProcessTokenWeights:
    """Valida que _process_token_weights agrega por token ID do vocabulario."""

    def test_aggregates_by_token_id_not_position(self):
        """Mesmo token em posicoes diferentes deve ser agregado com max weight."""
        from pipeline.rag.embeddings.local_provider import _process_token_weights
        import numpy as np

        token_weights = np.array([0.0, 0.5, 0.0, 0.3, 0.8, 0.0], dtype=np.float32)
        input_ids = [0, 101, 102, 101, 104, 1]

        result = _process_token_weights(token_weights, input_ids, cls_id=0, eos_id=1, pad_id=2, unk_id=3)

        assert 0 not in result, "CLS token should be excluded"
        assert 1 not in result, "EOS token should be excluded"
        assert 102 not in result, "Token 102 with weight 0 should not appear"
        assert result[101] == pytest.approx(0.5), "Max weight for duplicate token 101"
        assert result[104] == pytest.approx(0.8)

    def test_all_unused_tokens_excluded(self):
        from pipeline.rag.embeddings.local_provider import _process_token_weights
        import numpy as np

        token_weights = np.array([0.9, 0.8, 0.7, 0.6, 0.0], dtype=np.float32)
        input_ids = [0, 1, 2, 3, 100]

        result = _process_token_weights(token_weights, input_ids, cls_id=0, eos_id=1, pad_id=2, unk_id=3)

        assert len(result) == 0, (
            "Token 100 with weight 0.0 should be excluded (zero weight)"
        )

    def test_max_weight_for_duplicates(self):
        """Token 50 aparece 3 vezes com pesos 0.2, 0.7, 0.4 — deve ficar com 0.7."""
        from pipeline.rag.embeddings.local_provider import _process_token_weights
        import numpy as np

        token_weights = np.array([0.2, 0.7, 0.4, 0.1], dtype=np.float32)
        input_ids = [50, 50, 50, 60]

        result = _process_token_weights(token_weights, input_ids, cls_id=0, eos_id=1, pad_id=2, unk_id=3)

        assert result[50] == pytest.approx(0.7), "Should keep max weight (0.7)"
        assert result[60] == pytest.approx(0.1)
        assert len(result) == 2
