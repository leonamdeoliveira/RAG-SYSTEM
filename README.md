# RAG System — Pipeline Local-First Completo

Skill unificada: **OCR de documentos brutos** + **RAG completo**. 100% local.

Compatível com opencode, Claude Code, Cursor, GitHub Copilot, Windsurf, Antigravity, Codex.

---

## Instalação

```bash
pip install -r requirements.txt
pip install zvec onnxruntime huggingface-hub transformers
```

**Requisitos:** Python 3.10+, Zvec v0.5.1+.

---

## Uso rápido

```bash
# Indexar
python main.py ingest documento.pdf

# Buscar (query expansion — execute sequencialmente)
python main.py retrieve "prazo de pagamento" --top-k 20
python main.py retrieve "payment due date" --top-k 20

# Com LLM local
python main.py query "Qual o prazo?" --mode answer_with_citations

# Teste offline
python main.py ingest --rag-only --provider dummy --dimension 64
python main.py retrieve "teste" --provider dummy --dimension 64
```

---

## Comandos

| Comando | Descrição |
|---------|-----------|
| `ingest <input>` | OCR + indexação |
| `retrieve <q> --top-k 20` | Busca híbrida (dense+sparse+FTS), formata p/ chat IA |
| `query <q>` | Com LLM local |
| `reindex --purge` | Reset completo |

---

## Arquitetura

```
PDF/DOCX → OCR (PyMuPDF + Tesseract + LM Studio) → .md
.md → Chunking → BGE-M3 ONNX INT8 (543 MB, dense 1024d + sparse) → Zvec
Query → Retrieval híbrido (dense + sparse + FTS) → Chat IA gera resposta
```

**Tempo:** ~11s por consulta (CPU), ~90s para indexar 1000 chunks.

---

## Configuração

```bash
RAG_EMBEDDING_BACKEND=onnx          # onnx (recomendado) | torch (fallback)
RAG_RETRIEVAL_MODE=hybrid
RAG_RETRIEVAL_TOP_K=10
RAG_LLM_BASE_URL=http://localhost:1234/v1
RAG_OCR_MODE=hybrid
```

---

## Estrutura

```
rag-system/
├── main.py / SKILL.md / README.md / COMO_USAR.md
├── requirements.txt / .env.example
├── pipeline/
│   ├── ocr/                    # OCR: PDF/DOCX/PPTX → Markdown
│   └── rag/                    # RAG: Markdown → Zvec → Query
├── tests/                      # 88 testes
├── data/   markdown/   index/
```

---

## Troubleshooting

| Problema | Solução |
|----------|---------|
| Zvec não instalado | `pip install zvec>=0.5.1` |
| Embeddings não carregam | `--provider dummy --dimension 64` |
| ONNX não carrega | `pip install onnxruntime huggingface-hub` |
| OCR não funciona | `--ocr-mode classic_only` |
