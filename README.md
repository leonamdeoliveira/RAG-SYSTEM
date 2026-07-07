# RAG System — Pipeline Local-First Completo

Transforma documentos (PDF, Word, PowerPoint, imagens) em uma base de conhecimento pesquisavel. Funciona integrado ao chat do seu assistente de IA (opencode, Claude Code, Cursor, Copilot e outros) — voce pergunta em linguagem natural e recebe respostas diretas com citacoes, tudo rodando localmente no seu computador, sem internet.

Compativel com opencode, Claude Code, Cursor, GitHub Copilot, Windsurf, Antigravity, Codex.

---

## Instalacao

```bash
pip install -r requirements.txt
pip install zvec onnxruntime huggingface-hub transformers
```

**GPU AMD/Intel/NVIDIA (Windows):**
```bash
pip uninstall onnxruntime
pip install onnxruntime-directml
```
O sistema detecta automaticamente. Indexacao fica ate **2.8x mais rapida** em batch.

**Requisitos:** Python 3.10+, Zvec v0.5.1+.

---

## Uso rapido

```bash
# Indexar
python main.py ingest documento.pdf

# Buscar (query expansion)
python main.py retrieve "prazo de pagamento" --top-k 20

# Multi-query eficiente (modelo carregado 1x)
python main.py retrieve-batch "pergunta PT" "question EN" "variacao" --top-k 20

# Com LLM local
python main.py query "Qual o prazo?" --mode answer_with_citations

# Teste offline
python main.py ingest --rag-only --provider dummy --dimension 64
python main.py retrieve "teste" --provider dummy --dimension 64
```

---

## Comandos

| Comando | Descricao |
|---------|-----------|
| `ingest <input>` | OCR + indexacao |
| `retrieve <q> --top-k 20` | Busca hibrida (dense+sparse+FTS), formata p/ chat IA |
| `retrieve-batch "q1" "q2" "q3"` | Multiplas queries, modelo 1x (~3.5s total) |
| `retrieve --interactive` | REPL com modelo vivo |
| `query <q>` | Com LLM local |
| `reindex --purge` | Reset completo |

---

## Arquitetura

```
PDF/DOCX -> OCR (PyMuPDF + Tesseract + LM Studio) -> .md
.md -> Chunking semantico -> BGE-M3 ONNX INT8 (543 MB, dense 1024d + sparse) -> Zvec
Query -> Retrieval hibrido (dense + FTS + RRF adaptativo) -> Chat IA gera resposta
```

**Tempo:** ~3.5s por consulta, ~12s para indexar 55 chunks (DML GPU) / ~33s (CPU).

---

## Configuracao

```bash
# Embeddings
RAG_EMBEDDING_BACKEND=onnx              # onnx (recomendado) | torch
RAG_EMBEDDING_ONNX_DEVICE=auto          # auto | cpu | dml (DirectML GPU)
RAG_EMBEDDING_MODEL_ONNX=gpahal/bge-m3-onnx-int8

# Retrieval
RAG_RETRIEVAL_MODE=hybrid
RAG_RETRIEVAL_TOP_K=20
RAG_RETRIEVAL_MAX_CONTEXT_CHUNKS=8
RAG_RETRIEVAL_MAX_PER_DOC=3

# Prompt (economia de tokens)
RAG_PROMPT_MAX_CHARS_PER_CHUNK=800      # chars por chunk no prompt do LLM

# LLM
RAG_LLM_BASE_URL=http://localhost:1234/v1
RAG_LLM_MODEL=local
RAG_ANSWER_MODE=answer_with_citations

# OCR
RAG_OCR_MODE=hybrid
RAG_OCR_LANGS=por+eng
```

---

## Estrutura

```
rag-system/
├── main.py / SKILL.md / README.md / COMO_USAR.md
├── requirements.txt / .env.example
├── pipeline/
│   ├── ocr/                    # OCR: PDF/DOCX/PPTX -> Markdown
│   └── rag/                    # RAG: Markdown -> Zvec -> Query
├── tests/                      # 88 testes
├── data/   markdown/   index/
```

---

## Troubleshooting

| Problema | Solucao |
|----------|---------|
| Zvec nao instalado | `pip install zvec>=0.5.1` |
| Embeddings nao carregam | `--provider dummy --dimension 64` |
| ONNX nao carrega | `pip install onnxruntime huggingface-hub` |
| OCR nao funciona | `--ocr-mode classic_only` |
| GPU AMD/Intel/NVIDIA | `pip install onnxruntime-directml` |
