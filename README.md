# RAG System — Pipeline Completo

Transforma documentos (PDF, Word, PowerPoint, imagens) em uma base de conhecimento pesquisavel. Funciona integrado ao chat do seu assistente de IA (opencode, Claude Code, Cursor, Copilot e outros) — voce pergunta em linguagem natural e recebe respostas diretas com citacoes, tudo rodando localmente no seu computador.

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

O fluxo padrao e o proprio chat IA gerar a resposta — o `retrieve` busca os trechos relevantes e devolve para o assistente.

```bash
# 1. Indexar documento
python main.py ingest documento.pdf

# 2. Perguntar no chat — o assistente chama retrieve internamente
#    ou voce pode rodar manualmente:
python main.py retrieve "Qual o prazo de pagamento?" --top-k 20

# 3. Multiplas variacoes em uma so chamada (query expansion)
python main.py retrieve-batch "prazo PT" "deadline EN" "variacao" --top-k 20

# (Opcional) Modo standalone com LLM local (LM Studio / Ollama)
python main.py query "Qual o prazo?" --mode answer_with_citations

# Teste offline
python main.py ingest --rag-only --provider dummy --dimension 64
```

---

## Comandos

| Comando | Descricao |
|---------|-----------|
| `ingest <input>` | OCR + indexacao |
| `retrieve <q> --top-k 20` | Busca hibrida — retorna chunks para o chat IA responder |
| `retrieve-batch "q1" "q2" "q3"` | Multiplas queries, modelo carregado 1x |
| `retrieve --interactive` | REPL com modelo vivo |
| `query <q>` | *(opcional)* Com LLM local standalone (LM Studio/Ollama) |
| `reindex --purge` | Reset completo |

---

## Arquitetura

```
Documentos (PDF/Word/PPT/imagens)
    -> OCR (PyMuPDF + Tesseract) -> Markdown
    -> Chunking semantico -> BGE-M3 ONNX INT8 (embeddings) -> indice Zvec

Pergunta no chat -> retrieve (dense + FTS + RRF) -> chunks relevantes
    -> assistente IA gera resposta com citacoes
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
RAG_PROMPT_MAX_CHARS_PER_CHUNK=1600      # chars por chunk no prompt do LLM

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
