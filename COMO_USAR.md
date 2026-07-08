# Como Usar — RAG System

---

## Primeira execucao

```bash
pip install -r requirements.txt
pip install zvec onnxruntime huggingface-hub transformers
```

**GPU AMD/Intel/NVIDIA (Windows) — recomendado:**
```bash
pip uninstall onnxruntime
pip install onnxruntime-directml
```
O sistema detecta automaticamente. Indexacao ate **2.8x mais rapida**.

**LLM local** (opcional, so para `query`):
- LM Studio: https://lmstudio.ai/
- Ollama: `ollama pull llama3.2 && ollama serve`

**OCR** (opcional, para PDFs/imagens):
```bash
winget install UB-Mannheim.TesseractOCR
pip install PyMuPDF Pillow requests pytesseract
```

---

## Fluxos

### 1. Indexar e consultar com chat IA (recomendado)

```bash
python main.py ingest documento.pdf
python main.py retrieve "Qual o prazo?" --top-k 20

# Query expansion (multi-idioma)
python main.py retrieve-batch "prazo de pagamento" "payment due date" "data limite" --top-k 20
```

### 2. Indexar e consultar com LLM local

```bash
python main.py ingest documento.pdf
python main.py query "Qual o prazo?" --mode answer_with_citations
```

### 3. Teste offline (sem modelos)

```bash
python main.py ingest --rag-only --provider dummy --dimension 64
python main.py retrieve "teste" --provider dummy --dimension 64
```

---

## Comandos

| Comando | O que faz |
|---------|-----------|
| `ingest doc.pdf` | OCR -> .md -> indexar |
| `ingest --rag-only` | So indexa .md existentes |
| `ingest --ocr-only` | So OCR (.md, nao indexa) |
| `retrieve "q" --top-k 20` | Busca hibrida |
| `retrieve-batch "q1" "q2" "q3"` | Multi-queries, modelo 1x |
| `retrieve --interactive` | REPL interativo |
| `query "q"` | Com LLM local |
| `reindex --purge` | Reset completo |

---

## Onde os arquivos ficam

```
data/       <- arquivos brutos
markdown/   <- .md gerados pelo OCR
index/      <- Zvec + manifesto + embed_cache.db
```

---

## Cache de Embeddings

O sistema cacheia embeddings no SQLite para acelerar reindexacoes. Se voce edita 1 paragrafo em um documento de 200 chunks, apenas o chunk alterado e re-embeddado.

**Desabilitar:**
```bash
RAG_EMBED_CACHE_ENABLED=false python main.py ingest documento.pdf
```

**Resetar cache:**
```bash
rm index/embed_cache.db
```

---

## Configuracoes principais

| Variavel | Padrao | Descricao |
|----------|--------|-----------|
| `RAG_EMBED_CACHE_ENABLED` | `true` | Cache de embeddings (SQLite) |
| `RAG_EMBED_CACHE_DIR` | `index/embed_cache.db` | Caminho do cache |
| `RAG_EMBEDDING_ONNX_DEVICE` | `auto` | `auto` / `cpu` / `dml` (GPU) |
| `RAG_RETRIEVAL_TOP_K` | `20` | Chunks recuperados por query |
| `RAG_RETRIEVAL_MODE` | `hybrid` | `dense` / `fts` / `hybrid` / `sparse` |
| `RAG_PROMPT_MAX_CHARS_PER_CHUNK` | `1600` | Limite de chars/chunk no prompt |
| `RAG_OCR_MODE` | `hybrid` | `hybrid` / `classic_only` / `legacy` |

---

## Dicas de query expansion

| Pergunta | Variacoes |
|----------|-----------|
| "prazo de entrega" | "delivery deadline", "data limite envio" |
| "multa por atraso" | "late payment penalty", "juros mora" |
| "requisitos tecnicos" | "technical requirements", "system specs" |

---

## Troubleshooting

| Problema | Solucao |
|----------|---------|
| Zvec nao instalado | `pip install zvec>=0.5.1` |
| Embeddings nao carregam | `--provider dummy --dimension 64` |
| Tesseract nao instalado | `winget install UB-Mannheim.TesseractOCR` |
| GPU nao detectada | `pip install onnxruntime-directml` |
| ONNX erro de lock | Aguarde 2s (lock de escrita pendente) |
