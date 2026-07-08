---
name: rag-system
description: Pipeline RAG completo — OCR de documentos brutos (PDF, DOCX, PPTX, XLSX, EPUB, imagens) + chunking semântico + embeddings BGE-M3 ONNX INT8 + Zvec + retrieval híbrido (dense+FTS+RRF) + resposta fundamentada com citações. Use quando o usuário pedir para "ingerir documento", "indexar PDF", "perguntar sobre documentos", "RAG", "buscar nos documentos", "consultar base de conhecimento". Compatível com opencode, Claude Code, Cursor, GitHub Copilot, Windsurf, Antigravity, Codex.
---

# RAG System — Pipeline Completo Local-First

Skill unificada: **OCR de documentos brutos** + **RAG completo**. Tudo 100% local.
Embeddings: BGE-M3 ONNX INT8 (544 MB, dense 1024d + sparse, tokenizer Rust `tokenizers` 21ms).
Motor de busca: Zvec 0.5.1 (HNSW + FTS nativo + RRF adaptativo).
Latência: ~3.5s (primeira query com modelo) / ~15ms (queries subsequentes em batch/interativo).

## Quando Usar

Ative quando o usuário:
- Enviar/arrastar um arquivo e pedir "leia", "indexe", "consulte", "pergunte sobre"
- Pedir "RAG", "busca semântica", "base de conhecimento local", "chat with documents"
- Mencionar extensões: `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.epub`, `.png`, `.jpg`, `.csv`, `.html`, `.md`, `.tex`, `.txt`

## Quando o usuário envia um arquivo — Pergunte primeiro

**SEMPRE pergunte** o que fazer antes de processar.

### PDF, DOCX, PPTX, imagem:
> Detalhes do arquivo. O que fazer?
> **1.** Extrair texto (.md) → `python main.py ingest "ARQUIVO" --ocr-only`
> **2.** Indexar no RAG → `python main.py ingest "ARQUIVO"`

### Markdown (.md):
> Detectei `nome.md`. O que fazer?
> **1.** Indexar no RAG → `python main.py ingest --rag-only`
> **2.** Apenas ler → mostro o conteúdo

## Estratégia de Retrieval

### Query Expansion com batch (recomendado)

Para maximizar cobertura, expanda a pergunta em 2-3 variações PT+EN e use `retrieve-batch`:

```bash
python main.py retrieve-batch "pergunta original" "variacao em ingles" "variacao alternativa" --top-k 20
```

O comando carrega o modelo **1 vez** (~3.5s) e processa todas as queries em ~15ms cada. Retorna JSON estruturado com todos os resultados. Merge, remova duplicatas por `chunk_id`, selecione top 6-8. Gere resposta em português com citações `[1]`, `[2]`.

### Modo interativo (exploração)

```bash
python main.py retrieve --interactive
>>> pergunta 1
>>> pergunta 2
>>> :mode fts       # trocar modo de busca
>>> :quit
```

### Modo CLI simples (pergunta única)

```bash
python main.py retrieve "pergunta" --top-k 20
```

Cada chamada recarrega o modelo (~4s). Use só para perguntas isoladas.

## Comandos

| Comando | Uso | Latência |
|---------|-----|----------|
| `ingest "ARQUIVO"` | OCR + indexação | — |
| `ingest "ARQUIVO" --ocr-only` | Só OCR (.md, não indexa) | — |
| `ingest --rag-only` | Só indexa .md existentes | — |
| `retrieve "q" --top-k 20` | Busca híbrida (CLI) | ~4s |
| `retrieve-batch "q1" "q2" "q3"` | Múltiplas queries, um processo | ~3.5s total |
| `retrieve --interactive` | REPL com modelo vivo | 3.5s + 15ms/q |
| `query "q"` | Com LLM local (LM Studio/Ollama) | ~4s + LLM |
| `query --interactive` | REPL com LLM | 3.5s + LLM |
| `reindex --purge` | Reset completo do índice | — |

## Modos de Retrieval

| Modo | Combinação | RRF k | Melhor para |
|------|-----------|-------|-------------|
| `hybrid` **(padrão)** | dense + FTS + RRF | adaptativo (5-60) | Qualquer tipo de query |
| `semantic` | dense + sparse + RRF | 10 | Queries conceituais puras |
| `dense` | Só vetorial (HNSW) | — | Similaridade semântica |
| `fts` | Só BM25 (Zvec nativo) | — | Termos específicos, nomes, números |
| `sparse` | Só lexical (BGE-M3) | — | Matching lexical |

**RRF adaptativo**: O sistema detecta automaticamente se a query é factual (nomes, números → k=5, FTS domina) ou conceitual ("o que é X" → k=60, balanceado). Isso elimina contaminação cruzada entre documentos.

## Arquitetura

| Camada | Tecnologia | Detalhe |
|--------|-----------|---------|
| Text Cleaner | Regex/stdlib | Remove ruídos: refs [1], URLs duplicadas, fragmentos quebrados |
| Chunker | Semântico Markdown | Hierarquia de headings, janela deslizante, filtro anti-ruído |
| Tokenizer | `tokenizers` (Rust) | 21ms vs 12s do `transformers` |
| Embedding | BGE-M3 ONNX INT8 (C++ via onnxruntime) | 1024d dense + sparse, 6 threads (metade CPU) |
| Embed Cache | SQLite (stdlib) | Cacheia dense+sparse por chunk_id, versionado por modelo |
| Query Cache | LRU em memória | 128 queries, zero I/O para queries repetidas |
| Dense search | Zvec HNSW (COSINE, C++ nativo) | ~1ms |
| FTS | Zvec FTS nativo (BM25, C++ nativo) | ~1ms |
| Hybrid fusion | Zvec RRF (Reciprocal Rank Fusion) | k adaptativo por tipo de query |
| Reranker | BGE Reranker ONNX | Opcional, desabilitado por padrão |

## Modos de OCR

| Modo | Fluxo | Quando |
|------|-------|--------|
| `hybrid` **(padrão)** | Layout → Tesseract → AI fallback | Sempre |
| `classic_only` | Layout → Tesseract | LM Studio offline |

## Instalação

```bash
pip install -r "SKILL_DIR/requirements.txt"
pip install zvec onnxruntime tokenizers huggingface-hub
```

## Cache de Embeddings

Habilitado por padrao. SQLite em `index/embed_cache.db` (zero deps externas).

- **Reindex incremental**: se 1 paragrafo muda em doc de 200 chunks, apenas 1 chunk e re-embeddado
- **Versionado por modelo**: trocar torch/onnx invalida cache automaticamente
- **Query cache LRU**: 128 queries em memoria para modo interativo

```bash
# Desabilitar
RAG_EMBED_CACHE_ENABLED=false python main.py ingest "ARQUIVO"

# Resetar cache
rm index/embed_cache.db
```

## Troubleshooting

| Sintoma | Solução |
|---------|---------|
| Zvec não instalado | `pip install zvec>=0.5.1` |
| Embeddings não carregam | `--provider dummy --dimension 64` |
| ONNX não carrega | `pip install onnxruntime huggingface-hub` |
| Tokenizer não carrega | `pip install tokenizers>=0.19` |
| OCR/LM Studio offline | `--ocr-mode classic_only` |
| Nenhum chunk | Execute `ingest` primeiro |
| Erro de lock Zvec | Aguarde 2s (lock de escrita pendente). O sistema retry 5x automaticamente |
| Documento novo não aparece no hybrid | Use `--retrieval-mode fts` para termos exatos |
| Computador travando na indexação | Threads ONNX usam metade da CPU automaticamente (`max(2, cpu/2)`) |
| Resposta contaminada (documento errado) | RRF adaptativo resolve; para casos extremos use `:mode fts` |
