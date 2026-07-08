---
name: rag-system
description: Pipeline RAG completo — OCR de documentos brutos (PDF, DOCX, PPTX, XLSX, EPUB, imagens) + chunking semântico + embeddings BGE-M3 ONNX INT8 + Zvec + retrieval híbrido (dense+FTS+RRF) + resposta fundamentada com citações. Use quando o usuário pedir para "ingerir documento", "indexar PDF", "perguntar sobre documentos", "RAG", "buscar nos documentos", "consultar base de conhecimento". Compatível com opencode, Claude Code, Cursor, GitHub Copilot, Windsurf, Antigravity, Codex.
---

# RAG System — Pipeline Completo Local-First

Skill unificada: **OCR de documentos brutos** + **RAG completo**. Tudo 100% local.
Embeddings: BGE-M3 ONNX INT8 (544 MB, dense 1024d + sparse, tokenizer Rust `tokenizers` 21ms).
Motor de busca: Zvec 0.5.1 (HNSW + FTS nativo + RRF adaptativo).
Latência: ~3.5s (primeira query com modelo) / ~15ms (queries subsequentes em batch/interativo).

---

## 🤖 PARA O ASSISTENTE IA — GUIA COMPLETO DE USO

### Fluxo de Decisão Rápido

```
Usuário menciona documentos ou envia arquivo
├─ ARQUIVO ENVIADO (.pdf, .docx, etc)?
│  └─ ⚠️ SEMPRE PERGUNTAR antes de processar:
│     "Arquivo recebido. O que deseja?"
│     "1. Extrair texto para .md (OCR apenas)"
│     "2. Indexar no RAG para consultas"
│  
├─ PERGUNTA sobre documentos indexados?
│  ├─ 1️⃣ Verificar: existe index/manifest.json?
│  │  ├─ ✅ SIM → Continuar para busca
│  │  └─ ❌ NÃO → Responder: "Nenhum documento indexado ainda. Envie um arquivo primeiro."
│  │
│  ├─ 2️⃣ BUSCAR EVIDÊNCIAS (query expansion obrigatória!)
│  │  └─ Comando: python main.py retrieve-batch "pergunta PT" "question EN" "sinônimo" --top-k 20
│  │  └─ Resultado: JSON estruturado (parse conforme exemplo abaixo)
│  │
│  ├─ 3️⃣ PROCESSAR JSON
│  │  ├─ Parse o JSON retornado pelo comando
│  │  ├─ Colete todos os chunks de todas as queries
│  │  ├─ ⚠️ DEDUPLIQUE por chunk_id (evita repetição)
│  │  ├─ Ordene por score descendente
│  │  └─ Selecione top 6-8 chunks mais relevantes
│  │
│  └─ 4️⃣ GERAR RESPOSTA EM PORTUGUÊS
│     ├─ Use as evidências para fundamentar sua resposta
│     ├─ Adicione citações numeradas inline: [1], [2], [3]
│     ├─ Ao final, liste as fontes no formato:
│     │   [1] arquivo.md > Seção > Subseção (score=0.85)
│     │   [2] outro.md > Capítulo (score=0.78)
│     ├─ ⚠️ Se confidence < 0.3: avisar "Informação limitada nos documentos"
│     └─ ⚠️ Se nenhum resultado: "Não encontrei informação sobre isso nos documentos indexados"
│
└─ INDEXAÇÃO solicitada?
   ├─ python main.py ingest "ARQUIVO"
   └─ Confirmar: "✅ Indexado X chunks de ARQUIVO.md"
```

### Estrutura do JSON (retrieve-batch)

Quando você executa `retrieve-batch`, o sistema retorna este JSON:

```json
{
  "total_time_ms": 3542,
  "model_load_ms": 3510,
  "query_count": 3,
  "queries": [
    {
      "query": "prazo de pagamento",
      "confidence": 0.847,
      "count": 20,
      "time_ms": 15,
      "results": [
        {
          "chunk_id": "doc123_chunk_005",
          "source_path": "contrato.md",
          "score": 0.847,
          "snippet": "O prazo para pagamento é de 30 dias corridos...",
          "h1": "Condições Comerciais",
          "h2": "Pagamento",
          "title": "Contrato de Prestação de Serviços"
        },
        {
          "chunk_id": "doc123_chunk_012",
          "source_path": "contrato.md",
          "score": 0.723,
          "snippet": "Multa de 2% sobre valor em caso de atraso...",
          "h1": "Penalidades",
          "h2": "",
          "title": "Contrato de Prestação de Serviços"
        }
      ]
    },
    {
      "query": "payment deadline",
      "confidence": 0.801,
      "count": 18,
      "time_ms": 12,
      "results": [
        {
          "chunk_id": "doc123_chunk_005",
          "source_path": "contrato.md",
          "score": 0.801,
          "snippet": "O prazo para pagamento é de 30 dias corridos...",
          "h1": "Condições Comerciais",
          "h2": "Pagamento",
          "title": "Contrato de Prestação de Serviços"
        }
      ]
    }
  ]
}
```

**Como processar**:
1. Iterar por `queries[]` e coletar todos os `results[]`
2. Criar dicionário: `{chunk_id: chunk}` para deduplicar
3. Ordenar por `score` descendente
4. Pegar top 6-8

### Exemplo Completo de Interação

**👤 Usuário**: "Qual o prazo de pagamento no contrato?"

**🤖 Você (passo a passo)**:

```bash
# 1. Executar busca com query expansion
python main.py retrieve-batch "prazo de pagamento" "payment deadline" "data vencimento" --top-k 20
```

```python
# 2. Processar JSON internamente (pseudo-código)
import json
output = json.loads(command_output)

all_chunks = []
for query_result in output["queries"]:
    all_chunks.extend(query_result["results"])

# Deduplica mantendo maior score
unique = {}
for chunk in all_chunks:
    cid = chunk["chunk_id"]
    if cid not in unique or chunk["score"] > unique[cid]["score"]:
        unique[cid] = chunk

# Ordena e pega top 6
top_chunks = sorted(unique.values(), key=lambda x: x["score"], reverse=True)[:6]
```

**🤖 Sua resposta ao usuário**:

> O prazo de pagamento é de **30 dias corridos** a partir da emissão da nota fiscal [1]. 
> 
> Em caso de atraso, será aplicada multa de 2% sobre o valor devido, além de juros de 1% ao mês [2].
> 
> **Fontes consultadas**:
> - [1] contrato.md > Condições Comerciais > Pagamento (score=0.85)
> - [2] contrato.md > Penalidades (score=0.72)

### Comandos Essenciais

| Situação | Comando | Quando usar |
|----------|---------|-------------|
| **Buscar evidências** | `python main.py retrieve-batch "q1" "q2" "q3" --top-k 20` | ✅ Sempre (mais eficiente) |
| **Indexar documento** | `python main.py ingest "arquivo.pdf"` | Arquivo novo |
| **Indexar pasta** | `python main.py ingest "caminho/pasta/"` | Múltiplos arquivos |
| **Só OCR (extrair texto)** | `python main.py ingest "arquivo.pdf" --ocr-only` | Usuário só quer texto |
| **Só indexar .md** | `python main.py ingest --rag-only` | .md já existem |
| **Reset completo** | `python main.py reindex --purge` | Recomeçar do zero |
| **Busca única (não recomendado)** | `python main.py retrieve "pergunta" --top-k 20` | Pergunta isolada (lento) |

### ⚠️ REGRAS CRÍTICAS

1. **Query expansion é OBRIGATÓRIA**: Sempre use 2-3 variações (PT + EN + sinônimos)
   - ❌ Ruim: `retrieve "prazo"`
   - ✅ Bom: `retrieve-batch "prazo de pagamento" "payment deadline" "data vencimento"`

2. **Deduplicação é OBRIGATÓRIA**: Mesmo chunk pode aparecer em múltiplas queries
   - Use `chunk_id` como chave única

3. **Top 6-8 chunks**: Equilíbrio entre contexto e foco
   - Menos: perde contexto importante
   - Mais: confunde e dilui informação

4. **Citações numeradas**: Use [1], [2], não links ou nomes
   - ❌ Ruim: "segundo o arquivo contrato.md..."
   - ✅ Bom: "o prazo é de 30 dias [1]"

5. **Confidence < 0.3**: Avisar usuário
   - "⚠️ A informação nos documentos é limitada sobre este tópico"

6. **Sempre responda em português**: Mesmo se documento estiver em inglês
   - Traduza snippets quando necessário

7. **Liste fontes ao final**: Sempre, mesmo se usuário não pedir

### Filtros Permitidos

Use `--filter chave=valor` para refinar busca:

| Chave | Exemplo | Descrição |
|-------|---------|-----------|
| `language` | `--filter language=pt` | Idioma do documento |
| `doc_type` | `--filter doc_type=manual` | Tipo customizado |
| `source_path` | `--filter source_path=contrato.md` | Arquivo específico |
| `file_name` | `--filter file_name=contrato` | Nome do arquivo |

**Exemplo com filtro**:
```bash
python main.py retrieve-batch "erro 404" "error 404" --filter doc_type=manual --top-k 20
```

---

## 📖 PARA O USUÁRIO FINAL — Quando Usar

Ative esta skill quando:
- Enviar/arrastar um arquivo e pedir "leia", "indexe", "consulte", "pergunte sobre"
- Pedir "RAG", "busca semântica", "base de conhecimento local", "chat with documents"
- Mencionar extensões: `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.epub`, `.png`, `.jpg`, `.csv`, `.html`, `.md`, `.tex`, `.txt`

### Quando o usuário envia um arquivo

**SEMPRE pergunte** o que fazer antes de processar:

#### Para PDF, DOCX, PPTX, imagem:
> 📄 Arquivo recebido: `nome.pdf`. O que deseja fazer?
> 
> **1.** Extrair texto para Markdown (OCR apenas) → `python main.py ingest "ARQUIVO" --ocr-only`
> **2.** Indexar no RAG para consultas → `python main.py ingest "ARQUIVO"`

#### Para Markdown (.md):
> 📝 Detectei arquivo Markdown: `nome.md`. O que fazer?
> 
> **1.** Indexar no RAG → `python main.py ingest --rag-only`
> **2.** Apenas ler e exibir conteúdo

---

## Comandos Completos

| Comando | Uso | Latência |
|---------|-----|----------|
| `ingest "ARQUIVO"` | OCR + indexação completa | ~variável |
| `ingest "ARQUIVO" --ocr-only` | Só OCR, gera .md (não indexa) | ~variável |
| `ingest --rag-only` | Só indexa .md existentes | ~rápido |
| `retrieve "q" --top-k 20` | Busca híbrida (CLI simples) | ~4s |
| `retrieve-batch "q1" "q2" "q3"` | Múltiplas queries, 1 processo | ~3.5s total |
| `retrieve --interactive` | REPL com modelo carregado | 3.5s init + 15ms/q |
| `query "q"` | Com LLM local (LM Studio/Ollama) | ~4s + LLM |
| `query --interactive` | REPL com LLM local | 3.5s + LLM |
| `reindex --purge` | Reset completo do índice | ~variável |
| `run <arquivo> <pergunta>` | Pipeline one-shot: OCR + ingest + query | ~4s + LLM |

### Flags Comuns

| Flag | Comandos | Descrição |
|------|----------|-----------|
| `--filter chave=valor` | `query`, `retrieve`, `retrieve-batch` | Filtro escalar (ex: `language=pt`, `doc_type=manual`) |
| `--retrieval-mode <modo>` | `query`, `retrieve`, `retrieve-batch` | `dense` \| `fts` \| `hybrid` \| `semantic` \| `sparse` |
| `--top-k N` | `retrieve`, `retrieve-batch` | Número de chunks a recuperar (padrão: 20) |
| `--mode <modo>` | `query` | Modo de resposta do LLM (ver abaixo) |
| `--llm-model <modelo>` | `query` | Modelo no LM Studio/Ollama (ex: `llama-3.2`) |
| `--provider dummy --dimension 64` | Todos | Teste offline sem modelos carregados |

---

## Modos de Retrieval

| Modo | Combinação | RRF k | Melhor para |
|------|-----------|-------|-------------|
| `hybrid` **(padrão)** | dense + FTS + RRF | adaptativo (5-60) | Qualquer tipo de query |
| `semantic` | dense + sparse + RRF | 10 | Queries conceituais ("o que é...") |
| `dense` | Só vetorial (HNSW) | — | Similaridade semântica pura |
| `fts` | Só BM25 (Zvec nativo) | — | Termos exatos, nomes, números |
| `sparse` | Só lexical (BGE-M3) | — | Matching lexical |

**RRF adaptativo**: O sistema detecta automaticamente o tipo de query:
- Queries factuais (nomes, números) → k=5 (FTS domina)
- Queries conceituais ("o que é") → k=60 (balanceado)
- Isso elimina contaminação entre documentos diferentes

---

## Arquitetura Técnica

| Camada | Tecnologia | Detalhe |
|--------|-----------|---------|
| Text Cleaner | Regex/stdlib | Remove ruídos: refs [1], URLs duplicadas, fragmentos |
| Chunker | Semântico Markdown | Hierarquia de headings, janela deslizante, anti-ruído |
| Tokenizer | `tokenizers` (Rust) | 21ms vs 12s do `transformers` |
| Embedding | BGE-M3 ONNX INT8 (C++) | 1024d dense + sparse, 6 threads (metade CPU) |
| Embed Cache | SQLite (stdlib) | Cacheia dense+sparse por chunk_id, versionado |
| Query Cache | LRU em memória | 128 queries, zero I/O para queries repetidas |
| Dense search | Zvec HNSW (COSINE) | ~1ms, C++ nativo |
| FTS | Zvec FTS nativo (BM25) | ~1ms, C++ nativo |
| Hybrid fusion | Zvec RRF adaptativo | k=5-60 por tipo de query |
| Reranker | BGE Reranker ONNX | Opcional (desabilitado por padrão) |

---

## Modos de Resposta LLM

Usado com `python main.py query "pergunta" --mode <modo>`:

| Modo | Comportamento | Quando usar |
|------|---------------|-------------|
| `answer` | Resposta direta sem fontes | Quick answers |
| `answer_with_citations` | Resposta + citações [n] + lista de fontes | **Padrão recomendado** |
| `extractive_summary` | Passagens literais (verbatim) | Quando precisar texto original |
| `study_mode` | Tutor didático estruturado + key points | Aprendizado/estudo |

---

## Modos de OCR

| Modo | Fluxo | Quando usar |
|------|-------|-------------|
| `hybrid` **(padrão)** | Layout → Tesseract → AI fallback | Sempre (melhor qualidade/custo) |
| `classic_only` | Layout → Tesseract apenas | LM Studio offline |
| `ai_only` | Só IA (LM Studio), sem Tesseract | Docs complexos/baixa qualidade |
| `legacy` | Só IA (ignora layout) | Compatibilidade com modelos antigos |

---

## Instalação

```bash
pip install -r "SKILL_DIR/requirements.txt"
pip install zvec onnxruntime tokenizers huggingface-hub

# Opcional: GPU AMD/Intel/NVIDIA (Windows) — 2.8x mais rápido
pip uninstall onnxruntime
pip install onnxruntime-directml
```

---

## Cache de Embeddings

Habilitado por padrão. SQLite em `index/embed_cache.db` (zero deps externas).

**Benefícios**:
- **Reindex incremental**: se 1 parágrafo muda em doc de 200 chunks, apenas 1 é re-embeddado
- **Versionado por modelo**: trocar torch/onnx invalida cache automaticamente
- **Query cache LRU**: 128 queries em memória para modo interativo

**Gerenciar cache**:
```bash
# Desabilitar
RAG_EMBED_CACHE_ENABLED=false python main.py ingest "ARQUIVO"

# Resetar cache (Windows cmd)
del index\embed_cache.db

# Resetar cache (Windows PowerShell)
Remove-Item index/embed_cache.db

# Resetar cache (Linux/macOS)
rm index/embed_cache.db
```

---

## Troubleshooting

| Sintoma | Solução |
|---------|---------|
| Zvec não instalado | `pip install zvec>=0.5.1` |
| Embeddings não carregam | Teste com `--provider dummy --dimension 64` |
| ONNX não carrega | `pip install onnxruntime huggingface-hub tokenizers` |
| Tokenizer não carrega | `pip install tokenizers>=0.19` |
| OCR/LM Studio offline | Use `--ocr-mode classic_only` |
| Nenhum chunk retornado | Execute `ingest` primeiro para indexar |
| Erro de lock Zvec | Aguarde 2s (lock pendente). Sistema retry 5x automaticamente |
| Documento novo não aparece | Cache? Use `--retrieval-mode fts` para termos exatos |
| CPU 100% na indexação | Normal. Threads ONNX usam metade da CPU: `max(2, cpu/2)` |
| Resposta com doc errado | RRF adaptativo resolve. Caso extremo: `:mode fts` |
| "Filtro inválido" | Chaves permitidas: `language`, `doc_type`, `source_path`, `file_name`, `doc_id` |

---

## Variáveis de Ambiente Úteis

```bash
# Cache de embeddings
RAG_EMBED_CACHE_ENABLED=true
RAG_EMBED_CACHE_DIR=index/embed_cache.db

# Performance
RAG_EMBEDDING_ONNX_DEVICE=auto    # auto | cpu | dml (GPU)
RAG_EMBEDDING_BATCH_SIZE=32

# Retrieval
RAG_RETRIEVAL_TOP_K=20
RAG_RETRIEVAL_MODE=hybrid         # dense | fts | hybrid | sparse
RAG_RETRIEVAL_MAX_CONTEXT_CHUNKS=8

# OCR
RAG_OCR_MODE=hybrid               # hybrid | classic_only | ai_only
RAG_OCR_QUALITY_THRESHOLD=0.70
RAG_OCR_LANGS=por+eng

# Logging
RAG_LOG_LEVEL=INFO                # DEBUG | INFO | WARNING | ERROR
```
