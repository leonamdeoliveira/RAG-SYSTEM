# Como Usar — RAG System

---

## Primeira execução

```bash
pip install -r requirements.txt
pip install zvec onnxruntime huggingface-hub transformers
```

**LLM local** (opcional, só para `query`):
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
python main.py retrieve "payment due date" --top-k 20
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
| `ingest doc.pdf` | OCR → .md → indexar |
| `ingest --rag-only` | Só indexa .md existentes |
| `ingest --ocr-only` | Só OCR (.md, não indexa) |
| `retrieve "q" --top-k 20` | Busca híbrida |
| `query "q"` | Com LLM local |
| `reindex --purge` | Reset completo |

---

## Onde os arquivos ficam

```
data/       ← arquivos brutos
markdown/   ← .md gerados pelo OCR
index/      ← Zvec + manifesto
```

---

## Dicas de query expansion

| Pergunta | Variações |
|----------|-----------|
| "prazo de entrega" | "delivery deadline", "data limite envio" |
| "multa por atraso" | "late payment penalty", "juros mora" |
| "requisitos técnicos" | "technical requirements", "system specs" |

---

## Troubleshooting

| Problema | Solução |
|----------|---------|
| Zvec não instalado | `pip install zvec>=0.5.1` |
| Embeddings não carregam | `--provider dummy --dimension 64` |
| Tesseract não instalado | `winget install UB-Mannheim.TesseractOCR` |
