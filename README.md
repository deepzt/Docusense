# DocuSense Pro

A local document Q&A system powered by hybrid RAG (Retrieval-Augmented Generation). Upload a PDF, CSV, Excel, or TXT file and ask questions in plain English — answers are grounded in your document with inline source citations.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Gradio](https://img.shields.io/badge/UI-Gradio-orange) ![FAISS](https://img.shields.io/badge/Index-FAISS%20HNSW-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Features

- **Hybrid retrieval** — FAISS HNSW dense search + BM25 sparse search fused with Reciprocal Rank Fusion
- **Parent-child chunking** — indexes small chunks for precision, expands to full page/section window for LLM context
- **Semantic chunking** — splits text at topic-shift boundaries instead of fixed character windows
- **CrossEncoder reranking** — BAAI/bge-reranker-base re-scores retrieved passages before generation
- **Adaptive confidence threshold** — auto-calibrates the relevance floor from session history
- **Faithfulness check** — warns when an answer may stray outside the retrieved context
- **Streaming responses** — tokens stream to the UI as they are generated
- **Voice I/O** — microphone input (Google Speech API) and text-to-speech output
- **Dual LLM backend** — OpenAI (`gpt-4o-mini`) or local Ollama (`llama3.1`, `gemma`)
- **Proper system prompts** — system/user message roles used for both OpenAI and Ollama
- **Log rotation** — query log capped at 50 MB, rotates to `.1` automatically

---

## Supported File Types

| Format | Notes |
|--------|-------|
| PDF | Text-based and scanned (OCR via Tesseract, optional) |
| TXT | Plain text and log files |
| CSV | Each row indexed; column statistics summary prepended |
| Excel (.xls / .xlsx) | Same as CSV |

---

## Architecture

```
Document
   │
   ├─ Semantic chunking (topic-boundary splits)
   ├─ HNSW FAISS index  (dense embeddings — BAAI/bge-base-en-v1.5)
   ├─ BM25 index        (sparse keyword scores)
   └─ Parent store      (full page/section windows for context)

Query
   │
   ├─ Conditional rewrite  (skip for short / quoted queries)
   ├─ Hybrid retrieval     (FAISS + BM25 → RRF fusion)
   ├─ CrossEncoder rerank  (BAAI/bge-reranker-base)
   ├─ Deduplication        (Jaccard similarity filter)
   ├─ Adaptive threshold   (session 20th-percentile gate)
   └─ Parent expansion     (child chunk → full page context)

LLM  (SystemMessage + HumanMessage)
   │
   ├─ Streaming answer
   ├─ Faithfulness check
   └─ Optional polish pass
```

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone <repository-url>
cd RAG
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key (optional — Ollama works without it):

```
OPENAI_API_KEY=sk-...
```

### 3. Start the app

```bash
python ui_app.py
```

Open **http://127.0.0.1:7860** in your browser.

---

## LLM Backends

### OpenAI (cloud)
Add `OPENAI_API_KEY` to `.env`. The app uses `gpt-4o-mini` by default.

### Ollama (local, no API key needed)
1. Install [Ollama](https://ollama.com)
2. Pull a model:
   ```bash
   ollama pull llama3.1
   # or
   ollama pull gemma
   ```
3. Leave `OPENAI_API_KEY` blank — the app auto-detects Ollama.

---

## Command-Line Interface

For scripted or batch usage without the web UI:

```bash
python script.py
```

You will be prompted for a file path and can then ask questions interactively. History is kept for the last 20 exchanges.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | No | Enables OpenAI `gpt-4o-mini` |
| `GRADIO_USERNAME` | No | Enables login gate on the web UI |
| `GRADIO_PASSWORD` | No | Password for the login gate |
| `TESSERACT_CMD` | No | Path to Tesseract executable (scanned PDFs) |
| `POPPLER_PATH` | No | Path to Poppler `bin/` directory (scanned PDFs) |

---

## Docker

```bash
docker build -t docusense-pro .
docker run -p 7860:7860 --env-file .env docusense-pro
```

The image runs as a non-root user and includes a health check at `GET /`.

---

## Project Structure

```
RAG/
├── rag_core.py        # Full pipeline: ingest → embed → index → retrieve → generate
├── ui_app.py          # Gradio web interface
├── script.py          # CLI interface
├── requirements.txt   # Python dependencies
├── Dockerfile
├── .env               # Your secrets (git-ignored)
├── .env.example       # Template
└── logs/
    └── rag_log.csv    # Per-query log (auto-rotates at 50 MB)
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| CPU | 4 cores | 8+ cores |
| GPU | — | CUDA GPU (faster embeddings + Ollama) |
| Disk | 2 GB | 5 GB (model weights cached in `~/.cache`) |

Embedding and reranking run on CPU by default; a CUDA GPU is used automatically if available and has > 2 GB free VRAM.

---

## Troubleshooting

**App hangs on startup**
Ollama test calls can time out if no model is running. Start Ollama first (`ollama run llama3.1`) or set `OPENAI_API_KEY`.

**"No relevant content found"**
Try rephrasing the question with more specific terms. For summarization, use the word *summarize* — the pipeline switches to a broader retrieval strategy automatically.

**Scanned PDF returns no text**
Install Tesseract and Poppler, then set `TESSERACT_CMD` and `POPPLER_PATH` in `.env`.

**Old UI showing after restart**
Hard-refresh the browser: `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (macOS).

**Kill stale processes on Windows**
```powershell
Get-Process python* | Stop-Process -Force
```

---

## License

MIT
