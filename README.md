# NerdFarm

Document processing system for legal documents. Extracts text from PDFs and images, indexes in ChromaDB, generates drafts with source citations, learns from operator edits.

## Quick Start

### Requirements

- Python 3.12+
- uv package manager
- Poppler: `apt install poppler-utils` (Linux) / `brew install poppler` (macOS)
- OpenRouter API key

### Setup

```bash
cd nerdfarm
cp .env.example .env
# Edit .env: OPENROUTER_API_KEY=sk-or-xxxxx
uv sync
```

### Run

**Backend:**
```bash
uv run uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
uv run streamlit run frontend.py
```

- API: http://localhost:8000
- API docs: http://localhost:8000/docs
- Frontend: http://localhost:8501

## System Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Document Upload │────▶│  Vision Extractor │────▶│   Vector Store  │
│  (PDF/IMG/TXT)   │     │  (GPT-4o-mini)    │     │   (ChromaDB)    │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              │
│  Operator Edits  │────▶│   Edit Store     │     ┌───────▼────────┐
│  (Submit/Learn)  │     │   (ChromaDB)     │────▶│  Draft Generator│
└─────────────────┘     └──────────────────┘     │  (GPT-4o)       │
                                                  └────────────────┘
```

## Features

- **Document Processing:** PDF, images, text files → extracted text with structured data
- **Hybrid Search:** Vector + BM25 + cross-encoder reranking
- **Grounded Generation:** Drafts include `[Source: filename, Page X]` citations
- **Learning:** Quality-gated edit storage with few-shot retrieval

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/documents/upload | Upload and process document |
| GET | /api/documents | List all documents |
| DELETE | /api/documents/{id} | Delete document |
| POST | /api/drafts/generate | Generate grounded draft |
| GET | /api/drafts/{id} | Get draft by ID |
| POST | /api/edits/submit | Submit operator edit |
| GET | /api/edits/history | View edit history |

## Draft Types

| Type | Description |
|------|-------------|
| case_fact_summary | Case facts and parties |
| title_review_summary | Property title analysis |
| notice_summary | Legal notice summary |
| document_checklist | Required documents list |
| internal_memo | Professional memo |

## Tests

```bash
uv run pytest tests/ -v
```

26 tests covering QualityGate, EditStore, VectorStore, DraftGenerator.

## Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture diagrams, data flow, design choices, technology stack
- **[docs/EVALUATION.md](docs/EVALUATION.md)** - Evaluation methodology, test results, performance metrics, end-to-end scenarios

## Project Structure

```
app/
├── main.py              # FastAPI routes
├── config.py            # Settings from .env
├── api/                 # Document, draft, edit endpoints
├── document_processor/  # Extractor, vision processor
├── retrieval/           # Vector store, BM25, reranker
├── generation/          # Draft generator, LLM client
├── learning/            # Edit store, quality gate
└── persistence/         # JSON file stores

frontend.py              # Streamlit UI
sample_docs/             # Sample legal documents
sample_outputs/          # Example generated drafts
```

## Configuration

```
OPENROUTER_API_KEY=sk-or-xxxxx
CHROMA_PERSIST_DIR=./chroma_data
UPLOAD_DIR=./uploads
MAX_UPLOAD_SIZE=52428800
```

## Limitations

- No authentication (single-user)
- Character-based chunking (500 chars)
- ChromaDB single-machine
- No async job queue
