# NerdFarm — Document Understanding & Grounded Drafting System

An AI-powered internal tooling system for **Pearson Specter Litt** that processes messy legal-style documents, extracts structured information, and produces grounded draft outputs that improve from operator feedback.

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Poppler (for PDF processing):
  - Linux: `sudo apt install poppler-utils`
  - macOS: `brew install poppler`
- OpenRouter API key ([get one here](https://openrouter.ai/))

### Setup

```bash
# Clone and enter the project
cd nerdfarm

# Copy environment file and add your API key
cp .env.example .env
# Edit .env and set: OPENROUTER_API_KEY=sk-or-...

# Install dependencies
uv sync

# Start the backend server
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Running the Frontend

In a separate terminal:

```bash
# Start the Streamlit frontend
uv run streamlit run frontend.py
```

**Access:**
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Frontend UI: http://localhost:8501

## System Overview

NerdFarm handles the full document understanding pipeline:

1. **Document Processing** — Vision-based extraction from PDFs, images, and text files using GPT-4o Vision
2. **Grounded Retrieval** — Hybrid vector + keyword search with cross-encoder reranking
3. **Draft Generation** — LLM-powered legal draft creation grounded in retrieved evidence
4. **Improvement from Edits** — Captures operator edits and uses few-shot learning

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Document Upload │────▶│  Vision Extractor │────▶│   Vector Store  │
│  (PDF/IMG/TXT)   │     │  (GPT-4o Vision)  │     │   (ChromaDB)    │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              │
│  Operator Edits  │────▶│   Edit Store     │     ┌───────▼────────┐
│  (Submit/Learn)  │     │   (ChromaDB)     │────▶│  Draft Generator│
└─────────────────┘     └──────────────────┘     │  (LLM + RAG)   │
                                                  └────────────────┘
```

## Frontend Guide

The Streamlit frontend provides 5 pages:

### 🏠 Home
System overview and workflow description.

### 📤 Upload
- Drag-and-drop file upload
- Supports: PDF, PNG, JPG, JPEG, TIFF, BMP, TXT
- Shows processing progress and results
- Displays: pages, chunks, extraction confidence

### 📝 Generate
- Select documents to query against
- Choose draft type (case summary, title review, memo, etc.)
- Enter query in natural language
- View generated draft with:
  - Confidence score
  - Citations (expandable)
  - Retrieved chunks (expandable)
  - Grounding warnings (if insufficient evidence)

### ✏️ Edit
- Side-by-side original vs edited view
- Submit corrections for learning
- Optional edit reason

### 📊 Learning
- View edit statistics
- Draft type distribution chart
- Recent edits with quality scores
- Effectiveness metrics

## API Endpoints

### Documents
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/documents/upload` | Upload and process a document |
| `GET` | `/api/documents` | List all documents |
| `GET` | `/api/documents/{id}` | Get document details |
| `GET` | `/api/documents/{id}/chunks` | Get document chunks |
| `DELETE` | `/api/documents/{id}` | Delete a document |

### Drafts
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/drafts/generate` | Generate a grounded draft |
| `GET` | `/api/drafts/{id}` | Get a generated draft |
| `GET` | `/api/drafts/{id}/formatted` | Get draft with inline citations |
| `GET` | `/api/drafts/types/available` | List available draft types |

### Edits (Learning)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/edits/submit` | Submit an operator edit |
| `GET` | `/api/edits/history` | View edit history |
| `GET` | `/api/edits/{id}` | Get specific edit |
| `GET` | `/api/edits/effectiveness` | Get learning metrics |

## Sample Usage (CLI)

### Upload a document
```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@sample_docs/case_facts_001.txt"
```

### Generate a draft
```bash
curl -X POST http://localhost:8000/api/drafts/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize the Smith v. Johnson case",
    "draft_type": "case_fact_summary"
  }'
```

### Submit an edit
```bash
curl -X POST http://localhost:8000/api/edits/submit \
  -H "Content-Type: application/json" \
  -d '{
    "draft_id": "<draft_id>",
    "original_text": "The plaintiff purchased the property.",
    "edited_text": "The plaintiff, Martha Smith, purchased the property at 456 Oak Avenue in August 2020.",
    "edit_reason": "Added specific details"
  }'
```

## Project Structure

```
nerdfarm/
├── app/
│   ├── main.py                  # FastAPI application
│   ├── config.py                # Settings and configuration
│   ├── api/
│   │   ├── documents.py         # Document endpoints
│   │   ├── drafts.py            # Draft generation endpoints
│   │   └── edits.py             # Edit/learning endpoints
│   ├── document_processor/
│   │   ├── extractor.py         # Document extraction logic
│   │   ├── vision_processor.py  # GPT-4o Vision processing
│   │   └── schemas.py           # Data structures
│   ├── retrieval/
│   │   ├── store.py             # ChromaDB vector store
│   │   ├── retriever.py         # High-level retrieval
│   │   ├── embeddings.py        # Embedding client
│   │   ├── bm25_store.py        # BM25 keyword search
│   │   ├── hybrid_search.py     # RRF fusion
│   │   └── reranker.py          # Cross-encoder reranking
│   ├── generation/
│   │   ├── drafter.py           # Draft generation
│   │   ├── llm.py               # LLM client
│   │   └── prompts.py           # Prompt templates
│   ├── learning/
│   │   └── simple_edit_store.py # Edit storage & retrieval
│   └── persistence/
│       └── stores.py            # File-backed stores
├── frontend.py                  # Streamlit UI
├── sample_docs/                 # Sample legal documents
├── sample_outputs/              # Example generated drafts
└── docs/
    ├── ARCHITECTURE.md          # Detailed architecture
    └── EVALUATION.md            # Evaluation analysis
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Key Features

### Document Processing
- ✅ PDF, PNG, JPG, JPEG, TIFF, BMP, TXT support
- ✅ Vision-based extraction (GPT-4o) for images and scanned PDFs
- ✅ Digital PDF text extraction (PyMuPDF)
- ✅ Table detection and markdown conversion
- ✅ Signature and stamp detection
- ✅ Structured data extraction (parties, dates, amounts, case IDs)
- ✅ Unclear text marking: `[unclear: description]`

### Retrieval & Grounding
- ✅ Hybrid search (Vector + BM25 with RRF fusion)
- ✅ Cross-encoder reranking for relevance
- ✅ Citations with source document and page
- ✅ Insufficient evidence flagging
- ✅ Retrieved chunks visibility

### Learning System
- ✅ Quality gate for edit storage (rejects trivial edits)
- ✅ Deduplication of similar edits
- ✅ Weighted retrieval by recency/quality/validation
- ✅ Few-shot learning from past corrections
- ✅ Effectiveness metrics tracking

## Design Decisions

### Vision Model vs Traditional OCR
**Choice:** GPT-4o Vision via OpenRouter

For legal documents, accuracy and structure preservation are more important than cost. Vision models handle tables, handwriting, and layout better than traditional OCR.

### ChromaDB vs Cloud Vector DBs
**Choice:** ChromaDB (local, persistent)

For an internal tool with <10K documents, ChromaDB provides zero setup, no cloud costs, and survives restarts.

### Few-Shot Learning vs Fine-Tuning
**Choice:** Few-shot with RAG

Legal drafting preferences change frequently. Few-shot allows immediate adaptation without retraining, and our quality gate ensures only meaningful edits are stored.

## Limitations

- **API dependency:** Requires OpenRouter API key
- **Single-user scope:** No authentication or multi-tenancy
- **Chunking:** Character-based (500 chars), not token-aware
- **Scale:** Designed for single-machine deployment

## Documentation

- [Architecture Details](docs/ARCHITECTURE.md) - Complete technical architecture
- [Evaluation Analysis](docs/EVALUATION.md) - Evaluation approach and results

## License

MIT
