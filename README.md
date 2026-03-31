# NerdFarm

Document processing system for legal documents. Extracts text from PDFs/images, indexes in ChromaDB, generates drafts with source citations.

## Requirements

- Python 3.12+
- uv package manager
- Poppler (PDF processing): `apt install poppler-utils`
- OpenRouter API key

## Setup

```bash
cd nerdfarm
cp .env.example .env
# Edit .env: OPENROUTER_API_KEY=sk-or-xxxxx
uv sync
```

## Running

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

## API

### Documents

| Method | Endpoint | Body | Response |
|--------|----------|------|----------|
| POST | /api/documents/upload | multipart file | `{document_id, filename, chunks_created, avg_confidence}` |
| GET | /api/documents | - | `[{document_id, filename, total_pages, chunks_created, avg_confidence}]` |
| GET | /api/documents/{id} | - | `{document_id, filename, metadata, pages}` |
| DELETE | /api/documents/{id} | - | `{deleted: true, chunks_removed}` |

### Drafts

| Method | Endpoint | Body | Response |
|--------|----------|------|----------|
| POST | /api/drafts/generate | `{query, draft_type, document_ids?}` | `{draft_id, content, citations, confidence, is_grounded, retrieved_chunks}` |
| GET | /api/drafts/{id} | - | `{draft_id, content, citations, confidence, draft_type, query}` |
| GET | /api/drafts/types/available | - | `{draft_types: [{value, name, description}]}` |

### Edits

| Method | Endpoint | Body | Response |
|--------|----------|------|----------|
| POST | /api/edits/submit | `{draft_id, original_text, edited_text, edit_reason?, draft_type}` | `{edit_id, quality_rejected, message}` |
| GET | /api/edits/history?limit=50 | - | `{edits: [{edit_id, before, after, reason, quality_score}]}` |
| GET | /api/edits/effectiveness | - | `{total_edits, avg_quality_score}` |

## CLI Examples

```bash
# Upload
curl -X POST http://localhost:8000/api/documents/upload -F "file=@sample_docs/case_facts_001.txt"

# Generate
curl -X POST http://localhost:8000/api/drafts/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the case", "draft_type": "case_fact_summary"}'

# Submit edit
curl -X POST http://localhost:8000/api/edits/submit \
  -H "Content-Type: application/json" \
  -d '{"draft_id": "xxx", "original_text": "A", "edited_text": "A with details", "draft_type": "case_fact_summary"}'
```

## Project Structure

```
app/
├── main.py              # FastAPI routes
├── config.py            # Settings from .env
├── api/
│   ├── documents.py     # Upload, list, delete documents
│   ├── drafts.py        # Generate, retrieve drafts
│   └── edits.py         # Submit edits, history
├── document_processor/
│   ├── extractor.py     # Routes .pdf/.txt/.jpg to appropriate processor
│   ├── vision_processor.py  # GPT-4o-mini API calls
│   └── schemas.py       # ExtractedDocument, Page dataclasses
├── retrieval/
│   ├── store.py         # ChromaDB operations
│   ├── retriever.py     # Hybrid search orchestration
│   ├── embeddings.py    # text-embedding-3-small API calls
│   ├── bm25_store.py    # rank_bm25 keyword search
│   ├── hybrid_search.py # RRF fusion
│   └── reranker.py      # ms-marco-MiniLM cross-encoder
├── generation/
│   ├── drafter.py       # DraftGenerator class
│   ├── llm.py           # OpenRouter chat completions
│   └── prompts.py       # Prompt templates per draft type
├── learning/
│   └── simple_edit_store.py  # QualityGate, SimpleEditStore
└── persistence/
    └── stores.py        # JSON file persistence
```

## Tests

```bash
uv run pytest tests/ -v
```

26 tests cover: QualityGate, EditExample, VectorStore chunking, DraftOutput, Citations.

## Configuration

`.env`:
```
OPENROUTER_API_KEY=sk-or-xxxxx
CHROMA_PERSIST_DIR=./chroma_data
UPLOAD_DIR=./uploads
MAX_UPLOAD_SIZE=52428800
```

`app/config.py`:
```python
class Settings:
    vision_model: str = "openai/gpt-4o-mini"
    llm_model: str = "openai/gpt-4o"
    embedding_model: str = "openai/text-embedding-3-small"
    use_hybrid_search: bool = True
    use_reranker: bool = True
```

## Draft Types

| Value | Description |
|-------|-------------|
| case_fact_summary | Case facts and parties |
| title_review_summary | Property title analysis |
| notice_summary | Legal notice summary |
| document_checklist | Required documents |
| internal_memo | Professional memo |

## Grounding Thresholds

```python
MIN_RETRIEVAL_CONFIDENCE = 0.2  # Per-chunk minimum
MIN_RELEVANT_CHUNKS = 1         # Minimum chunks required
MIN_AVG_SIMILARITY = 0.25       # Average across chunks
```

If thresholds not met, returns `is_grounded: false` with `grounding_warning` message.

## Learning System

### Quality Gate
Rejects edits with <15% text change, <20 character length, or whitespace/punctuation only changes.

### Edit Storage
Stored in ChromaDB `edit_examples` collection with embeddings. Metadata includes: quality_score, timestamp, times_seen, draft_type.

### Retrieval Weights
```
score = 0.5 * semantic + 0.3 * quality * recency + 0.2 * validation
```
Recency: <7 days = 1.2x, 7-30 days = 1.0x, >30 days = 0.8x

## Limitations

- No authentication
- Character-based chunking (500 chars)
- Single-machine ChromaDB
- No async job queue

## Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Data models, API contracts, implementation details
- [EVALUATION.md](docs/EVALUATION.md) - Test results, performance metrics
