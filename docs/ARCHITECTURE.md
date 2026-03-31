# Architecture

## System Diagram

```
                                    ┌──────────────────────────────────────────┐
                                    │              Streamlit UI                │
                                    │         (frontend.py:410 lines)          │
                                    │   Pages: Home | Upload | Generate | Edit │
                                    └─────────────────┬────────────────────────┘
                                                      │ HTTP
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                  FastAPI Backend                                    │
│                              (app/main.py:55 lines)                                 │
│                                                                                     │
│   ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐        │
│   │   /api/documents    │  │    /api/drafts      │  │    /api/edits       │        │
│   │   (4 endpoints)     │  │    (4 endpoints)    │  │    (5 endpoints)    │        │
│   └──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘        │
└──────────────┼─────────────────────────┼─────────────────────────┼──────────────────┘
               │                         │                         │
               ▼                         ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                 Core Modules                                        │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                        Document Processing                                   │   │
│  │                                                                              │   │
│  │    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐            │   │
│  │    │   Extractor  │─────▶│    Vision    │─────▶│   Extracted  │            │   │
│  │    │              │      │  Processor   │      │   Document   │            │   │
│  │    │ Routes by    │      │              │      │              │            │   │
│  │    │ file type    │      │ GPT-4o-mini  │      │ pages[]      │            │   │
│  │    └──────────────┘      └──────────────┘      │ raw_text     │            │   │
│  │                                                │ metadata     │            │   │
│  │                                                └──────┬───────┘            │   │
│  └───────────────────────────────────────────────────────┼────────────────────┘   │
│                                                          │                          │
│  ┌───────────────────────────────────────────────────────┼────────────────────┐   │
│  │                        Retrieval                       │                    │   │
│  │                                                        ▼                    │   │
│  │    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐            │   │
│  │    │  VectorStore │      │   BM25Store  │      │  HybridSearch│            │   │
│  │    │              │      │              │      │              │            │   │
│  │    │  ChromaDB    │      │  Keyword     │─────▶│  RRF Fusion  │            │   │
│  │    │  Cosine sim  │      │  Matching    │      │              │            │   │
│  │    └──────────────┘      └──────────────┘      └──────┬───────┘            │   │
│  │                                                        │                    │   │
│  │                                                        ▼                    │   │
│  │                                                ┌──────────────┐            │   │
│  │                                                │   Reranker   │            │   │
│  │                                                │              │            │   │
│  │                                                │ Cross-encoder│            │   │
│  │                                                │ ms-marco     │            │   │
│  │                                                └──────────────┘            │   │
│  └────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  ┌────────────────────────────────────────────────────────────────────────────┐   │
│  │                        Generation                                          │   │
│  │                                                                            │   │
│  │    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐           │   │
│  │    │   Retriever  │─────▶│    Drafter   │─────▶│   LLMClient  │           │   │
│  │    │              │      │              │      │              │           │   │
│  │    │ Gets chunks  │      │ Builds prompt│      │ GPT-4o API   │           │   │
│  │    │ + edits      │      │ Validates    │      │ calls        │           │   │
│  │    └──────────────┘      └──────────────┘      └──────────────┘           │   │
│  └────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  ┌────────────────────────────────────────────────────────────────────────────┐   │
│  │                        Learning                                            │   │
│  │                                                                            │   │
│  │    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐           │   │
│  │    │ QualityGate  │─────▶│  EditStore   │─────▶│  Few-shot    │           │   │
│  │    │              │      │              │      │  Retrieval   │           │   │
│  │    │ Filters      │      │ ChromaDB     │      │              │           │   │
│  │    │ trivial edits│      │ embeddings   │      │ Weighted by  │           │   │
│  │    └──────────────┘      └──────────────┘      │ recency/qual │           │   │
│  │                                                └──────────────┘           │   │
│  └────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow: Document Upload

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Upload    │────▶│   Extract   │────▶│    Chunk    │────▶│   Embed &   │
│   File      │     │   Text      │     │   500 chars │     │   Store     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │  PDF → PyMuPDF or   │
               │       Vision        │
               │  Image → Vision     │
               │  TXT → Direct read  │
               └─────────────────────┘
```

## Data Flow: Draft Generation

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Query    │────▶│   Retrieve  │────▶│   Validate  │────▶│   Generate  │
│             │     │   Chunks    │     │  Grounding  │     │   Draft     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                          │                    │
                          ▼                    ▼
               ┌─────────────────────┐   ┌─────────────────────┐
               │  Vector + BM25      │   │  Min 1 chunk        │
               │  RRF Fusion         │   │  Avg sim >= 0.25    │
               │  Cross-encoder      │   │  Per-chunk >= 0.2   │
               │  rerank             │   │                     │
               └─────────────────────┘   └─────────────────────┘
```

## Data Flow: Learning Loop

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Submit    │────▶│   Quality   │────▶│    Embed    │────▶│   Store in  │
│    Edit     │     │    Gate     │     │             │     │  ChromaDB   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │  Reject if:         │
               │  - <15% change      │
               │  - <20 chars        │
               │  - Whitespace only  │
               └─────────────────────┘
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Frontend | Streamlit | Web UI for document upload and draft generation |
| API | FastAPI | REST endpoints with async support |
| Vision | GPT-4o-mini | Text extraction from images/scanned PDFs |
| Generation | GPT-4o | Draft generation via OpenRouter API |
| Embeddings | text-embedding-3-small | 1536-dim vectors for semantic search |
| Vector DB | ChromaDB | Persistent local vector storage |
| Keyword Search | rank_bm25 | BM25 algorithm for exact term matching |
| Reranking | ms-marco-MiniLM-L-6-v2 | Cross-encoder for relevance scoring |

## Storage

| Data | Location | Format |
|------|----------|--------|
| Uploaded files | ./uploads/ | Original format |
| Document metadata | ./uploads/.doc_store/ | JSON files |
| Vector embeddings | ./chroma_data/ | SQLite (ChromaDB) |
| Edit examples | ./chroma_data/ | SQLite (ChromaDB) |
| Drafts | ./uploads/.draft_store/ | JSON files |

## Design Choices

### Why GPT-4o Vision instead of OCR
Traditional OCR (Tesseract) loses table structure, cannot read handwriting, and produces linear text. Vision models understand document layout, extract tables as markdown, detect signatures/stamps, and handle degraded images better.

### Why ChromaDB instead of Pinecone/Weaviate
Internal tool with expected <10K documents. ChromaDB requires zero configuration, persists locally, has no cloud costs, and survives restarts. Cloud vector DBs add complexity without benefit at this scale.

### Why Few-shot Learning instead of Fine-tuning
Legal drafting preferences change frequently. Fine-tuning requires hours of training and produces frozen models. Few-shot learning via retrieved examples adapts immediately to new edit patterns without retraining.

### Why Hybrid Search
Vector search captures semantic meaning but misses exact term matches (case numbers, property addresses). BM25 catches exact terms but misses synonyms. Reciprocal Rank Fusion combines both, then cross-encoder reranking produces final relevance scores.

### Why Quality Gate for Edits
Storing whitespace fixes and typo corrections pollutes the example pool. Only edits with >15% text change teach meaningful patterns. This keeps the few-shot context window focused on substantial improvements.

## Scalability

Current design: Single-machine, file-backed storage, no authentication.

Scaling path if needed:
1. Add load balancer + multiple API workers
2. Migrate ChromaDB to Pinecone/Weaviate for distributed search
3. Move uploads to S3/GCS
4. Add Celery/Redis for async job processing
5. Add JWT authentication for multi-tenant support
