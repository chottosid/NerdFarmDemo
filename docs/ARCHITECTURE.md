# NerdFarm Architecture

A complete technical overview of the Document Understanding & Grounded Drafting System.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERACTION LAYER                                  │
│                                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                     Streamlit Frontend (frontend.py)                         │   │
│   │                                                                              │   │
│   │   Pages: Home | Upload | Generate | Edit | Learning                          │   │
│   │   - Document upload with drag-and-drop                                       │   │
│   │   - Draft generation with type selection                                     │   │
│   │   - Side-by-side edit comparison                                             │   │
│   │   - Learning statistics visualization                                        │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                           │                                          │
└───────────────────────────────────────────┼──────────────────────────────────────────┘
                                            │ HTTP/JSON
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER (FastAPI)                                     │
│                                                                                      │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐                 │
│   │  /api/documents  │  │   /api/drafts    │  │   /api/edits     │                 │
│   │                  │  │                  │  │                  │                 │
│   │  • upload        │  │  • generate      │  │  • submit        │                 │
│   │  • list          │  │  • get           │  │  • history       │                 │
│   │  • get           │  │  • formatted     │  │  • similar       │                 │
│   │  • delete        │  │  • types         │  │  • effectiveness │                 │
│   └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘                 │
│            │                     │                     │                            │
└────────────┼─────────────────────┼─────────────────────┼────────────────────────────┘
             │                     │                     │
             ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              CORE SERVICES                                           │
│                                                                                      │
│   ┌──────────────────────────────────────────────────────────────────────────────┐  │
│   │                        DOCUMENT PROCESSING PIPELINE                           │  │
│   │                                                                               │  │
│   │   ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐              │  │
│   │   │ File Upload │───▶│ DocumentExtractor│───▶│ VisionProcessor │              │  │
│   │   │ (PDF/IMG/   │    │                 │    │ (GPT-4o Vision) │              │  │
│   │   │  TXT)       │    │ Routes by type: │    │                 │              │  │
│   │   └─────────────┘    │ • PDF → Digital │    │ • Text extract  │              │  │
│   │                      │   or Vision     │    │ • Table detect  │              │  │
│   │                      │ • Image → Vision│    │ • Signature det │              │  │
│   │                      │ • TXT → Direct  │    │ • Structured    │              │  │
│   │                      └─────────────────┘    │   data extract  │              │  │
│   │                                             └────────┬────────┘              │  │
│   │                                                      │                       │  │
│   │                                                      ▼                       │  │
│   │                                             ExtractedDocument                 │  │
│   │                                             (pages, text, metadata)          │  │
│   └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│   ┌──────────────────────────────────────────────────────────────────────────────┐  │
│   │                        RETRIEVAL & STORAGE LAYER                              │  │
│   │                                                                               │  │
│   │   ┌─────────────────────────────────────────────────────────────────────┐    │  │
│   │   │                        VectorStore (ChromaDB)                        │    │  │
│   │   │                                                                      │    │  │
│   │   │   ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐   │    │  │
│   │   │   │   Documents   │  │  Embeddings   │  │     Metadata          │   │    │  │
│   │   │   │   Collection  │  │  (text-       │  │  • source_doc_id      │   │    │  │
│   │   │   │               │  │  embedding-3- │  │  • filename           │   │    │  │
│   │   │   │   Chunked     │  │  small)       │  │  • page_num           │   │    │  │
│   │   │   │   documents   │  │               │  │  • chunk_index        │   │    │  │
│   │   │   └───────────────┘  └───────────────┘  └───────────────────────┘   │    │  │
│   │   └─────────────────────────────────────────────────────────────────────┘    │  │
│   │                                                                               │  │
│   │   ┌─────────────────────────────────────────────────────────────────────┐    │  │
│   │   │                     Hybrid Search Pipeline                           │    │  │
│   │   │                                                                      │    │  │
│   │   │   Query ──▶ ┌───────────────┐  ┌───────────────┐                    │    │  │
│   │   │             │ Vector Search │  │  BM25 Search  │                    │    │  │
│   │   │             │ (Embedding)   │  │  (Keyword)    │                    │    │  │
│   │   │             └───────┬───────┘  └───────┬───────┘                    │    │  │
│   │   │                     │                  │                            │    │  │
│   │   │                     └────────┬─────────┘                            │    │  │
│   │   │                              ▼                                      │    │  │
│   │   │                    ┌───────────────────┐                            │    │  │
│   │   │                    │ Reciprocal Rank   │                            │    │  │
│   │   │                    │ Fusion (RRF)      │                            │    │  │
│   │   │                    └─────────┬─────────┘                            │    │  │
│   │   │                              │                                      │    │  │
│   │   │                              ▼                                      │    │  │
│   │   │                    ┌───────────────────┐                            │    │  │
│   │   │                    │ Cross-Encoder     │                            │    │  │
│   │   │                    │ Reranking         │                            │    │  │
│   │   │                    │ (ms-marco-MiniLM) │                            │    │  │
│   │   │                    └─────────┬─────────┘                            │    │  │
│   │   │                              │                                      │    │  │
│   │   │                              ▼                                      │    │  │
│   │   │                      Top-K Chunks                                    │    │  │
│   │   └─────────────────────────────────────────────────────────────────────┘    │  │
│   └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│   ┌──────────────────────────────────────────────────────────────────────────────┐  │
│   │                        GENERATION PIPELINE                                    │  │
│   │                                                                               │  │
│   │   ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐              │  │
│   │   │   Query     │───▶│    Retriever    │───▶│ DraftGenerator  │              │  │
│   │   │             │    │                 │    │                 │              │  │
│   │   │ + Doc IDs   │    │ 1. Hybrid search│    │ 1. Build prompt │              │  │
│   │   │ + Draft Type│    │ 2. Rerank       │    │ 2. Call LLM     │              │  │
│   │   │             │    │ 3. Validate     │    │ 3. Extract cites│              │  │
│   │   └─────────────┘    └─────────────────┘    │ 4. Score conf   │              │  │
│   │                                             └────────┬────────┘              │  │
│   │                                                      │                       │  │
│   │   ┌──────────────────────────────────────────────────┼───────────────────┐  │  │
│   │   │              Learning Integration                │                   │  │  │
│   │   │                                                  ▼                   │  │  │
│   │   │   ┌─────────────────┐    ┌─────────────────────────────────────┐     │  │  │
│   │   │   │ SimpleEditStore │───▶│ Get similar past edits for few-shot │     │  │  │
│   │   │   │ (ChromaDB)      │    │ learning before generation          │     │  │  │
│   │   │   │                 │    └─────────────────────────────────────┘     │  │  │
│   │   │   │ • Before/After  │                                                   │  │  │
│   │   │   │ • Quality score │                                                   │  │  │
│   │   │   │ • Recency weight│                                                   │  │  │
│   │   │   │ • Validation    │                                                   │  │  │
│   │   │   └─────────────────┘                                                   │  │  │
│   │   └──────────────────────────────────────────────────────────────────────┘  │  │
│   └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│   ┌──────────────────────────────────────────────────────────────────────────────┐  │
│   │                        LLM CLIENT (OpenRouter)                                │  │
│   │                                                                               │  │
│   │   ┌───────────────────────────────────────────────────────────────────────┐  │  │
│   │   │                          GPT-4o via OpenRouter                         │  │  │
│   │   │                                                                        │  │  │
│   │   │   Vision Tasks:              Generation Tasks:                         │  │  │
│   │   │   • openai/gpt-4o-mini       • openai/gpt-4o                           │  │  │
│   │   │   • Text extraction          • Draft generation                        │  │  │
│   │   │   • Table detection          • Few-shot learning                       │  │  │
│   │   │   • Signature detection      • Citation formatting                     │  │  │
│   │   └───────────────────────────────────────────────────────────────────────┘  │  │
│   └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Document Processing Pipeline

**Entry Point:** `app/api/documents.py`

**Flow:**
1. File upload validation (type, size)
2. Route to appropriate extractor based on file type
3. Extract text and structured data
4. Chunk and embed for vector storage
5. Persist metadata

**File Type Handling:**

| Type | Processor | Method | Confidence |
|------|-----------|--------|------------|
| PDF (digital) | PyMuPDF | Direct text extraction | 99% |
| PDF (scanned) | VisionProcessor | GPT-4o Vision | 85-98% |
| PNG/JPG/JPEG | VisionProcessor | GPT-4o Vision | 85-98% |
| TXT | Direct read | File read | 100% |

**Vision Processor** (`app/document_processor/vision_processor.py`):
- Upscales images < 1500px width
- Sends to GPT-4o Vision with structured prompt
- Extracts: text, tables (markdown), signatures, stamps, parties, dates, amounts, case IDs
- Returns confidence score and unclear section markers

### 2. Retrieval Layer

**Components:**
- `VectorStore` - ChromaDB collection management
- `EmbeddingClient` - OpenAI text-embedding-3-small
- `BM25Store` - Keyword search using rank_bm25
- `HybridSearch` - RRF fusion of vector + keyword results
- `Reranker` - Cross-encoder ms-marco-MiniLM-L-6-v2

**Search Pipeline:**
```python
# 1. Vector search (semantic)
vector_results = vector_store.search(query, k=10)

# 2. BM25 search (keyword)
bm25_results = bm25_store.search(query, k=10)

# 3. Reciprocal Rank Fusion
fused = hybrid_search.fuse(vector_results, bm25_results, k=60)

# 4. Cross-encoder reranking
reranked = reranker.rerank(query, fused, top_k=5)
```

**Citation Tracking:**
Every chunk includes metadata:
- `source_doc_id` - Original document UUID
- `filename` - Human-readable filename
- `page_num` - Page number in source
- `chunk_index` - Position in chunk sequence

### 3. Draft Generation

**Draft Types:**
| Type | Description | Use Case |
|------|-------------|----------|
| `case_fact_summary` | Summary of case facts and events | Litigation support |
| `title_review_summary` | Property title analysis | Real estate |
| `notice_summary` | Legal notice summary | Compliance |
| `document_checklist` | Required documents list | Transaction prep |
| `internal_memo` | Professional memo format | Internal communication |

**Generation Flow:**
1. Retrieve similar past edits for few-shot learning
2. Hybrid search for relevant document chunks
3. Validate grounding (minimum chunks, similarity thresholds)
4. Build prompt with context, examples, and query
5. Generate with GPT-4o
6. Extract citations and calculate confidence

**Grounding Validation:**
```python
MIN_RETRIEVAL_CONFIDENCE = 0.2  # Per-chunk threshold
MIN_RELEVANT_CHUNKS = 1        # Minimum chunks needed
MIN_AVG_SIMILARITY = 0.25      # Average similarity threshold
```

If validation fails, returns warning message with suggestions instead of draft.

### 4. Learning System

**Architecture:**
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Draft Generated│────▶│  Operator Edit   │────▶│  Quality Gate   │
│                 │     │  (Submit)        │     │  (Filter)       │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌─────────────────────────────────┘
                        │
                        ▼
               ┌──────────────────┐     ┌───────────────────┐
               │  Quality Gate    │────▶│  Pass: Embed &    │
               │  Checks:         │     │  Store in ChromaDB│
               │  • >15% change   │     └───────────────────┘
               │  • Not whitespace│
               │  • Not punctuation│    ┌───────────────────┐
               └──────────────────┘────▶│  Reject: Skip     │
                                        └───────────────────┘
```

**Weighted Retrieval for Edits:**
```python
score = (
    0.5 * semantic_similarity +      # How similar to current query
    0.3 * quality_score * recency +  # Quality with recency boost
    0.2 * validation_score           # Past acceptance rate
)

# Recency multipliers:
# < 7 days: 1.2x boost
# 7-30 days: 1.0x (no change)
# > 30 days: 0.8x decay
```

**Few-Shot Integration:**
```python
# Before generation:
similar_edits = edit_store.get_similar_edits(query, draft_type, k=3)
few_shot_prompt = format_examples_for_prompt(similar_edits)

# Injected into system prompt:
"Learn from these past improvements:
Example 1: Before → After (reason)
Example 2: Before → After (reason)
..."
```

## Data Flow Diagrams

### Document Upload Flow
```
User uploads file
       │
       ▼
┌──────────────────┐
│ Validate type    │──── Invalid ───▶ 400 Error
│ and size         │
└────────┬─────────┘
         │ Valid
         ▼
┌──────────────────┐
│ Save to uploads/ │
│ Generate doc_id  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ DocumentExtractor│
│ Route by type    │
└────────┬─────────┘
         │
    ┌────┴────┬─────────┐
    ▼         ▼         ▼
  PDF       Image      TXT
    │         │         │
    ▼         ▼         ▼
Digital   Vision     Direct
or Vision Processor   Read
    │         │         │
    └────┬────┴─────────┘
         │
         ▼
┌──────────────────┐
│ ExtractedDocument│
│ - pages[]        │
│ - raw_text       │
│ - structured_data│
│ - metadata       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Chunk & Embed    │
│ Store in ChromaDB│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Persist metadata │
│ to .doc_store/   │
└────────┬─────────┘
         │
         ▼
   Return doc_id
   + processing stats
```

### Draft Generation Flow
```
User submits query + draft_type
              │
              ▼
┌─────────────────────────┐
│ Get similar past edits  │
│ for few-shot learning   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Hybrid Search:          │
│ 1. Vector similarity    │
│ 2. BM25 keyword match   │
│ 3. RRF fusion           │
│ 4. Cross-encoder rerank │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Validate Grounding:     │
│ • Min chunks met?       │
│ • Avg similarity OK?    │
│ • Per-chunk scores OK?  │
└───────────┬─────────────┘
            │
       ┌────┴────┐
       │         │
    Failed     Passed
       │         │
       ▼         ▼
┌──────────┐ ┌──────────────────┐
│ Return   │ │ Build prompt:    │
│ Warning  │ │ • System prompt  │
│ Message  │ │ • Few-shot edits │
└──────────┘ │ • Context chunks │
             │ • User query     │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Generate with    │
             │ GPT-4o           │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Post-process:    │
             │ • Extract cites  │
             │ • Calc confidence│
             │ • Persist draft  │
             └────────┬─────────┘
                      │
                      ▼
               Return DraftOutput
```

## Configuration

**Environment Variables** (`.env`):
```bash
OPENROUTER_API_KEY=sk-or-...     # Required: API key for LLM access
CHROMA_PERSIST_DIR=./chroma_data # Vector DB location
UPLOAD_DIR=./uploads             # Uploaded files location
MAX_UPLOAD_SIZE=52428800         # 50MB max file size
```

**Key Settings** (`app/config.py`):
```python
class Settings:
    openrouter_api_key: str
    chroma_persist_dir: str = "./chroma_data"
    upload_dir: str = "./uploads"
    max_upload_size: int = 50 * 1024 * 1024  # 50MB

    # Model selection
    vision_model: str = "openai/gpt-4o-mini"
    llm_model: str = "openai/gpt-4o"
    embedding_model: str = "openai/text-embedding-3-small"

    # Feature flags
    use_vision_for_images: bool = True
    use_hybrid_search: bool = True
    use_reranker: bool = True
```

## Scalability Considerations

**Current Design:**
- Single-machine deployment
- File-backed persistence
- ChromaDB local storage
- No authentication

**Scaling Path:**
1. **Horizontal:** Add load balancer + multiple API instances
2. **Vector DB:** Migrate ChromaDB to Pinecone/Weaviate for distributed search
3. **Storage:** Move uploads to S3/GCS
4. **Queue:** Add Celery/Redis for async processing
5. **Auth:** Add JWT-based authentication

## Error Handling

**Document Processing:**
- Invalid file type → 400 Bad Request
- File too large → 413 Payload Too Large
- Processing failure → 500 with error details
- Unclear text → Marked as `[unclear: description]`

**Draft Generation:**
- No documents → Warning message with suggestions
- Low retrieval quality → Grounding warning, returns partial results
- LLM failure → 500 with retry suggestion

**Learning System:**
- Low-quality edit → Rejected with explanation
- Duplicate edit → Merged with existing, increment count
- Storage failure → Logged, operation continues
