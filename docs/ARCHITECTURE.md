# NerdFarm Architecture

## Directory Structure

```
nerdfarm/
├── app/
│   ├── main.py                  # FastAPI app, route registration
│   ├── config.py                # Pydantic settings from .env
│   ├── api/
│   │   ├── documents.py         # POST /upload, GET /, GET /{id}, DELETE /{id}
│   │   ├── drafts.py            # POST /generate, GET /{id}, GET /types/available
│   │   └── edits.py             # POST /submit, GET /history, GET /effectiveness
│   ├── document_processor/
│   │   ├── extractor.py         # DocumentExtractor class, routes by file extension
│   │   ├── vision_processor.py  # VisionProcessor class, GPT-4o-mini API calls
│   │   └── schemas.py           # ExtractedDocument, Page, DocumentMetadata dataclasses
│   ├── retrieval/
│   │   ├── store.py             # VectorStore class, ChromaDB operations
│   │   ├── retriever.py         # Retriever class, orchestrates hybrid search
│   │   ├── embeddings.py        # EmbeddingClient class, OpenRouter API calls
│   │   ├── bm25_store.py        # BM25Store class, rank_bm25 implementation
│   │   ├── hybrid_search.py     # reciprocal_rank_fusion() function
│   │   └── reranker.py          # CrossEncoderReranker class, sentence-transformers
│   ├── generation/
│   │   ├── drafter.py           # DraftGenerator class, RAG pipeline
│   │   ├── llm.py               # LLMClient class, OpenRouter chat completions
│   │   └── prompts.py           # build_draft_prompt(), DraftType enum
│   ├── learning/
│   │   ├── simple_edit_store.py # SimpleEditStore, QualityGate classes
│   │   └── __init__.py          # format_examples_for_prompt() function
│   └── persistence/
│       └── stores.py            # DocumentStore, DraftStore JSON file persistence
├── frontend.py                  # Streamlit app, 5 pages
├── tests/                       # pytest test files
├── sample_docs/                 # .txt sample legal documents
├── sample_outputs/              # .md example outputs
├── uploads/                     # Uploaded files (gitignored)
├── chroma_data/                 # ChromaDB SQLite files (gitignored)
└── .env                         # OPENROUTER_API_KEY
```

## Data Models

### ExtractedDocument
```python
@dataclass
class ExtractedDocument:
    id: str                    # UUID
    filename: str              # Original filename
    pages: list[Page]          # List of Page objects
    raw_text: str              # Concatenated page text
    metadata: DocumentMetadata
    structured_data: dict | None  # parties, dates, amounts, tables, etc.

@dataclass
class Page:
    page_num: int
    text: str
    confidence: float          # 0.0-1.0
    has_unclear: bool          # True if [unclear: ...] markers present

@dataclass
class DocumentMetadata:
    total_pages: int
    avg_confidence: float
    file_size: int             # bytes
    file_type: str             # .pdf, .png, .txt, etc.
```

### DraftOutput
```python
@dataclass
class DraftOutput:
    draft_id: str
    content: str               # Generated text with [Source: ...] citations
    citations: list[Citation]
    confidence: float          # 0.0-1.0, based on retrieval scores
    generated_at: datetime
    draft_type: str
    query: str
    retrieved_chunks: list[RetrievedChunkInfo]
    is_grounded: bool          # False if insufficient evidence
    grounding_warning: str     # Empty if grounded, warning message if not

@dataclass
class Citation:
    text: str                  # First 200 chars of chunk
    source_doc: str            # Filename
    page: int
    chunk_id: str
```

### EditExample (ChromaDB)
```python
# Stored in ChromaDB "edit_examples" collection
# Metadata fields:
{
    "before": str,              # Original generated text
    "after": str,               # Operator's corrected version
    "reason": str | None,       # Optional edit reason
    "draft_type": str,          # "global" or specific type
    "timestamp": str,           # ISO format
    "quality_score": float,     # 0.0-1.0 from QualityGate
    "times_seen": int,          # Deduplication counter
    "times_retrieved": int,     # How often used in generation
    "times_accepted": int,      # Positive feedback count
}
```

## API Endpoints

### POST /api/documents/upload
- Accepts: multipart/form-data with `file` field
- Supported types: .pdf, .png, .jpg, .jpeg, .tiff, .bmp, .txt
- Max size: 50MB (configurable via MAX_UPLOAD_SIZE)
- Returns:
```json
{
    "document_id": "uuid",
    "filename": "original.pdf",
    "total_pages": 3,
    "avg_confidence": 0.95,
    "chunks_created": 12,
    "has_unclear_sections": false,
    "structured_data": {"parties": [...], "dates": [...]}
}
```

### POST /api/drafts/generate
- Accepts: application/json
```json
{
    "query": "string",
    "draft_type": "case_fact_summary",
    "document_ids": ["uuid1", "uuid2"]  // optional filter
}
```
- Returns:
```json
{
    "draft_id": "uuid",
    "content": "generated text...",
    "citations": [{"text": "...", "source_doc": "...", "page": 1}],
    "confidence": 0.85,
    "draft_type": "case_fact_summary",
    "edit_examples_used": 2,
    "is_grounded": true,
    "grounding_warning": "",
    "retrieved_chunks": [{"text": "...", "score": 0.82, "source_doc": "..."}]
}
```

### POST /api/edits/submit
```json
{
    "draft_id": "uuid",
    "original_text": "string",
    "edited_text": "string",
    "edit_reason": "string or null",
    "draft_type": "string"
}
```
- Returns:
```json
{
    "edit_id": "uuid",
    "quality_rejected": false,
    "duplicate_of": null,
    "message": "Edit stored"
}
```

## Document Processing Pipeline

### File Type Routing (extractor.py:62-94)
```
.pdf  → _extract_pdf_async()
         → Try PyMuPDF direct extraction (lines 137-155)
         → If no text or low quality, fallback to _extract_pdf_via_vision()
         → Convert to images with pdf2image, process each page

.png/.jpg/.jpeg/.tiff/.bmp → _extract_image_async()
                              → VisionProcessor.process_image()

.txt  → _extract_text_file()
         → Direct file.read(), confidence=1.0
```

### Vision Processing (vision_processor.py)
```python
async def process_image(self, image: PIL.Image, page_num: int) -> tuple[Page, dict]:
    # 1. Upscale if width < 1500px
    if image.width < 1500:
        scale = 1500 / image.width
        image = image.resize((1500, int(image.height * scale)))

    # 2. Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode()

    # 3. Call OpenRouter API
    response = await httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    {"type": "text", "text": VISION_PROMPT}
                ]
            }]
        }
    )

    # 4. Parse JSON response
    result = json.loads(response.json()["choices"][0]["message"]["content"])
    # Returns: full_text, tables[], signatures[], stamps_seals[],
    #          parties[], dates[], amounts[], case_ids[], document_type, confidence
```

### Chunking (store.py:131-173)
```python
def _chunk_document_with_pages(self, doc: ExtractedDocument, chunk_size: int = 500):
    chunks_with_pages = []
    for page in doc.pages:
        paragraphs = page.text.split("\n\n")
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if current_chunk and len(current_chunk) + len(para) > chunk_size:
                chunks_with_pages.append((current_chunk.strip(), page.page_num))
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        if current_chunk.strip():
            chunks_with_pages.append((current_chunk.strip(), page.page_num))
    return chunks_with_pages
```

## Retrieval Pipeline

### Hybrid Search (retriever.py:25-70)
```python
async def retrieve(self, query: str, k: int = 5, doc_ids: list[str] | None = None):
    # 1. Vector search
    vector_results = await self.vector_store.search(query, k=k*2, doc_ids=doc_ids)

    # 2. BM25 search
    bm25_results = self.bm25_store.search(query, k=k*2)

    # 3. Reciprocal Rank Fusion
    fused = self.hybrid_search.reciprocal_rank_fusion(
        vector_results, bm25_results, k=60
    )

    # 4. Cross-encoder reranking
    reranked = self.reranker.rerank(query, fused, top_k=k)

    return [RetrievedChunk(...) for ... in reranked]
```

### RRF Formula (hybrid_search.py)
```python
def reciprocal_rank_fusion(results_list: list[list], k: int = 60) -> list:
    scores = {}
    for results in results_list:
        for rank, result in enumerate(results):
            chunk_id = result["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### Cross-Encoder Reranking (reranker.py)
```python
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5):
        pairs = [(query, chunk["content"]) for chunk in chunks]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in ranked[:top_k]]
```

## Grounding Validation (drafter.py:174-211)

```python
MIN_RETRIEVAL_CONFIDENCE = 0.2  # Per-chunk minimum
MIN_RELEVANT_CHUNKS = 1        # At least 1 chunk required
MIN_AVG_SIMILARITY = 0.25      # Average similarity threshold

def _validate_grounding(self, chunks: list[RetrievedChunk]) -> dict:
    if not chunks:
        return {"is_grounded": False, "reason": "No relevant documents found"}

    if len(chunks) < MIN_RELEVANT_CHUNKS:
        return {"is_grounded": False, "reason": f"Only {len(chunks)} chunk(s) found"}

    avg_similarity = sum(c.similarity_score for c in chunks) / len(chunks)
    if avg_similarity < MIN_AVG_SIMILARITY:
        return {"is_grounded": False, "reason": f"Low relevance: {avg_similarity:.2f}"}

    high_quality = [c for c in chunks if c.similarity_score >= MIN_RETRIEVAL_CONFIDENCE]
    if not high_quality:
        return {"is_grounded": False, "reason": "No chunks meet minimum threshold"}

    return {"is_grounded": True, "reason": ""}
```

## Learning System

### QualityGate (simple_edit_store.py:49-95)
```python
class QualityGate:
    MIN_LENGTH = 20
    MIN_CHANGE_RATIO = 0.15  # 15% of text must change

    def is_meaningful(self, before: str, after: str) -> tuple[bool, float]:
        if len(before) < self.MIN_LENGTH or len(after) < self.MIN_LENGTH:
            return False, 0.0

        if before.strip() == after.strip():
            return False, 0.0

        # Calculate change using difflib
        matcher = difflib.SequenceMatcher(None, before, after)
        change_ratio = 1 - matcher.ratio()

        if change_ratio < self.MIN_CHANGE_RATIO:
            return False, 0.0

        quality_score = min(change_ratio * 2, 1.0)
        return True, quality_score
```

### Edit Retrieval with Weights (simple_edit_store.py:180-230)
```python
async def get_similar_edits(self, query: str, draft_type: str, k: int = 3):
    query_embedding = await self.embedding_client.embed_single(query)

    results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=k * 3,
        where={"draft_type": {"$in": ["global", draft_type]}}
    )

    weighted_edits = []
    now = datetime.now(timezone.utc)

    for i, meta in enumerate(results["metadatas"][0]):
        distance = results["distances"][0][i]
        semantic_score = 1 - distance

        # Recency multiplier
        timestamp = datetime.fromisoformat(meta["timestamp"])
        age_days = (now - timestamp).days
        if age_days < 7:
            recency_mult = 1.2
        elif age_days < 30:
            recency_mult = 1.0
        else:
            recency_mult = 0.8

        quality_score = meta.get("quality_score", 0.5)
        times_seen = meta.get("times_seen", 1)
        usage_mult = min(1 + (times_seen * 0.1), 1.5)

        weighted_score = (
            semantic_score * 0.5 +
            quality_score * 0.3 * recency_mult +
            (usage_mult - 1) * 0.2
        )

        if semantic_score > 0.6:
            weighted_edits.append({...})

    return sorted(weighted_edits, key=lambda x: x["score"], reverse=True)[:k]
```

## Configuration

### .env
```
OPENROUTER_API_KEY=sk-or-xxxxxxxxxx
CHROMA_PERSIST_DIR=./chroma_data
UPLOAD_DIR=./uploads
MAX_UPLOAD_SIZE=52428800
```

### config.py
```python
class Settings(BaseSettings):
    openrouter_api_key: str
    chroma_persist_dir: str = "./chroma_data"
    upload_dir: str = "./uploads"
    max_upload_size: int = 50 * 1024 * 1024
    vision_model: str = "openai/gpt-4o-mini"
    llm_model: str = "openai/gpt-4o"
    embedding_model: str = "openai/text-embedding-3-small"
    use_vision_for_images: bool = True
    use_hybrid_search: bool = True
    use_reranker: bool = True
```

## Error Codes

| HTTP Status | Condition |
|-------------|-----------|
| 400 | Invalid file type, invalid draft_type |
| 404 | Document/draft/edit not found |
| 413 | File exceeds MAX_UPLOAD_SIZE |
| 500 | Processing failure, LLM API error |
