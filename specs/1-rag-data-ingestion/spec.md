# Feature Specification: RAG Data Ingestion & Vectorization

**Feature Branch**: `1-rag-data-ingestion`
**Created**: 2025-12-19
**Status**: Draft
**Input**: User description: "Spec 1 — RAG Data Ingestion & Vectorization: Design and specify a deterministic, repeatable, and idempotent ingestion pipeline that crawls and extracts content from deployed Docusaurus website URLs, filters non-book content, chunks content while preserving document structure, generates vector embeddings using Cohere, stores embeddings in Qdrant Cloud, and enables precise downstream retrieval."

## 1. Scope Definition

### In Scope
- URL crawling and content extraction from deployed Docusaurus websites
- Content filtering to exclude navigation, sidebar, footer, and non-book UI elements
- Text normalization and cleaning
- Content chunking that preserves section hierarchy and logical reading order
- Generation of vector embeddings using Cohere embedding models
- Storage of embeddings and metadata in Qdrant Cloud (Free Tier)
- Implementation of idempotent, deterministic, and repeatable pipeline
- Rich metadata storage including document_id, source_url, page_title, section_heading, chunk_index, and raw_text
- Safe re-run capabilities with duplicate prevention

### Out of Scope
- Retrieval logic or search functionality
- Agent behavior or chatbot implementation
- FastAPI endpoints or API design
- Frontend UI components or user interfaces
- Real-time data synchronization
- Content creation or authoring tools
- Downstream application logic or business processes

## 2. System Inputs & Outputs

### Inputs
- **Website URLs**: List of deployed Docusaurus website URLs containing book content
- **Configuration values**:
  - Cohere API key
  - Qdrant Cloud endpoint and API key
  - Chunking parameters (size, overlap)
  - Filtering rules for content exclusion
- **Environment variables**:
  - COHERE_API_KEY
  - QDRANT_API_KEY
  - QDRANT_URL
  - CHUNK_SIZE
  - CHUNK_OVERLAP

### Outputs
- **Vector records**: Embeddings stored in Qdrant with associated metadata
- **Metadata schema**: Structured data including document_id, source_url, page_title, section_heading, chunk_index, raw_text
- **Qdrant collection state**: Properly configured collection with vectors and payload data
- **Processing logs**: Detailed logs of ingestion process for monitoring and debugging

## 3. Data Flow Architecture

### End-to-end ingestion flow:
1. **URL Discovery**: Identify and validate input URLs for Docusaurus websites
2. **Content Extraction**: Crawl and extract HTML content from URLs
3. **Content Cleaning**: Filter out non-book content (navigation, sidebar, footer)
4. **Text Normalization**: Clean and standardize extracted text
5. **Document Structure Preservation**: Maintain section hierarchy and reading order
6. **Chunking**: Split content into semantically coherent chunks with overlap
7. **Embedding Generation**: Create vector embeddings using Cohere models
8. **Metadata Enrichment**: Add document metadata to each chunk
9. **Storage**: Save vectors and metadata to Qdrant Cloud collection
10. **Validation**: Verify successful storage and retrieval of all chunks

### Processing Stages:
- **Extraction Stage**: Web crawling and HTML parsing
- **Cleaning Stage**: Content filtering and normalization
- **Structuring Stage**: Document hierarchy preservation
- **Chunking Stage**: Content segmentation with overlap
- **Embedding Stage**: Vector generation using Cohere
- **Storage Stage**: Qdrant Cloud vector storage

## 4. Content Extraction Strategy

### Docusaurus HTML/Markdown Structure Handling:
- Target main content area (typically `.main-wrapper` or similar selectors)
- Extract text from markdown elements (h1-h6, p, li, blockquote, code)
- Preserve semantic structure of headings and content hierarchy
- Identify and extract page titles from appropriate meta tags or h1 elements

### Content Exclusion Rules:
- **Navigation**: Exclude header navigation, breadcrumbs, and navigation menus
- **Sidebar**: Exclude table of contents, navigation sidebar, and related links
- **Footer**: Exclude footer elements, copyright notices, and site navigation
- **Non-book UI elements**: Exclude search bars, social media widgets, cookies banners
- **Code elements**: Extract code content but preserve formatting context

### Heading and Section Boundary Handling:
- Preserve heading hierarchy (h1, h2, h3, etc.) as structural metadata
- Capture section boundaries to maintain context across chunks
- Extract section headings as metadata for each content chunk
- Maintain document tree structure for accurate context preservation

## 5. Chunking & Embedding Strategy

### Chunk Size Rationale:
- **Target size**: 512-1024 tokens (approximately 400-800 words)
- **Rationale**: Balance between semantic coherence and embedding model token limits
- **Flexibility**: Adjust based on Cohere model constraints and retrieval effectiveness

### Overlap Strategy:
- **Overlap percentage**: 20% of chunk size
- **Rationale**: Maintain context continuity across chunk boundaries
- **Implementation**: Overlap occurs at semantic boundaries (sentence/paragraph breaks)

### Section-Level Context Preservation:
- Maintain section hierarchy within each chunk
- Include parent section headings as context for child content
- Preserve document structure through metadata fields
- Ensure chunks don't split important semantic units

### Cohere Model Selection:
- **Primary model**: Cohere's embed-english-v3.0 or latest recommended model
- **Model constraints**: Observe token limits and rate limits
- **Embedding dimensions**: Use standard dimensionality for chosen model
- **Rate limiting**: Implement appropriate delays to respect API limits

## 6. Vector Database Schema

### Qdrant Collection Configuration:
- **Collection name**: `docusaurus_book_embeddings`
- **Vector size**: Match Cohere embedding dimensions (typically 1024)
- **Distance function**: Cosine similarity for semantic search
- **Sharding**: Single shard for Free Tier compatibility

### Payload Schema Definition:
```json
{
  "document_id": "string",
  "source_url": "string",
  "page_title": "string",
  "section_heading": "string",
  "chunk_index": "integer",
  "raw_text": "string",
  "chunk_size": "integer",
  "content_type": "string",
  "processed_at": "datetime"
}
```

### Indexing and Filtering Considerations:
- **Indexed fields**: source_url, page_title, section_heading for fast filtering
- **Full-text search**: Enable on raw_text field if supported
- **Range queries**: Support for chunk_index for ordered retrieval

### Metadata Fields for Downstream Retrieval:
- **URL-based retrieval**: source_url for content location
- **Title-based retrieval**: page_title for page identification
- **Section-based retrieval**: section_heading for content context
- **Position-based retrieval**: chunk_index for sequential access
- **Text boundary support**: raw_text for span-based queries

## 7. Idempotency, Versioning & Failure Handling

### Re-ingestion Strategy:
- **Document identification**: Use URL + content hash for unique identification
- **Reprocessing logic**: Compare content hashes to detect changes
- **Incremental updates**: Process only changed documents

### Duplicate Prevention Logic:
- **Document IDs**: Generate unique document_id based on URL and content hash
- **Qdrant deduplication**: Use document_id as point ID to prevent duplicates
- **Content comparison**: Hash-based comparison to detect identical content

### Partial Failure Handling:
- **Batch processing**: Process content in configurable batches
- **Error isolation**: Continue processing other documents when one fails
- **Retry mechanism**: Implement exponential backoff for transient failures
- **Error logging**: Detailed logs for failed documents with reasons

### Safe Rollback or Overwrite Behavior:
- **Incremental updates**: Update only changed documents, preserve others
- **Backup strategy**: Option to preserve previous embeddings before updates
- **Atomic operations**: Ensure consistency during updates

### Content Update/Version Detection:
- **Content hashing**: Compare current vs stored content hashes
- **URL monitoring**: Track changes in source URLs
- **Update triggers**: Process only changed content on re-run

## 8. Acceptance Criteria

### Testable Conditions for Successful Completion:

1. **Content Extraction Verification**:
   - Given a list of Docusaurus URLs, when the pipeline runs, then it extracts only book content while excluding navigation, sidebar, and footer elements

2. **Chunking Accuracy**:
   - Given content with hierarchical structure, when chunked, then each chunk preserves section context and maintains reading order

3. **Embedding Generation**:
   - Given extracted content chunks, when processed by Cohere, then valid embeddings are generated without exceeding API limits

4. **Metadata Completeness**:
   - Given processed content, when stored in Qdrant, then each vector record contains all required metadata fields (document_id, source_url, page_title, section_heading, chunk_index, raw_text)

5. **Idempotency Verification**:
   - Given an already processed URL, when pipeline runs again, then no duplicate records are created and existing records remain unchanged

6. **Retrieval Readiness**:
   - Given stored embeddings, when queried by URL, page title, or section heading, then relevant content chunks can be retrieved accurately

7. **Error Handling**:
   - Given invalid URLs or network failures, when pipeline runs, then it continues processing other URLs and logs specific errors

8. **Deterministic Output**:
   - Given identical input URLs, when pipeline runs multiple times, then it produces identical vector storage results

9. **Performance Requirements**:
   - Given reasonable content volume, when pipeline runs, then it completes within acceptable timeframes (e.g., 100 pages in under 30 minutes)

10. **Free Tier Compatibility**:
    - Given Qdrant Cloud Free Tier constraints, when pipeline runs, then it operates within resource limits without exceeding quotas