# Research Document: RAG Data Ingestion & Vectorization

**Feature**: 1-rag-data-ingestion
**Created**: 2025-12-19
**Status**: Completed

## Research Findings Summary

### 0. UV Package Manager Setup

**Decision**: Use UV as the package manager for the project
**Rationale**: Modern, fast Python package manager with excellent dependency resolution and virtual environment management
**Configuration**:
- Initialize with `uv init` command
- Manage dependencies in `pyproject.toml`
- Use `uv venv` for virtual environment creation
- Leverage `uv pip` for fast dependency installation

### 1. Cohere Embedding Models

**Decision**: Use `embed-english-v3.0` model
**Rationale**: Best balance of performance and cost for English text content. Supports up to 512 tokens per request with 1024-dimensional embeddings.
**Alternatives considered**:
- `embed-multilingual-v3.0` (for non-English content, not needed for this project)
- Older v2 models (less efficient than v3)

### 2. Qdrant Python Client

**Decision**: Use official `qdrant-client` library
**Rationale**: Official client with active maintenance, good documentation, and proper async support
**Alternatives considered**: Direct HTTP API calls (more complex, error-prone)

### 3. Docusaurus HTML Structure

**Decision**: Target `.markdown` and `[class*="docItem"]` selectors
**Rationale**: Standard Docusaurus content containers that exclude navigation and sidebars
**Additional selectors**: `article`, `[class*="theme-doc-markdown"]`

### 4. Chunking Algorithms

**Decision**: RecursiveCharacterTextSplitter with semantic boundaries
**Rationale**: Preserves document structure while maintaining semantic coherence
**Parameters**:
- Chunk size: 512 tokens
- Overlap: 20% (102 tokens)
- Separators: ["\n\n", "\n", " ", ""]

### 5. Rate Limiting Strategies

**Decision**: Token bucket algorithm with exponential backoff
**Rationale**: Smooths API usage while handling rate limits gracefully
**Parameters**:
- Max requests per minute based on Cohere free tier
- Exponential backoff: 1s, 2s, 4s, 8s, 16s

## Technical Decisions Made

### Content Extraction Strategy
- Use BeautifulSoup with lxml parser for robust HTML parsing
- Target specific Docusaurus selectors to exclude UI elements
- Extract heading hierarchy to preserve document structure
- Implement content filtering based on CSS class patterns

### Idempotency Implementation
- Generate document_id using SHA256 hash of URL + content
- Use Qdrant point IDs for duplicate prevention
- Compare content hashes to detect changes during re-processing
- Implement upsert behavior for updated content

### Error Handling Approach
- Implement circuit breaker pattern for external API calls
- Use retry mechanisms with exponential backoff
- Log errors with sufficient context for debugging
- Continue processing other documents when one fails

### Performance Optimization
- Batch process content chunks for embedding generation
- Use connection pooling for HTTP requests
- Implement parallel processing for independent operations
- Cache frequently accessed configuration values

## Validation Strategy

### Testing Approach
- Unit tests for each module with 90%+ coverage
- Integration tests with mock external services
- End-to-end tests with sample Docusaurus sites
- Performance tests to validate processing time requirements

### Quality Assurance
- Static code analysis using linters
- Type checking with mypy
- Security scanning for dependencies
- Performance monitoring and metrics collection

## Implementation Notes

### Security Considerations
- Store API keys in environment variables only
- Validate all URLs before processing
- Implement proper input sanitization
- Use secure HTTP connections only

### Scalability Considerations
- Configurable batch sizes for different environments
- Asynchronous processing where possible
- Memory-efficient processing for large documents
- Distributed processing capability (future enhancement)

### Monitoring & Observability
- Comprehensive logging with structured format
- Performance metrics collection
- Error tracking and alerting
- Processing progress indicators