# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LightRAG is a Retrieval-Augmented Generation (RAG) framework that uses graph-based knowledge representation for enhanced information retrieval. The system extracts entities and relationships from documents, builds a knowledge graph, and uses multi-modal retrieval (local, global, hybrid, mix, naive) for queries.

**This is a fork** of [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) (`upstream` remote). The `origin` remote points to `tomorrowflow/LightRAG`. The fork adds RAGAnything integration, a retrieval-specific LLM option, a Scheme Manager UI, and various fixes. See [Fork-Specific Changes](#fork-specific-changes) below.

## Core Architecture

### Key Components

- **lightrag.py**: Main orchestrator class (`LightRAG`) that coordinates document insertion, query processing, and storage management. Critical: Always call `await rag.initialize_storages()` after instantiation.

- **operate.py**: Core extraction and query operations including entity/relation extraction, chunking, and multi-mode retrieval logic.

- **base.py**: Abstract base classes for storage backends (`BaseKVStorage`, `BaseVectorStorage`, `BaseGraphStorage`, `BaseDocStatusStorage`).

- **kg/**: Storage implementations (JSON, NetworkX, Neo4j, PostgreSQL, MongoDB, Redis, Milvus, Qdrant, Faiss, Memgraph). Each storage type provides different trade-offs for production vs. development use.

- **llm/**: LLM provider bindings (OpenAI, Ollama, Azure, Gemini, Bedrock, Anthropic, etc.). All use async patterns with caching support.

- **api/**: FastAPI server (`lightrag_server.py`) with REST endpoints and Ollama-compatible API, plus React 19 + TypeScript WebUI.

### Storage Layer

LightRAG uses 4 storage types with pluggable backends:
- **KV_STORAGE**: LLM response cache, text chunks, document info
- **VECTOR_STORAGE**: Entity/relation/chunk embeddings
- **GRAPH_STORAGE**: Entity-relation graph structure
- **DOC_STATUS_STORAGE**: Document processing status tracking

Workspace isolation is implemented differently per storage type (subdirectories for file-based, prefixes for collections, fields for relational DBs).

### Query Modes

- **local**: Context-dependent retrieval focused on specific entities
- **global**: Community/summary-based broad knowledge retrieval
- **hybrid**: Combines local and global
- **naive**: Direct vector search without graph
- **mix**: Integrates KG and vector retrieval (recommended with reranker)

## Development Commands

### Setup
```bash
# Install core package (development mode)
uv sync
source .venv/bin/activate  # Or: .venv\Scripts\activate on Windows

# Install with API support
uv sync --extra api

# Install specific extras
uv sync --extra offline-storage  # Storage backends
uv sync --extra offline-llm      # LLM providers
uv sync --extra test             # Testing dependencies
```

### API Server
```bash
# Copy and configure environment
cp env.example .env  # Edit with your LLM/embedding configs

# Build WebUI
cd lightrag_webui
bun install --frozen-lockfile
bun run build
cd ..

# Run server
lightrag-server                                           # Production
uvicorn lightrag.api.lightrag_server:app --reload        # Development
lightrag-gunicorn                                         # Multi-worker (gunicorn)
```

### Testing
```bash
pytest tests                          # Offline tests only (default)
pytest tests --run-integration        # Include integration tests (requires external services)
pytest tests/test_chunking.py         # Run specific test file
pytest tests --keep-artifacts         # Keep temp dirs for debugging
pytest tests --test-workers 4         # Custom parallel workers (default: 3)
pytest tests --stress-test            # Enable stress test mode
```

Pytest is configured with `asyncio_mode = "auto"` — async test functions are automatically detected (no `@pytest.mark.asyncio` needed).

Environment variable overrides: `LIGHTRAG_RUN_INTEGRATION`, `LIGHTRAG_KEEP_ARTIFACTS`, `LIGHTRAG_TEST_WORKERS`, `LIGHTRAG_STRESS_TEST`.

Test markers: `offline`, `integration`, `requires_db`, `requires_api`.

### Linting
```bash
ruff check .
```

## Key Implementation Patterns

### LightRAG Initialization (Critical)

The most common error is forgetting to initialize storages:

```python
import asyncio
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

async def main():
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed
    )

    # REQUIRED: Initialize storage backends
    await rag.initialize_storages()

    # Now safe to use
    await rag.ainsert("Your text here")
    result = await rag.aquery("Your question", param=QueryParam(mode="hybrid"))

    # Cleanup
    await rag.finalize_storages()

asyncio.run(main())
```

### Custom Embedding Functions

Use `@wrap_embedding_func_with_attrs` decorator and call `.func` when wrapping:

```python
from lightrag.utils import wrap_embedding_func_with_attrs

@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
async def custom_embed(texts: list[str]) -> np.ndarray:
    # Call underlying function, not wrapped version
    return await openai_embed.func(texts, model="text-embedding-3-large")
```

### Storage Configuration

Configure via environment variables or constructor params:

```python
# Environment-based (recommended for production)
# See env.example for full list

# Constructor-based
rag = LightRAG(
    working_dir="./storage",
    workspace="project_name",  # For data isolation
    kv_storage="PGKVStorage",
    vector_storage="PGVectorStorage",
    graph_storage="Neo4JStorage",
    doc_status_storage="PGDocStatusStorage",
    vector_db_storage_cls_kwargs={
        "cosine_better_than_threshold": 0.2
    }
)
```

### Document Insertion

```python
# Single document
await rag.ainsert("Text content")

# Batch insertion
await rag.ainsert(["Text 1", "Text 2", ...])

# With custom IDs
await rag.ainsert("Text", ids=["doc-123"])

# With file paths (for citation)
await rag.ainsert(["Text 1", "Text 2"], file_paths=["doc1.pdf", "doc2.pdf"])

# Configure batch size
rag = LightRAG(..., max_parallel_insert=4)  # Default: 2, max recommended: 10
```

### Query Configuration

```python
from lightrag import QueryParam

result = await rag.aquery(
    "Your question",
    param=QueryParam(
        mode="mix",                    # Recommended with reranker
        top_k=60,                      # KG entities/relations to retrieve
        chunk_top_k=20,                # Text chunks to retrieve
        max_entity_tokens=6000,
        max_relation_tokens=8000,
        max_total_tokens=30000,
        enable_rerank=True,
        user_prompt="Additional instructions for LLM",
        stream=False
    )
)
```

## WebUI Development

### Structure
- `lightrag_webui/src/`: React components (TypeScript)
- Uses Vite + Bun build system
- Tailwind CSS for styling
- React 19 with functional components and hooks

### Commands
```bash
cd lightrag_webui
bun install --frozen-lockfile  # Install dependencies
bun run dev                    # Development server
bun run build                  # Production build
bun test                       # Run tests
```

## Common Issues

### 1. Storage Not Initialized
**Error**: `AttributeError: __aenter__` or `KeyError: 'history_messages'`
**Solution**: Always call `await rag.initialize_storages()` after creating LightRAG instance

### 2. Embedding Model Changes
When switching embedding models, you MUST clear the data directory (except optionally `kv_store_llm_response_cache.json` for LLM cache).

### 3. Nested Embedding Functions
Cannot wrap already-decorated embedding functions. Use `.func` to access underlying function:
```python
# Wrong: EmbeddingFunc(func=openai_embed)
# Right: EmbeddingFunc(func=openai_embed.func)
```

### 4. Context Length for Ollama
Ollama models default to 8k context; LightRAG requires 32k+. Configure via:
```python
llm_model_kwargs={"options": {"num_ctx": 32768}}
```

## Configuration Files

### .env Configuration
Primary configuration file for API server. Key sections:
- Server settings (HOST, PORT, CORS)
- Storage backends (connection strings via environment variables)
- Query parameters (TOP_K, MAX_TOTAL_TOKENS, etc.)
- Reranking configuration (RERANK_BINDING, RERANK_MODEL)
- Authentication (AUTH_ACCOUNTS, LIGHTRAG_API_KEY)

See `env.example` for comprehensive template.

### Workspace Isolation
Each LightRAG instance can use a `workspace` parameter for data isolation. Implementation varies by storage type:
- File-based: subdirectories
- Collection-based: collection name prefixes
- Relational DB: workspace column filtering
- Qdrant: payload-based partitioning

## Fork-Specific Changes

This fork (`tomorrowflow/LightRAG`) diverges from upstream (`HKUDS/LightRAG`) in the following ways:

### RAGAnything / Multimodal Content Integration

- **`lightrag/ragmanager.py`** (new): Singleton `RAGManager` holding a global reference to a RAGAnything instance for multimodal processing.
- **`lightrag/base.py`**: Added `READY` and `HANDLING` doc statuses. Added `multimodal_content` and `scheme_name` fields to `DocProcessingStatus`.
- **`lightrag/lightrag.py`**: `insert()`/`ainsert()` accept `multimodal_content` and `scheme_name` params. After text processing, if multimodal content exists, `RAGManager._process_multimodal_content()` is called before marking as `PROCESSED`.
- **`raganything`** submodule: Expected at `./raganything/` as a local dependency (configured in `pyproject.toml` under `[tool.uv.sources]`).

### Retrieval-Specific LLM

- **`lightrag/lightrag.py`**: New `retrieval_llm_model_func` field allows a separate LLM for query operations (falls back to `llm_model_func`). New `retrieval_llm_model_name` field (env: `RETRIEVAL_LLM_MODEL`) is baked into the retrieval function's `partial()` wrapper via `model_name` kwarg, so LLM bindings use the correct model name.
- **`lightrag/operate.py`**: `kg_query`, `extract_keywords_only`, and `naive_query` use `retrieval_llm_model_func` when available.
- **LLM bindings** (`ollama.py`, `openai.py`, `anthropic.py`, `bedrock.py`, `hf.py`, `lollms.py`, `gemini.py`): All check for a `model_name` kwarg before falling back to `global_config["llm_model_name"]`.

### Scheme Manager (WebUI)

- **`lightrag_webui/src/components/documents/SchemeManager/`** (new): Dialog UI for managing processing "schemes" — configurations that select between LightRAG and RAGAnything frameworks, extraction tools (MinerU/Docling), and model sources.
- **`lightrag_webui/src/contexts/SchemeContext.tsx`** (new): React context for sharing selected scheme across the UI.
- **`lightrag_webui/src/api/lightrag.ts`**: Added `Scheme` type and CRUD API methods (`getSchemes`, `saveSchemes`, `addScheme`, `deleteScheme`) hitting `/documents/schemes` endpoints.

### File Lifecycle on Insert

- **`lightrag/lightrag.py`**: New `input_dir` field (env: `INPUT_DIR`, default `./inputs`). After enqueuing, source files are moved from `input_dir` into an `__enqueued__` subdirectory with collision-safe naming.

### Parse Cache Cleanup

- **`lightrag/lightrag.py`**: New methods `aclean_parse_cache_by_doc_ids()`, `clean_parse_cache_by_doc_ids()`, `aclean_all_parse_cache()`, `clean_all_parse_cache()` for removing cached parsing results from `kv_store_parse_cache.json`.

### PostgreSQL Conditional pgvector

- **`lightrag/kg/postgres_impl.py`**: `POSTGRES_ENABLE_VECTOR` env var / config option. When `false`, skips creating the vector extension and `register_vector`. `PGVectorStorage` raises an explicit error if used with vector disabled.

### Bug Fixes (vs. upstream)

- **Entity merge preserves existing data** (`operate.py`): `merge_nodes_and_edges` now fetches existing entities/relations before merging, preventing data loss on re-processing.
- **Chinese space removal guard** (`utils.py`): The Chinese-specific space-stripping regex in `normalize_extracted_info` is now conditional on detecting Chinese characters, preventing mangling of English text like "AI Framework".
- **Ollama embedding robustness** (`llm/ollama.py`): Empty/whitespace texts replaced with placeholder, NaN embeddings replaced with zeros, retry decorator added to `ollama_embed()`.
- **Ollama retry policy**: Increased from 3 to 5 attempts, max wait from 10s to 60s, added `ResponseError` to retryable exceptions.
- **Typo fixes**: `seperator` → `separator`, `descpriton` → `description`, `seperate` → `separate`.

### Docker / Deployment

- **Dockerfile**: Copies local `RAG-Anything/` into the build, installs system deps for MinerU/OpenCV (`libgl1`, `libglib2.0-0`, X11 libs).
- **`docker-compose.yaml`**: Multi-service architecture with PostgreSQL, Neo4j, and vLLM reranker services. Separate vision model host configuration.
- **`lightrag/utils.py`**: `.env` path hardcoded to `/app/.env` for Docker convention. **Note**: This may need adjustment for non-Docker development.

### Build / Dependency Changes

- **`pyproject.toml`**: Added `onnxruntime` as core dep. New `offline-docs` extra with `raganything` and document processing libs. Local `raganything` source in `[tool.uv.sources]`. Added `[tool.uv]` prerelease config.
- **`uv.lock`** removed from `.gitignore` (tracked in repo).

## Code Style

### Language
- Comment Language - Use English for comments and documentation
- Backend Language - Use English for backend code and messages
- Frontend Internationalization: i18next for multi-language support

### Python
- Follow PEP 8 with 4-space indentation
- Use type annotations
- Prefer dataclasses for state management
- Use `lightrag.utils.logger` instead of print
- Async/await patterns throughout
- Keep storage implementations in `kg/` with consistent base class inheritance

### TypeScript/React
- Functional components with hooks
- 2-space indentation
- PascalCase for components
- Tailwind utility-first styling

## Important Architectural Notes

### LLM Requirements
- Minimum 32B parameters recommended
- 32KB context minimum (64KB recommended)
- Avoid reasoning models during indexing
- Stronger models for query stage than indexing stage

### Embedding Models
- Must be consistent across indexing and querying
- Recommended: `BAAI/bge-m3`, `text-embedding-3-large`
- Changing models requires clearing vector storage and recreating with new dimensions

### Reranker Configuration
- Significantly improves retrieval quality
- Recommended models: `BAAI/bge-reranker-v2-m3`, Jina rerankers
- Use "mix" mode when reranker is enabled
