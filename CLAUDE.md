# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LightRAG is a Retrieval-Augmented Generation (RAG) framework that uses graph-based knowledge representation for enhanced information retrieval. The system extracts entities and relationships from documents, builds a knowledge graph, and uses multi-modal retrieval (local, global, hybrid, mix, naive) for queries.

**This is a fork** of [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) (`upstream` remote). The `origin` remote points to `tomorrowflow/LightRAG`. The fork adds an explicit-document-ID insert option, an input-dir file lifecycle, parse-cache cleanup helpers, conditional pgvector, and various fixes. (An earlier bespoke RAGAnything integration and Scheme Manager UI were **removed** once upstream merged native multimodal parsing — see below.) See [Fork-Specific Changes](#fork-specific-changes) below.

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

**Gotcha — a local `.env` poisons the config tests.** `load_dotenv` reads `.env`
from the working directory, so running `pytest` in a checkout that has a real
`.env` leaks your bindings into tests that assert on binding/role resolution.
With this fork's `.env` (`RETRIEVAL_LLM_BINDING=ollama`, `KEYWORD_LLM_BINDING=ollama`)
that produces ~11 spurious failures in `tests/api/config/` — mostly
`SystemExit: Cross-provider error for role ...` — plus one in
`test_ollama_think_startup_validation.py`. They are not real breakage. Run the
suite from a clean worktree (`git worktree add <dir> <branch>`; untracked `.env`
is not copied) to get a trustworthy result.

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
bun run dev                    # Development server (Node + Vite)
bun run dev:bun                # Development server (Bun native)
bun run build                  # Production build
bun run preview                # Preview production build locally

# Linting (ESLint with TypeScript, React hooks, Stylistic rules)
bun run lint                   # Run ESLint on all *.ts/tsx/js/jsx files

# Testing (Bun built-in test runner)
bun test                       # Run all tests
bun test --watch               # Watch mode
bun test --coverage            # With coverage report
bun test src/api/lightrag.test.ts  # Run a single test file
```

### Lint Rules
ESLint is configured with TypeScript-ESLint, React Hooks plugin, Prettier integration, and `@stylistic` rules:
- 2-space indentation, single quotes enforced
- `@typescript-eslint/no-explicit-any` is disabled (allowed)

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

### RAGAnything / Multimodal — REMOVED (now upstream-native)

The fork originally vendored `raganything` (a `RAGManager` singleton, a Scheme Manager UI, `multimodal_content`/`scheme_name` plumbing, and `READY`/`HANDLING` doc statuses). Upstream has since merged **native** multimodal parsing (MinerU/Docling parser routing in `lightrag/parser/`, `lightrag/multimodal_context.py`, `lightrag/prompt_multimodal.py`, the `analyze_multimodal` pipeline stage, and a `vlm` LLM role). The bespoke fork integration was therefore fully removed in favor of upstream's: there is no `raganything` dependency, no `RAGManager`, no Scheme Manager, and no `multimodal_content`/`scheme_name` fields. Use upstream's parser-engine selection (filename hints / `LIGHTRAG_PARSER`) and the `vlm` role for vision.

### Retrieval-Specific LLM — SUPERSEDED by upstream role LLMs

The fork's `retrieval_llm_model_func` / `RETRIEVAL_LLM_MODEL` was superseded by upstream's role-based LLM system (`role_llm_funcs` with `query`/`keyword`/`extract`/`vlm` roles, configured via `QUERY_LLM_*` / `KEYWORD_LLM_*` env vars). `kg_query`, `extract_keywords_only`, and `naive_query` now route through `role_llm_funcs["query"]` / `["keyword"]`. To run a separate model for query-time operations, set `QUERY_LLM_*` / `KEYWORD_LLM_*` instead of `RETRIEVAL_LLM_*`.

A backward-compat shim in `lightrag/api/config.py` still maps any `RETRIEVAL_LLM_BINDING` / `RETRIEVAL_LLM_MODEL` / `RETRIEVAL_LLM_BINDING_HOST` / `RETRIEVAL_LLM_BINDING_API_KEY` onto the `QUERY_` and `KEYWORD_` role vars (only where those aren't set explicitly) and logs a deprecation warning. `docker-compose.yml` still sets `RETRIEVAL_LLM_BINDING_HOST`, which works via this shim — migrate it to `QUERY_LLM_BINDING_HOST` / `KEYWORD_LLM_BINDING_HOST` when convenient.

### File Lifecycle on Insert

- **`lightrag/lightrag.py`**: New `input_dir` field (env: `INPUT_DIR`, default `./inputs`). After enqueuing, source files are moved from `input_dir` into an `__enqueued__` subdirectory with collision-safe naming.

### Parse Cache Cleanup

- **`lightrag/lightrag.py`**: New methods `aclean_parse_cache_by_doc_ids()`, `clean_parse_cache_by_doc_ids()`, `aclean_all_parse_cache()`, `clean_all_parse_cache()` for removing cached parsing results from `kv_store_parse_cache.json`.

### Explicit Document IDs on Text Insert

- **`lightrag/api/routers/document_routes.py`**: `/documents/text` accepts `id` and
  `/documents/texts` accepts `ids`, forwarded through `pipeline_index_texts` to
  `apipeline_enqueue_documents`. Forwarded **only when supplied** — passing
  `ids=None` unconditionally changes the call shape and breaks upstream test
  doubles that don't declare the parameter. Upstream treats a provided `ids` as
  the SDK raw direct-insert path (always enqueued RAW, never `pending_parse`),
  which is the correct semantics for these text endpoints.

### RELATIONSHIP_TYPES — REMOVED (was inert, superseded)

The fork's `RELATIONSHIP_TYPES` / `DEFAULT_RELATIONSHIP_TYPES` "soft-priority
relationship keywords" param was removed during the v1.5.7 merge. It only ever
registered an addon-param key that no extraction code read, and upstream has
since replaced the list-based `entity_types` addon param with a free-text
`entity_types_guidance` string (`lightrag/addon_params.py`, `lightrag/prompt.py`).
The fork's `_default_addon_params()` helper was dead code by then and its
`DEFAULT_ENTITY_TYPES` constant broke an upstream invariant test. To steer
relationship extraction now, set `addon_params["entity_types_guidance"]` or
override the prompt template. `env.example` still documents the old
`RELATIONSHIP_TYPES` line — it has no effect.

### Pipeline Status History (upstream contract)

All writes to `pipeline_status["history_messages"]` must go through
`append_pipeline_history` from `lightrag.kg.shared_storage` (LR2 §10.3) — a
repo-wide test fails if a raw `.append(...)` reappears. The helper is
never-raising and skips a missing/None `history_messages`, so no guard is needed.

### PostgreSQL Conditional pgvector — SUPERSEDED by upstream

The fork's `POSTGRES_ENABLE_VECTOR` env var is gone. Upstream now derives pgvector
usage from the configured backend (`enable_vector = vector_storage == "PGVectorStorage"`),
so an unspecified vector backend no longer demands the extension. There is deliberately
no `None -> True` default. To run PostgreSQL without pgvector, simply use a non-PG
vector storage; `PGVectorStorage.initialize` requests the extension itself.

### PostgreSQL Parse-Cache Namespace

- **`lightrag/kg/postgres_impl.py`**: `PGKVStorage` supports the `KV_STORE_PARSE_CACHE`
  namespace (`LIGHTRAG_PARSE_CACHE` table, `get_by_id`/`get_by_ids`/`upsert_parse_cache`
  templates), so the parse cache works on PostgreSQL and not just the JSON backend.

### Bug Fixes (vs. upstream)

- **Entity merge data loss** (`operate.py`): SUPERSEDED. Upstream's issue #3400 rework
  moved the `full_entities` / `full_relations` writes into "Phase 0" — a write-ahead
  recovery anchor persisted from the full candidate superset *before* any graph
  mutation. The fork's post-merge "Phase 3" read-merge-write was removed as part of
  adopting that; re-adding it would reintroduce the post-mutation write that upstream
  deliberately deleted.
- **Chinese space removal guard** (`utils.py`): The Chinese-specific space-stripping regex in `normalize_extracted_info` is now conditional on detecting Chinese characters, preventing mangling of English text like "AI Framework".
- **Ollama embedding robustness** (`llm/ollama.py`): Empty/whitespace texts replaced with placeholder, NaN embeddings replaced with zeros, retry decorator added to `ollama_embed()`.
- **Ollama retry policy**: Increased from 3 to 5 attempts, max wait from 10s to 60s, added `ResponseError` to retryable exceptions.
- **Typo fixes**: `seperator` → `separator`, `descpriton` → `description`, `seperate` → `separate`.

### Docker / Deployment

- **Dockerfile**: Installs system deps for MinerU/OpenCV (`libgl1`, `libglib2.0-0`, X11 libs). The `RAG-Anything/` COPY is gone. Uses upstream's `docker-entrypoint.sh` ENTRYPOINT, which chowns the data dirs (honouring `WORKING_DIR`/`INPUT_DIR`/`PROMPT_DIR`) and drops to the non-root `lightrag` user via gosu — so the fork no longer overrides CMD with explicit `--working-dir`/`--input-dir` flags; compose sets those as env vars instead.
- **`docker-compose.yml`**: Multi-service architecture with PostgreSQL, Neo4j, Qdrant, Redis, Memgraph, MongoDB, and vLLM reranker services. Separate vision model host configuration. Kept wholesale over upstream's single-service template.
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

## AGENTS.md

Also follow the repository rules in [./AGENTS.md](./AGENTS.md).

@AGENTS.md
