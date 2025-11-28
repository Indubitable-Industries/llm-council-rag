# RAG + LM Studio additions

This fork of Andrej Karpathy’s **llm-council** adds local RAG and repo ZIP upload. The upstream project is awesome; this document only covers the incremental pieces added here.

## What was added
- **Repo ZIP upload per conversation** (`/api/conversations/{id}/upload_repo`): unzips into `temp_repos/{id}`, embeds content, and saves a FAISS index under `data/conversations/{id}_faiss`.
- **Local embeddings** via LM Studio (`text-embedding-nomic-embed-text-v1.5` by default) using `langchain_openai` + FAISS.
- **Auto-context**: every user message is augmented with retrieved snippets from the conversation’s repo index.
- **Two-stage retrieval**: FAISS top-N → LM Studio reranker (BGE) narrows to top-K with neighbor chunk expansion and a configurable cap.
- **Manual context**: file picker + @directives to force files/snippets; when manual context is supplied, RAG retrieval is skipped. Context metadata (scores, lines, tokens) is surfaced to the UI. A guidance blurb is added for RAG cases so models can flag missing context.
- **UX niceties**: stop button, scroll-to-bottom, collapsible context panel, and a CLI (`python -m backend.cli_context`) to inspect what would be sent.

## LM Studio setup
- Run LM Studio with both models loaded:
  - Embedder: `text-embedding-nomic-embed-text-v1.5`
  - Reranker: `text-embedding-bge-reranker-large`
- Endpoint shape: both use `/v1/embeddings`; rerank is client-side by cosine(query, doc) on reranker vectors (no `/v1/rerank`).
- Configure host/port/models in `.env` (`LMSTUDIO_BASE_URL`, `LMSTUDIO_EMBED_MODEL`, `LMSTUDIO_RERANK_MODEL`).
- **Frontend UX**: drag/drop ZIP with progress feedback; chat messages show which files were used for context.

## Environment variables
- `LMSTUDIO_BASE_URL` (default `http://localhost:1234/v1`)
- `LMSTUDIO_EMBED_MODEL` (default `text-embedding-nomic-embed-text-v1.5`)
- `LMSTUDIO_RERANK_MODEL` (default `text-embedding-bge-reranker-large`)
- `RERANK_ENABLED` (default `true`)
- `RETRIEVE_CANDIDATES` (default `50`), `RERANK_TOP_K` (default `20`)
- `CONTEXT_CHUNK_CAP` (default `60`)
- `LMSTUDIO_API_KEY` (ignored by LM Studio; default `lmstudio`)
- Model API key for completion (e.g., `OPENROUTER_API_KEY`), set in your shell or `.env` (not committed).

## Running locally
1. Start LM Studio with the embedding model loaded and API listening on `http://localhost:1234/v1`.
2. Backend: `source .venv/bin/activate` then `uvicorn backend.main:app --reload --port 8001`.
3. Frontend: `nvm use` (uses `.nvmrc`, Node 22+) then `cd frontend && npm install && npm run dev`.
4. Visit the frontend (Vite dev server) and create/select a conversation.
5. Drop a repo `.zip` into the dropzone. The status message will include chunk counts and elapsed time.
6. Ask questions; retrieved context is prepended and also shown in the UI (“Context used” with file names and line counts).

## Logging and observability
- Indexing logs: counts of files/chunks, skipped items, and embedding failures.
- Retrieval logs: number of docs, total characters, and source filenames.
- Upload response includes duration `(took Ns)`.

## Deployment notes
- Keep secrets (e.g., `OPENROUTER_API_KEY`) in environment, not in code.
- Provide a `.env.example` and load environment in process manager/systemd.
- Run backend as a service (systemd or similar) after setting env vars; ensure LM Studio (or another embed endpoint) is reachable.
- Frontend can be built with `npm run build` and served by any static host or Vite preview; respect `VITE_API_BASE` if backend is not on localhost.

## Attribution
Base project by **Andrej Karpathy** (llm-council). This document and the RAG/LM Studio additions describe the fork-specific features.
