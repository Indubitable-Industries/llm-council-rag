# Changelog

## v0.1.0 - Initial RAG + manual context release
- Local LM Studio retrieval: embed with `text-embedding-nomic-embed-text-v1.5`, rerank with `text-embedding-bge-reranker-large`, neighbor expansion, and configurable caps.
- Manual context controls: repo tree picker, `@file:` / `@token` directives, and manual context that bypasses RAG when present.
- Context clarity: collapsible panel with unique file/line summary, scores, and manual vs RAG tags; manual full files summarized.
- UX helpers: stop button to abort streaming, scroll-to-bottom shortcut, and clearer upload/indexing feedback.
- CLI tooling: `python -m backend.cli_context` to preview what context would be sent for a query (with optional manual files).
