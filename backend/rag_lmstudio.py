# backend/rag_lmstudio.py

import logging
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import math
import httpx
import subprocess
import shutil
import json
import fnmatch
import time

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import (
    LMSTUDIO_BASE_URL,
    LMSTUDIO_EMBED_MODEL,
    LMSTUDIO_RERANK_MODEL,
    RERANK_ENABLED,
    RETRIEVE_CANDIDATES,
    RERANK_TOP_K,
    CONTEXT_CHUNK_CAP,
    INDEX_INCLUDE_GLOBS,
    INDEX_EXCLUDE_GLOBS,
    INDEX_INCLUDE_UNTRACKED,
    INDEX_MANIFEST_PATH,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Directories we don't want to index (big junk, IDE stuff, etc.)
SKIP_DIR_NAMES = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
}


def _iter_source_files(root: Path) -> List[Path]:
    """
    Collect source-ish files from the repo for embedding.

    Start with a small set of extensions; we can add more later.
    """
    exts = {".py", ".ipynb", ".md", ".txt", ".json", ".yaml", ".yml"}
    files: List[Path] = []

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(skip in p.parts for skip in SKIP_DIR_NAMES):
            continue
        if p.suffix.lower() in exts:
            files.append(p)

    return files


# ---------------------------------------------------------------------------
# Embeddings + vectorstores
# ---------------------------------------------------------------------------

embeddings = OpenAIEmbeddings(
    base_url=LMSTUDIO_BASE_URL,
    api_key=os.getenv("LMSTUDIO_API_KEY", "lmstudio"),  # LM Studio ignores this
    model=LMSTUDIO_EMBED_MODEL,
    # Disable token-length re-chunking to keep payloads as plain strings.
    check_embedding_ctx_length=False,
)

# One FAISS index per conversation id
VECTORSTORES: Dict[str, FAISS] = {}


def index_repo_zip(zip_path: str, convo_id: str) -> str:
    """
    Unzip a repo under temp_repos/{convo_id}, build a FAISS index,
    and persist it under data/conversations/{convo_id}_faiss.
    """
    temp_dir = Path("temp_repos") / convo_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)

    return index_repo_dir(temp_dir, convo_id)


def _load_manifest() -> Dict[str, Any]:
    path = Path(INDEX_MANIFEST_PATH)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_manifest(manifest: Dict[str, Any]):
    path = Path(INDEX_MANIFEST_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _matches_globs(path: str, patterns: List[str]) -> bool:
    if not patterns:
        return False
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


def _git_ls_tracked(root: Path) -> List[str]:
    cmd = ["git", "ls-tree", "-r", "--name-only", "HEAD"]
    try:
        res = subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True)
        return [line.strip() for line in res.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _git_status_paths(root: Path) -> List[str]:
    cmd = ["git", "status", "--porcelain"]
    try:
        res = subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True)
        paths = []
        for line in res.stdout.splitlines():
            if len(line) < 4:
                continue
            path = line[3:].strip()
            if path:
                paths.append(path)
        return paths
    except Exception:
        return []


def build_worktree_snapshot(convo_id: str, repo_root: Path | None = None, include_untracked: bool | None = None) -> str:
    """
    Create a temp snapshot of the git working tree (with optional untracked files)
    under temp_repos/{convo_id}.
    """
    root = repo_root or Path(".").resolve()
    tracked = set(_git_ls_tracked(root))
    include_untracked = INDEX_INCLUDE_UNTRACKED if include_untracked is None else include_untracked
    status_paths = set(_git_status_paths(root)) if include_untracked else set()
    candidates = tracked | status_paths

    snapshot_dir = Path("temp_repos") / convo_id
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for rel in sorted(candidates):
        src = root / rel
        if not src.is_file():
            continue
        rel_str = rel.replace("\\", "/")
        if _matches_globs(rel_str, INDEX_EXCLUDE_GLOBS):
            continue
        if INDEX_INCLUDE_GLOBS and not _matches_globs(rel_str, INDEX_INCLUDE_GLOBS):
            continue
        if any(skip in src.parts for skip in SKIP_DIR_NAMES):
            continue

        dest = snapshot_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dest)
            copied += 1
        except Exception:
            continue

    return f"Prepared git snapshot with {copied} files for {convo_id}."

def index_repo_dir(root_dir: Path, convo_id: str) -> str:
    """
    Index an existing directory of files (already materialized under temp_repos/{convo_id}).
    """
    docs: List[Document] = []
    skipped_large = 0
    skipped_load_fail = 0

    for src in _iter_source_files(root_dir):
        # Skip huge files (just in case)
        if src.stat().st_size > 500_000:
            skipped_large += 1
            continue

        try:
            loader = TextLoader(str(src), encoding="utf-8", autodetect_encoding=True)
            raw_docs = loader.load()
            for d in raw_docs:
                d.metadata["source"] = str(src.relative_to(root_dir))
            docs.extend(raw_docs)
        except Exception:
            # Best-effort: if a file blows up, we just skip it
            skipped_load_fail += 1
            continue

    if not docs:
        return "No suitable source files found to index."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\nclass ", "\ndef ", "\n\n", "\n", " "],
    )

    chunks = splitter.split_documents(docs)

    # Assign per-file chunk indices and doc ids for later neighbor lookup
    per_file_counts: Dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        idx = per_file_counts.get(src, 0)
        chunk.metadata["doc_id"] = src
        chunk.metadata["chunk_index"] = idx
        per_file_counts[src] = idx + 1

    # Ensure we only send clean strings to the embed endpoint
    safe_chunks: List[Document] = []
    skipped_empty = 0
    skipped_bad = 0
    for chunk in chunks:
        content = chunk.page_content
        if not isinstance(content, str):
            try:
                content = str(content)
            except Exception:
                skipped_bad += 1
                continue
        content = content.strip()
        if not content:
            skipped_empty += 1
            continue

        safe_chunks.append(
            Document(page_content=content, metadata=chunk.metadata)
        )

    if not safe_chunks:
        logger.info(
            (
                "RAG indexing produced no valid chunks "
                "(convo=%s files=%d chunks_raw=%d skipped_empty=%d "
                "skipped_bad=%d skipped_large_files=%d skipped_load_fail=%d)"
            ),
            convo_id,
            len(docs),
            len(chunks),
            skipped_empty,
            skipped_bad,
            skipped_large,
            skipped_load_fail,
        )
        return "No valid text chunks found after cleaning."

    # Build vectorstore (explicitly pass plain strings + metadata to avoid bad payloads)
    texts = [doc.page_content for doc in safe_chunks]
    metadatas = [doc.metadata for doc in safe_chunks]

    try:
        vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    except Exception:
        logger.exception(
            "RAG embedding failed "
            "(convo=%s files=%d chunks_raw=%d chunks_clean=%d)",
            convo_id,
            len(docs),
            len(chunks),
            len(safe_chunks),
        )
        raise
    VECTORSTORES[convo_id] = vectorstore

    # Persist index for restart survival
    data_dir = Path("data") / "conversations"
    data_dir.mkdir(parents=True, exist_ok=True)
    save_path = data_dir / f"{convo_id}_faiss"
    vectorstore.save_local(str(save_path))

    # Manifest: record simple metadata for change detection
    manifest = _load_manifest()
    files_meta = []
    for src in _iter_source_files(root_dir):
        try:
            stat = src.stat()
            rel = str(src.relative_to(root_dir))
            files_meta.append(
                {
                    "path": rel,
                    "bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )
        except Exception:
            continue
    manifest[convo_id] = {
        "root": str(root_dir),
        "file_count": len(files_meta),
        "indexed_at": time.time(),
        "files": files_meta,
    }
    _save_manifest(manifest)

    logger.info(
        (
            "RAG indexing complete "
            "(convo=%s files=%d chunks_raw=%d chunks_clean=%d "
            "skipped_empty=%d skipped_bad=%d skipped_large_files=%d skipped_load_fail=%d)"
        ),
        convo_id,
        len(docs),
        len(chunks),
        len(safe_chunks),
        skipped_empty,
        skipped_bad,
        skipped_large,
        skipped_load_fail,
    )

    return (
        f"Indexed {len(safe_chunks)} chunks from {len(docs)} files for conversation {convo_id}."
    )


def _lazy_load_vectorstore(convo_id: str) -> FAISS | None:
    """
    Get a FAISS index for convo_id from memory or disk.
    Returns None if nothing exists yet.
    """
    if convo_id in VECTORSTORES:
        return VECTORSTORES[convo_id]

    data_dir = Path("data") / "conversations"
    path = data_dir / f"{convo_id}_faiss"

    if not path.exists():
        return None

    try:
        vs = FAISS.load_local(
            str(path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        VECTORSTORES[convo_id] = vs
        return vs
    except Exception:
        return None


def _estimate_tokens(text: str) -> int:
    # Quick heuristic: 4 chars per token
    return max(1, len(text) // 4)


def _doc_lookup_by_chunk(vs: FAISS) -> Dict[tuple, Document]:
    try:
        store = vs.docstore._dict  # type: ignore[attr-defined]
    except Exception:
        return {}

    lookup: Dict[tuple, Document] = {}
    for doc in store.values():
        meta = doc.metadata or {}
        doc_id = meta.get("doc_id")
        chunk_idx = meta.get("chunk_index")
        if doc_id is None or chunk_idx is None:
            continue
        lookup[(doc_id, chunk_idx)] = doc
    return lookup


def _compute_cosine(a: List[float], b: List[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def retrieve_candidates(convo_id: str, query: str, n: int) -> List[Document]:
    vs = _lazy_load_vectorstore(convo_id)
    if vs is None:
        return []

    retriever = vs.as_retriever(
        search_kwargs={"k": n},
    )
    docs = retriever.invoke(query)
    return docs or []


def rerank_with_bge(query: str, docs: List[Document]) -> List[tuple]:
    if not docs:
        return []

    url = f"{LMSTUDIO_BASE_URL.rstrip('/')}/embeddings"
    payload = {
        "model": LMSTUDIO_RERANK_MODEL,
        "input": [query] + [d.page_content for d in docs],
    }

    try:
        resp = httpx.post(url, json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception:
        logger.exception("Rerank call failed")
        return [(doc, None) for doc in docs]

    if len(data) != len(docs) + 1:
        return [(doc, None) for doc in docs]

    query_vec = data[0].get("embedding", [])
    doc_vecs = [item.get("embedding", []) for item in data[1:]]

    scores: List[float] = []
    for vec in doc_vecs:
        scores.append(_compute_cosine(query_vec, vec))

    ranked = sorted(zip(docs, scores), key=lambda x: (x[1] is not None, x[1]), reverse=True)
    return ranked


def expand_with_neighbors(vs: FAISS, ranked_docs: List[tuple]) -> List[tuple]:
    if not ranked_docs:
        return []

    lookup = _doc_lookup_by_chunk(vs)
    seen = set()
    expanded: List[tuple] = []

    def add_entry(doc: Document, score: Optional[float]):
        key = (
            doc.metadata.get("doc_id"),
            doc.metadata.get("chunk_index"),
        )
        if key in seen:
            return
        seen.add(key)
        expanded.append((doc, score))

    for doc, score in ranked_docs:
        add_entry(doc, score)

        doc_id = doc.metadata.get("doc_id")
        chunk_idx = doc.metadata.get("chunk_index")
        if doc_id is None or chunk_idx is None:
            continue

        prev_key = (doc_id, chunk_idx - 1)
        next_key = (doc_id, chunk_idx + 1)
        if prev_key in lookup:
            add_entry(lookup[prev_key], score)
        if next_key in lookup:
            add_entry(lookup[next_key], score)

    return expanded


def retrieve_reranked(convo_id: str, query: str) -> List[tuple]:
    vs = _lazy_load_vectorstore(convo_id)
    if vs is None:
        return []

    candidates = retrieve_candidates(convo_id, query, RETRIEVE_CANDIDATES)
    if not candidates:
        return []

    if RERANK_ENABLED:
        ranked = rerank_with_bge(query, candidates)
    else:
        ranked = [(doc, None) for doc in candidates]

    ranked = ranked[: RERANK_TOP_K]
    ranked_with_neighbors = expand_with_neighbors(vs, ranked)

    if len(ranked_with_neighbors) > CONTEXT_CHUNK_CAP:
        ranked_with_neighbors = ranked_with_neighbors[:CONTEXT_CHUNK_CAP]

    return ranked_with_neighbors


def _context_entry_from_doc(doc: Document, score: Optional[float]) -> Dict[str, Any]:
    content = (doc.page_content or "").strip()
    meta = doc.metadata or {}
    return {
        "source": meta.get("source") or meta.get("doc_id") or "unknown",
        "doc_id": meta.get("doc_id"),
        "chunk_index": meta.get("chunk_index"),
        "score": score,
        "content": content,
        "lines": content.count("\n") + 1 if content else 0,
        "chars": len(content),
        "bytes": len(content.encode("utf-8", errors="ignore")),
        "est_tokens": _estimate_tokens(content),
        "source_type": "rag",
    }


def _context_text_block(entries: List[Dict[str, Any]], header: str) -> str:
    lines: List[str] = [header, ""]
    for entry in entries:
        source = entry.get("source", "unknown")
        body = entry.get("content", "").strip()
        lines.append(f"--- {source} ---")
        lines.append(body)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_rag_context(convo_id: str, query: str) -> Tuple[str, List[Dict[str, Any]]]:
    ranked_with_neighbors = retrieve_reranked(convo_id, query)
    if not ranked_with_neighbors:
        return "", []

    entries = [_context_entry_from_doc(doc, score) for doc, score in ranked_with_neighbors]

    total_chars = sum(e["chars"] for e in entries)
    logger.info(
        (
            "RAG retrieved %d docs (convo=%s query_len=%d total_chars=%d top=%s)"
        ),
        len(entries),
        convo_id,
        len(query or ""),
        total_chars,
        [e.get("source") for e in entries[:5]],
    )

    context_text = _context_text_block(entries, "# Relevant repository context (local LM Studio RAG)")
    return context_text, entries


def build_manual_context(manual_items: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    if not manual_items:
        return "", []

    entries: List[Dict[str, Any]] = []
    for item in manual_items:
        content = (item.get("content") or "").strip()
        src = item.get("path") or item.get("source") or "manual"
        entry = {
            "source": src,
            "doc_id": src,
            "chunk_index": None,
            "score": item.get("score"),
            "content": content,
            "lines": content.count("\n") + 1 if content else 0,
            "chars": len(content),
            "bytes": len(content.encode("utf-8", errors="ignore")),
            "est_tokens": _estimate_tokens(content),
            "source_type": item.get("source_type", "manual"),
        }
        entries.append(entry)

    header = "# Manually selected context"
    context_text = _context_text_block(entries, header)
    total_bytes = sum(e["bytes"] for e in entries)
    total_tokens = sum(e["est_tokens"] for e in entries)
    logger.info(
        "Manual context used (items=%d bytes=%d est_tokens=%d sources=%s)",
        len(entries),
        total_bytes,
        total_tokens,
        [e.get("source") for e in entries],
    )
    return context_text, entries


def rank_paths_against_query(paths: List[Path], query: str) -> List[Tuple[Path, float]]:
    """Rank candidate file paths against the query using the embed model.

    If embedding fails, falls back to path-length heuristic.
    """
    if not paths:
        return []

    try:
        query_vec = embeddings.embed_query(query)
        contents: List[str] = []
        for p in paths:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = ""
            if len(text) > 4000:
                text = text[:4000]
            contents.append(text)

        doc_vecs = embeddings.embed_documents(contents)
        ranked = []
        for path_obj, vec in zip(paths, doc_vecs):
            score = _compute_cosine(query_vec, vec)
            ranked.append((path_obj, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
    except Exception:
        logger.exception("Failed to rank paths; falling back to heuristic")
        return sorted([(p, 0.0) for p in paths], key=lambda x: len(str(x[0])))


def get_context(
    convo_id: str,
    query: str,
    manual_items: Optional[List[Dict[str, Any]]] = None,
    allow_rag: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    if manual_items:
        return build_manual_context(manual_items)

    if not allow_rag:
        return "", []

    return build_rag_context(convo_id, query)
