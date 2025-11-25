# backend/rag_lmstudio.py

import logging
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Any

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_EMBED_MODEL = os.getenv("LMSTUDIO_EMBED_MODEL", "nomic-embed-text-v1.5")

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

    docs: List[Document] = []
    skipped_large = 0
    skipped_load_fail = 0

    for src in _iter_source_files(temp_dir):
        # Skip huge files (just in case)
        if src.stat().st_size > 500_000:
            skipped_large += 1
            continue

        try:
            loader = TextLoader(str(src), encoding="utf-8", autodetect_encoding=True)
            raw_docs = loader.load()
            for d in raw_docs:
                d.metadata["source"] = str(src.relative_to(temp_dir))
            docs.extend(raw_docs)
        except Exception:
            # Best-effort: if a file blows up, we just skip it
            skipped_load_fail += 1
            continue

    if not docs:
        return "No suitable source files found in ZIP."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\nclass ", "\ndef ", "\n\n", "\n", " "],
    )

    chunks = splitter.split_documents(docs)

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


def get_rag_context(convo_id: str, query: str, k: int = 6) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Retrieve top-k relevant chunks for a given query and convo_id.
    Returns a markdown-formatted string you can prepend to the user question,
    plus a metadata list describing the sources.
    """
    vs = _lazy_load_vectorstore(convo_id)
    if vs is None:
        return "", []  # no repo yet; let council behave normally

    retriever = vs.as_retriever(
        search_kwargs={"k": k, "score_threshold": 0.68},
    )
    docs = retriever.invoke(query)

    if not docs:
        return "", []  # nothing good enough

    sources_meta: List[Dict[str, Any]] = []
    total_chars = 0
    for d in docs:
        content = d.page_content or ""
        total_chars += len(content)
        sources_meta.append(
            {
                "source": d.metadata.get("source", "unknown"),
                "lines": content.count("\n") + 1 if content else 0,
                "chars": len(content),
            }
        )

    logger.info(
        (
            "RAG retrieved %d docs "
            "(convo=%s query_len=%d total_chars=%d sources=%s)"
        ),
        len(docs),
        convo_id,
        len(query or ""),
        total_chars,
        [m["source"] for m in sources_meta],
    )

    lines: List[str] = [
        "# Relevant repository context (local LM Studio RAG)\n",
    ]

    for d in docs:
        source = d.metadata.get("source", "unknown")
        lines.append(f"--- {source} ---")
        lines.append(d.page_content.strip())
        lines.append("")  # blank line between docs

    return "\n".join(lines).strip() + "\n", sources_meta
