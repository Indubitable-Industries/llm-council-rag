"""FastAPI backend for LLM Council."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
import logging

""" Indubitable Fork Additions"""
import tempfile
import time
from fastapi import UploadFile, File
from pathlib import Path
from backend.rag_lmstudio import (
    index_repo_zip,
    get_context,
    _iter_source_files,
    rank_paths_against_query,
)

CONTROL_PROMPT = (
    "\n\n# Retrieval guidance\n"
    "If the provided context seems incomplete or missing related files or functions, "
    "explicitly say what seems missing (by filename or concept) and ask the user to provide it."
)
import os

from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
import uuid
import json
import asyncio

from . import storage
from .council import (
    run_full_council,
    generate_conversation_title,
    stage1_collect_responses,
    stage2_collect_rankings,
    stage3_synthesize_final,
    calculate_aggregate_rankings,
)

app = FastAPI(title="LLM Council API")
logger = logging.getLogger(__name__)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    pass


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    content: str
    manual_context: List[Dict[str, Any]] | None = None


def _repo_root(conversation_id: str) -> Path:
    return Path("temp_repos") / conversation_id


def _safe_resolve(root: Path, target: Path) -> Path:
    root_resolved = root.resolve()
    target_resolved = target.resolve()
    if not str(target_resolved).startswith(str(root_resolved)):
        raise HTTPException(status_code=400, detail="Path is outside repository root")
    return target_resolved


def _build_tree(node: Path, base: Path) -> Dict[str, Any]:
    rel_path = node.relative_to(base)
    if node.is_dir():
        children = sorted(node.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        return {
            "type": "dir",
            "name": node.name,
            "path": str(rel_path),
            "children": [_build_tree(child, base) for child in children],
        }
    else:
        return {
            "type": "file",
            "name": node.name,
            "path": str(rel_path),
        }


class ConversationMetadata(BaseModel):
    """Conversation metadata for list view."""
    id: str
    created_at: str
    title: str
    message_count: int


class Conversation(BaseModel):
    """Full conversation with all messages."""
    id: str
    created_at: str
    title: str
    messages: List[Dict[str, Any]]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "LLM Council API"}


@app.get("/api/conversations", response_model=List[ConversationMetadata])
async def list_conversations():
    """List all conversations (metadata only)."""
    return storage.list_conversations()


@app.post("/api/conversations", response_model=Conversation)
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    conversation = storage.create_conversation(conversation_id)
    return conversation


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get a specific conversation with all its messages."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and run the 3-stage council process.
    Returns the complete response with all stages.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    # Raw user content as typed
    user_content = request.content

    # ---- Context: manual overrides (if provided) or RAG retrieval ----
    manual_context = request.manual_context or []
    context_block, context_sources = await asyncio.to_thread(
        get_context, conversation_id, user_content, manual_context
    )

    rag_used = len(manual_context) == 0
    if context_block:
        extra = CONTROL_PROMPT if rag_used else ""
        augmented_content = f"{context_block}{extra}\n\nUser question: {user_content}"
        logger.info(
            "Context appended (convo=%s user_len=%d ctx_chars=%d preview='%s')",
            conversation_id,
            len(user_content),
            len(context_block),
            context_block.replace("\n", " ")[:200],
        )
    else:
        augmented_content = user_content
        context_sources = []

    # Add user message (store original text, not augmented)
    storage.add_user_message(conversation_id, user_content)

    # If this is the first message, generate a title (from original question)
    if is_first_message:
        title = await generate_conversation_title(user_content)
        storage.update_conversation_title(conversation_id, title)

    # Run the 3-stage council process on the augmented content
    stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
        augmented_content
    )

    # Add assistant message with all stages
    storage.add_assistant_message(
        conversation_id,
        stage1_results,
        stage2_results,
        stage3_result,
        context_sources,
    )

    # Return the complete response with metadata
    return {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage3": stage3_result,
        "metadata": metadata,
        "context_sources": context_sources,
    }


@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and stream the 3-stage council process.
    Returns Server-Sent Events as each stage completes.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    async def event_generator():
        try:
            # Raw user content as typed
            user_content = request.content

            manual_context = request.manual_context or []

            # ---- Context: manual overrides (if provided) or RAG retrieval ----
            context_block, context_sources = await asyncio.to_thread(
                get_context, conversation_id, user_content, manual_context
            )

            rag_used = len(manual_context) == 0

            if context_block:
                extra = CONTROL_PROMPT if rag_used else ""
                augmented_content = (
                    f"{context_block}{extra}\n\nUser question: {user_content}"
                )
                logger.info(
                    "Context appended (stream) (convo=%s user_len=%d ctx_chars=%d preview='%s')",
                    conversation_id,
                    len(user_content),
                    len(context_block),
                    context_block.replace("\n", " ")[:200],
                )
            else:
                augmented_content = user_content
                context_sources = []

            # Emit context info early so the UI can display it
            if context_sources:
                yield "data: " + json.dumps({"type": "rag_context", "data": context_sources}) + "\n\n"

            # Add user message (store original)
            storage.add_user_message(conversation_id, user_content)

            # Start title generation in parallel (use original content)
            title_task = None
            if is_first_message:
                title_task = asyncio.create_task(
                    generate_conversation_title(user_content)
                )

            # Stage 1: Collect responses
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            stage1_results = await stage1_collect_responses(augmented_content)
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

            # Stage 2: Collect rankings
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
            stage2_results, label_to_model = await stage2_collect_rankings(
                augmented_content, stage1_results
            )
            aggregate_rankings = calculate_aggregate_rankings(
                stage2_results, label_to_model
            )
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "stage2_complete",
                        "data": stage2_results,
                        "metadata": {
                            "label_to_model": label_to_model,
                            "aggregate_rankings": aggregate_rankings,
                        },
                    }
                )
                + "\n\n"
            )

            # Stage 3: Synthesize final answer
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
            stage3_result = await stage3_synthesize_final(
                augmented_content, stage1_results, stage2_results
            )
            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"

            # Wait for title generation if it was started
            if title_task:
                title = await title_task
                storage.update_conversation_title(conversation_id, title)
                yield (
                    "data: "
                    + json.dumps(
                        {"type": "title_complete", "data": {"title": title}}
                    )
                    + "\n\n"
                )

            # Save complete assistant message
            storage.add_assistant_message(
                conversation_id,
                stage1_results,
                stage2_results,
                stage3_result,
                context_sources,
            )

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            # Send error event
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/conversations/{conversation_id}/repo_tree")
async def get_repo_tree(conversation_id: str):
    root = _repo_root(conversation_id)
    if not root.exists():
        return []

    tree = _build_tree(root, root)
    return tree.get("children", []) if tree else []


@app.get("/api/conversations/{conversation_id}/file")
async def get_file_contents(conversation_id: str, path: str):
    root = _repo_root(conversation_id)
    if not root.exists():
        raise HTTPException(status_code=404, detail="Repository not uploaded yet")

    target = _safe_resolve(root, root / path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        content = target.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")

    return {
        "path": path,
        "content": content,
        "lines": content.count("\n") + 1 if content else 0,
        "bytes": len(content.encode("utf-8", errors="ignore")),
    }


@app.get("/api/conversations/{conversation_id}/resolve_path")
async def resolve_path(conversation_id: str, q: str, user_query: str | None = None, limit: int = 5):
    root = _repo_root(conversation_id)
    if not root.exists():
        return {"matches": []}

    matches: List[Path] = []
    for path_obj in _iter_source_files(root):
        rel = path_obj.relative_to(root)
        if q.lower() in str(rel).lower():
            matches.append(path_obj)

    if not matches:
        return {"matches": []}

    ranked: List[Tuple[Path, float]]
    if len(matches) > 1 and user_query:
        ranked = rank_paths_against_query(matches, user_query)
    else:
        ranked = [(m, 0.0) for m in matches]

    ranked = ranked[:limit]

    results = []
    for path_obj, score in ranked:
        rel = path_obj.relative_to(root)
        try:
            content = path_obj.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            content = ""
        results.append(
            {
                "path": str(rel),
                "score": score,
                "content": content,
                "lines": content.count("\n") + 1 if content else 0,
                "bytes": len(content.encode("utf-8", errors="ignore")),
            }
        )

    return {"matches": results}


@app.get("/api/conversations/{conversation_id}/search")
async def search_repo(conversation_id: str, q: str, limit: int = 3):
    root = _repo_root(conversation_id)
    if not root.exists():
        return {"results": []}

    q_lower = q.lower()
    results = []
    for path_obj in _iter_source_files(root):
        try:
            content = path_obj.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        idx = content.lower().find(q_lower)
        if idx == -1:
            continue

        start = max(0, idx - 120)
        end = min(len(content), idx + 120)
        snippet = content[start:end]

        results.append(
            {
                "path": str(path_obj.relative_to(root)),
                "snippet": snippet,
                "lines": content.count("\n") + 1,
                "bytes": len(content.encode("utf-8", errors="ignore")),
            }
        )

        if len(results) >= limit:
            break

    return {"results": results}


@app.post("/api/conversations/{conversation_id}/upload_repo")
async def upload_repo(conversation_id: str, file: UploadFile = File(...)):
    """
    Upload a repository as a .zip and index it for this conversation.
    Uses local LM Studio embeddings + FAISS (see backend/rag_lmstudio.py).
    """
    if not file.filename.lower().endswith(".zip"):
        return {"status": "error", "message": "Please upload a .zip file."}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    start = time.monotonic()
    try:
        msg = index_repo_zip(tmp_path, conversation_id)
        duration = time.monotonic() - start
        return {"status": "success", "message": f"{msg} (took {duration:.2f}s)"}
    except Exception as e:
        # Surface indexing errors to the client instead of 500ing.
        duration = time.monotonic() - start
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Failed to index repo: {str(e)} (after {duration:.2f}s)",
            },
        )
    finally:
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
