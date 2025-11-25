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
from backend.rag_lmstudio import index_repo_zip, get_rag_context
import os

from pydantic import BaseModel
from typing import List, Dict, Any
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

    # ---- Indubitable: fetch RAG context for this conversation + query ----
    rag_context, rag_sources = await asyncio.to_thread(
        get_rag_context, conversation_id, user_content
    )

    if rag_context:
        augmented_content = f"{rag_context}\n\nUser question: {user_content}"
        logger.info(
            "RAG context appended (convo=%s user_len=%d ctx_chars=%d preview='%s')",
            conversation_id,
            len(user_content),
            len(rag_context),
            rag_context.replace("\n", " ")[:200],
        )
    else:
        augmented_content = user_content
        rag_sources = []

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
    )

    # Return the complete response with metadata
    return {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage3": stage3_result,
        "metadata": metadata,
        "rag_context_sources": rag_sources,
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

            # ---- Indubitable: fetch RAG context ----
            rag_context, rag_sources = await asyncio.to_thread(
                get_rag_context, conversation_id, user_content
            )

            if rag_context:
                augmented_content = (
                    f"{rag_context}\n\nUser question: {user_content}"
                )
                logger.info(
                    "RAG context appended (stream) (convo=%s user_len=%d ctx_chars=%d preview='%s')",
                    conversation_id,
                    len(user_content),
                    len(rag_context),
                    rag_context.replace("\n", " ")[:200],
                )
            else:
                augmented_content = user_content
                rag_sources = []

            # Emit context info early so the UI can display it
            if rag_sources:
                yield "data: " + json.dumps({"type": "rag_context", "data": rag_sources}) + "\n\n"

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
