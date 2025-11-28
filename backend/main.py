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
    _estimate_tokens,
    _iter_source_files,
    rank_paths_against_query,
    build_worktree_snapshot,
    index_repo_dir,
)

CONTROL_PROMPT = (
    "\n\n# Retrieval guidance\n"
    "If the provided context seems incomplete or missing related files or functions, "
    "explicitly say what seems missing (by filename or concept) and ask the user to provide it."
)
import os

from pydantic import BaseModel, Field
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
from .config import (
    COUNCIL_MODELS,
    MODEL_CONTEXT_LIMITS,
    CONTEXT_SAFETY_MARGIN,
    OUTPUT_TOKEN_ALLOWANCE,
    CHAIRMAN_MODEL,
)
from .openrouter import query_model
from .storage import reset_conversation
from .rag_lmstudio import _load_manifest

SETTINGS_PATH = Path("data/config.json")

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
    mode: str = "baseline"


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    content: str
    manual_context: List[Dict[str, Any]] | None = None
    # Future: accept structured directives; currently parsed inline.


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


async def _summarize_context_for_budget(
    user_question: str,
    context_block: str,
    target_tokens: int,
) -> str:
    """
    Use the Chairman model to compress context to fit within a tighter budget.
    """
    prompt = (
        "You are the Chairman of an LLM council. Summarize the provided context so it can be fed to"
        " another model with a smaller input window. Keep critical facts, constraints, and code/API"
        " signatures. Prefer bullet points. Include source hints (filenames/sections) when present."
        f" Fit the context portion into roughly {target_tokens} tokens or less. Do not omit key safety"
        " constraints or numbers. Return only the compressed context, not an answer."
        f"\n\nUser question:\n{user_question}\n\nContext to compress:\n{context_block}"
    )
    resp = await query_model(
        CHAIRMAN_MODEL,
        [{"role": "user", "content": prompt}],
        timeout=90.0,
    )
    return resp.get("content", "") if resp else ""


async def build_budgeted_prompts(
    user_content: str,
    context_block: str,
    rag_used: bool,
    force_summarize: bool = False,
    budget_override: int | None = None,
    extra_instructions: str = "",
    council_models: List[str] | None = None,
) -> tuple[str, Dict[str, str]]:
    """
    Build the base prompt and per-model overrides that fit within each model's context window.
    """
    tail = f"\n\n{extra_instructions}" if extra_instructions else ""
    base_prompt = (
        f"{context_block}{CONTROL_PROMPT if rag_used else ''}\n\nUser question: {user_content}{tail}"
        if context_block
        else f"{user_content}{tail}"
    )

    if not context_block:
        return base_prompt, {}

    base_tokens = _estimate_tokens(base_prompt)
    per_model_prompts: Dict[str, str] = {}
    summary_cache: Dict[int, str] = {}

    user_section_tokens = _estimate_tokens(f"User question: {user_content}")

    # If summarization is forced, compute a shared target using the smallest model window.
    forced_target = None
    if force_summarize:
        budgets = []
        for model in models:
            limit = MODEL_CONTEXT_LIMITS.get(model)
            if not limit:
                continue
            budget = int(limit * CONTEXT_SAFETY_MARGIN) - OUTPUT_TOKEN_ALLOWANCE
            if budget > 0:
                budgets.append(budget)
        if budgets:
            forced_budget = min(budgets)
            forced_target = max(500, forced_budget - user_section_tokens)

    models = council_models or COUNCIL_MODELS

    for model in models:
        limit = MODEL_CONTEXT_LIMITS.get(model)
        if not limit:
            continue

        budget = int(limit * CONTEXT_SAFETY_MARGIN) - OUTPUT_TOKEN_ALLOWANCE
        if budget_override:
            budget = min(budget, budget_override)
        if budget <= 0:
            continue

        # Skip summarization if under budget and not forced
        if not force_summarize and base_tokens <= budget:
            continue

        target_ctx_tokens = forced_target or max(500, budget - user_section_tokens)
        cache_key = target_ctx_tokens
        if cache_key not in summary_cache:
            compressed = await _summarize_context_for_budget(
                user_content,
                context_block,
                target_ctx_tokens,
            )
            summary_cache[cache_key] = compressed or context_block

        compressed_context = summary_cache[cache_key]
        per_model_prompts[model] = (
            f"{compressed_context}{CONTROL_PROMPT if rag_used else ''}\n\nUser question: {user_content}{tail}"
        )

    return base_prompt, per_model_prompts


class ConversationMetadata(BaseModel):
    """Conversation metadata for list view."""
    id: str
    created_at: str
    title: str
    message_count: int
    mode: str | None = "baseline"


class Conversation(BaseModel):
    """Full conversation with all messages."""
    id: str
    created_at: str
    title: str
    messages: List[Dict[str, Any]]
    mode: str = "baseline"


class ParsedDirectives(BaseModel):
    skip_rag: bool = False
    force_summarize: bool = False
    budget_override: int | None = None
    trace: bool = False
    length_hint: str | None = None  # "short" | "detailed"
    cite: bool = False
    noexecute: bool = False
    reset: bool = False
    temp_override: float | None = None
    maxtokens_override: int | None = None
    safety: str | None = None
    warnings: List[str] = Field(default_factory=list)


def _parse_directives(raw_text: str) -> tuple[str, ParsedDirectives]:
    """
    Extract @directives from the raw user text and return cleaned text + flags.
    """
    words = raw_text.split()
    remaining: List[str] = []
    flags = ParsedDirectives()

    def _try_parse_int(val: str) -> int | None:
        try:
            return int(val)
        except Exception:
            return None

    def _try_parse_float(val: str) -> float | None:
        try:
            return float(val)
        except Exception:
            return None

    i = 0
    while i < len(words):
        word = words[i]
        if not word.startswith("@"):
            remaining.append(word)
            i += 1
            continue

        token = word[1:]
        lower = token.lower()

        def _consume_next_as_value() -> str | None:
            nonlocal i
            if i + 1 < len(words) and not words[i + 1].startswith("@"):
                i += 1
                return words[i]
            return None

        def _split_inline_val(tok: str) -> str | None:
            for sep in ("=", ":"):
                if sep in tok:
                    return tok.split(sep, 1)[1]
            return None

        handled = True

        if lower in {"norag", "raw"}:
            flags.skip_rag = True
        elif lower == "summarize":
            flags.force_summarize = True
        elif lower.startswith("tokenbudget"):
            val = _split_inline_val(token) or _consume_next_as_value()
            parsed = _try_parse_int(val) if val else None
            if parsed:
                flags.budget_override = parsed
            else:
                flags.warnings.append("Invalid @tokenbudget value")
        elif lower in {"trace", "debug"}:
            flags.trace = True
        elif lower == "short":
            flags.length_hint = "short"
        elif lower == "detailed":
            flags.length_hint = "detailed"
        elif lower == "cite":
            flags.cite = True
        elif lower == "noexecute":
            flags.noexecute = True
        elif lower == "reset":
            flags.reset = True
        elif lower.startswith("temp"):
            val = _split_inline_val(token) or _consume_next_as_value()
            parsed = _try_parse_float(val) if val else None
            if parsed is not None and 0 <= parsed <= 1:
                flags.temp_override = parsed
            else:
                flags.warnings.append("Invalid @temp value; must be 0-1")
        elif lower.startswith("maxtokens"):
            val = _split_inline_val(token) or _consume_next_as_value()
            parsed = _try_parse_int(val) if val else None
            if parsed:
                flags.maxtokens_override = parsed
            else:
                flags.warnings.append("Invalid @maxtokens value")
        elif lower in {"safe", "relaxed"}:
            flags.safety = lower
        else:
            handled = False

        if not handled:
            # Unrecognized directive; keep as-is in text
            remaining.append(word)

        i += 1

    cleaned = " ".join(remaining).strip()
    return cleaned, flags


def _directive_instruction_text(flags: ParsedDirectives) -> str:
    lines: List[str] = []
    if flags.length_hint == "short":
        lines.append("Answer concisely (<= ~5 sentences unless code is needed).")
    elif flags.length_hint == "detailed":
        lines.append("Provide a detailed answer with rationale and steps.")
    if flags.cite:
        lines.append("When using provided context, include inline citations like [file:line].")
    if flags.noexecute:
        lines.append("Do not invoke tools or external actions; reasoning only.")
    return "\n".join(lines)


def _mode_instruction_text(mode: str) -> str:
    mode = (mode or "baseline").lower()
    if mode == "round_robin":
        return "Mode: Round Robin. Consider prior drafts, improve accuracy/clarity, keep useful detail."
    if mode == "fight":
        return "Mode: Fight. Provide your best answer; be crisp about risks; be ready for critique/defense."
    if mode == "stacks":
        return "Mode: Stacks. Provide an answer suitable for later merge/judge steps; retain optionality."
    if mode == "complex_iterative":
        return "Mode: Complex Iterative. Extract constraints and propose next steps succinctly."
    if mode == "complex_questioning":
        return "Mode: Complex Questioning. Provide answer and note uncertainties for later reflection."
    return ""


def load_runtime_settings() -> Dict[str, Any]:
    """Load persisted settings merged with defaults."""
    defaults = {
        "council_models": COUNCIL_MODELS,
        "chairman_model": CHAIRMAN_MODEL,
    }
    if not SETTINGS_PATH.exists():
        return defaults
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        defaults.update({k: v for k, v in data.items() if v is not None})
    except Exception:
        pass
    return defaults


def save_runtime_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    current = load_runtime_settings()
    allowed_keys = {"council_models", "chairman_model"}
    for k, v in payload.items():
        if k in allowed_keys:
            current[k] = v
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return current


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
    conversation = storage.create_conversation(conversation_id, request.mode)
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

    settings = load_runtime_settings()
    council_models = settings.get("council_models", COUNCIL_MODELS)
    chairman_model = settings.get("chairman_model", CHAIRMAN_MODEL)

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0
    settings = load_runtime_settings()
    council_models = settings.get("council_models", COUNCIL_MODELS)
    chairman_model = settings.get("chairman_model", CHAIRMAN_MODEL)

    # Raw user content as typed
    user_content_raw = request.content

    cleaned_content, directives = _parse_directives(user_content_raw)
    user_content = cleaned_content or user_content_raw

    if directives.reset:
        reset_conversation(conversation_id)
        return {
            "stage1": [],
            "stage2": [],
            "stage3": {"model": "system", "response": "Conversation reset as requested."},
            "metadata": {"directives": directives.dict(), "warnings": directives.warnings},
            "context_sources": [],
            "directives": directives.dict(),
            "warnings": directives.warnings,
        }

    # ---- Context: manual overrides (if provided) or RAG retrieval ----
    manual_context = request.manual_context or []
    context_block, context_sources = await asyncio.to_thread(
        get_context, conversation_id, user_content, manual_context, not directives.skip_rag
    )

    rag_used = len(manual_context) == 0 and not directives.skip_rag
    if not context_block:
        context_sources = []

    instruction_text_parts = [
        _directive_instruction_text(directives),
        _mode_instruction_text(conversation.get("mode", "baseline")),
    ]
    instruction_text = "\n".join([p for p in instruction_text_parts if p])

    augmented_content, per_model_prompts = await build_budgeted_prompts(
        user_content,
        context_block,
        rag_used,
        force_summarize=directives.force_summarize,
        budget_override=directives.budget_override,
        extra_instructions=instruction_text,
        council_models=council_models,
    )
    logger.info(
        "Context appended (convo=%s user_len=%d ctx_chars=%d budgeted=%s skip_rag=%s force_sum=%s preview='%s')",
        conversation_id,
        len(user_content),
        len(context_block),
        bool(per_model_prompts),
        directives.skip_rag,
        directives.force_summarize,
        context_block.replace("\n", " ")[:200] if context_block else "",
    )

    # Add user message (store original text, not augmented)
    storage.add_user_message(conversation_id, user_content)

    # If this is the first message, generate a title (from original question)
    if is_first_message:
        title = await generate_conversation_title(user_content)
        storage.update_conversation_title(conversation_id, title)

    # Run the 3-stage council process on the augmented content
    stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
        augmented_content,
        per_model_prompts if per_model_prompts else None,
        mode=conversation.get("mode", "baseline"),
        council_models=council_models,
        chairman_model=chairman_model,
    )
    metadata["directives"] = directives.dict()
    metadata["warnings"] = directives.warnings
    metadata["mode"] = conversation.get("mode", "baseline")

    # Add assistant message with all stages
    storage.add_assistant_message(
        conversation_id,
        stage1_results,
        stage2_results,
        stage3_result,
        context_sources,
        metadata={
            "label_to_model": metadata.get("label_to_model"),
            "aggregate_rankings": metadata.get("aggregate_rankings"),
            "directives": directives.dict(),
            "warnings": directives.warnings,
            "mode": conversation.get("mode", "baseline"),
            "chairman_model": chairman_model,
            "council_models": council_models,
            "steps": metadata.get("steps"),
        },
    )

    # Return the complete response with metadata
    return {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage3": stage3_result,
        "metadata": metadata,
        "context_sources": context_sources,
        "directives": directives.dict(),
        "warnings": directives.warnings,
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
            user_content_raw = request.content
            cleaned_content, directives = _parse_directives(user_content_raw)
            user_content = cleaned_content or user_content_raw

            # Refresh settings per-request in case they changed
            settings_local = load_runtime_settings()
            council_models_local = settings_local.get("council_models", COUNCIL_MODELS)
            chairman_model_local = settings_local.get("chairman_model", CHAIRMAN_MODEL)

            if directives.reset:
                reset_conversation(conversation_id)
                yield "data: " + json.dumps({"type": "reset", "message": "Conversation reset as requested."}) + "\n\n"
                yield f"data: {json.dumps({'type': 'complete'})}\n\n"
                return

            manual_context = request.manual_context or []

            # ---- Context: manual overrides (if provided) or RAG retrieval ----
            context_block, context_sources = await asyncio.to_thread(
                get_context, conversation_id, user_content, manual_context, not directives.skip_rag
            )

            rag_used = len(manual_context) == 0 and not directives.skip_rag

            if not context_block:
                context_sources = []

            instruction_text_parts = [
                _directive_instruction_text(directives),
                _mode_instruction_text(conversation.get("mode", "baseline")),
            ]
            instruction_text = "\n".join([p for p in instruction_text_parts if p])

            augmented_content, per_model_prompts = await build_budgeted_prompts(
                user_content,
                context_block,
                rag_used,
                force_summarize=directives.force_summarize,
                budget_override=directives.budget_override,
                extra_instructions=instruction_text,
                council_models=council_models_local,
            )
            logger.info(
                "Context appended (stream) (convo=%s user_len=%d ctx_chars=%d budgeted=%s skip_rag=%s force_sum=%s preview='%s')",
                conversation_id,
                len(user_content),
                len(context_block),
                bool(per_model_prompts),
                directives.skip_rag,
                directives.force_summarize,
                context_block.replace("\n", " ")[:200] if context_block else "",
            )

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

            # Unified mode runner (returns stage1/2/3 + metadata)
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            stage1_results, stage2_results, stage3_result, mode_metadata = await run_full_council(
                augmented_content,
                per_model_prompts if per_model_prompts else None,
                mode=conversation.get("mode", "baseline"),
                council_models=council_models_local,
                chairman_model=chairman_model_local,
            )
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

            if not stage1_results:
                err_msg = "No council responses received; check model availability or API key."
                yield f"data: {json.dumps({'type': 'error', 'message': err_msg})}\n\n"
                storage.add_assistant_message(
                    conversation_id,
                    [],
                    [],
                    {"model": "system", "response": err_msg},
                    context_sources,
                    metadata={
                        "directives": directives.dict(),
                        "warnings": directives.warnings,
                        "mode": conversation.get("mode", "baseline"),
                    },
                )
                yield f"data: {json.dumps({'type': 'complete'})}\n\n"
                return

            if stage2_results:
                yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
                stage_metadata = {
                    "label_to_model": mode_metadata.get("label_to_model"),
                    "aggregate_rankings": mode_metadata.get("aggregate_rankings"),
                    "directives": directives.dict(),
                    "warnings": directives.warnings,
                    "mode": conversation.get("mode", "baseline"),
                    "chairman_model": chairman_model_local,
                    "council_models": council_models_local,
                }
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "stage2_complete",
                            "data": stage2_results,
                            "metadata": stage_metadata,
                        }
                    )
                    + "\n\n"
                )

            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
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
                metadata={
                    "label_to_model": mode_metadata.get("label_to_model"),
                    "aggregate_rankings": mode_metadata.get("aggregate_rankings"),
                    "directives": directives.dict(),
                    "warnings": directives.warnings,
                    "mode": conversation.get("mode", "baseline"),
                    "chairman_model": chairman_model_local,
                    "council_models": council_models_local,
                    "steps": mode_metadata.get("steps"),
                },
            )

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            logger.exception("Streaming failure (convo=%s)", conversation_id)
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


@app.post("/api/conversations/{conversation_id}/reindex_git")
async def reindex_git(conversation_id: str, include_untracked: bool | None = None):
    """
    Index the current git working tree for this conversation.
    """
    include_untracked = (
        INDEX_INCLUDE_UNTRACKED if include_untracked is None else bool(include_untracked)
    )
    msg = build_worktree_snapshot(conversation_id, include_untracked=include_untracked)
    result = index_repo_dir(Path("temp_repos") / conversation_id, conversation_id)
    return {
        "status": "success",
        "message": f"{msg} {result}",
        "include_untracked": include_untracked,
    }


@app.get("/api/index_manifest")
async def index_manifest():
    """Return the current index manifest (per conversation)."""
    return _load_manifest()


@app.get("/api/settings")
async def get_settings():
    """Return runtime settings (council/chairman)."""
    return load_runtime_settings()


class UpdateSettingsRequest(BaseModel):
    council_models: List[str] | None = None
    chairman_model: str | None = None


@app.post("/api/settings")
async def update_settings(payload: UpdateSettingsRequest):
    data = payload.dict(exclude_none=True)
    saved = save_runtime_settings(data)
    return saved


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
