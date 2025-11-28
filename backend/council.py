"""3-stage LLM Council orchestration."""

import asyncio
from typing import List, Dict, Any, Tuple, Optional, Callable
from .openrouter import query_models_parallel, query_model
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL


async def stage1_collect_responses(
    user_query: str,
    per_model_prompts: Optional[Dict[str, str]] = None,
    council_models: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.

    Args:
        user_query: The user's question

    Returns:
        List of dicts with 'model' and 'response' keys
    """
    messages = [{"role": "user", "content": user_query}]

    models = council_models or COUNCIL_MODELS

    if per_model_prompts:
        tasks = []
        for model in models:
            prompt = per_model_prompts.get(model, user_query) if per_model_prompts else user_query
            tasks.append(
                query_model(
                    model,
                    [{"role": "user", "content": prompt}],
                )
            )
        responses_list = await asyncio.gather(*tasks)
        responses = {model: resp for model, resp in zip(models, responses_list)}
    else:
        # Query all models in parallel with the same prompt
        responses = await query_models_parallel(models, messages)

    # Format results
    stage1_results = []
    for model, response in responses.items():
        if response is not None:  # Only include successful responses
            stage1_results.append({
                "model": model,
                "response": response.get('content', '')
            })

    return stage1_results


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    council_models: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: Each model ranks the anonymized responses.

    Args:
        user_query: The original user query
        stage1_results: Results from Stage 1

    Returns:
        Tuple of (rankings list, label_to_model mapping)
    """
    # Create anonymized labels for responses (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...

    # Create mapping from label to model name
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }

    # Build the ranking prompt
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
    ])

    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

    messages = [{"role": "user", "content": ranking_prompt}]

    # Get rankings from all council models in parallel
    models = council_models or COUNCIL_MODELS
    responses = await query_models_parallel(models, messages)

    # Format results
    stage2_results = []
    for model, response in responses.items():
        if response is not None:
            full_text = response.get('content', '')
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": model,
                "ranking": full_text,
                "parsed_ranking": parsed
            })

    return stage2_results, label_to_model


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    chairman_model: str = CHAIRMAN_MODEL,
) -> Dict[str, Any]:
    """
    Stage 3: Chairman synthesizes final response.

    Args:
        user_query: The original user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2

    Returns:
        Dict with 'model' and 'response' keys
    """
    # Build comprehensive context for chairman
    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results
    ])

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

    messages = [{"role": "user", "content": chairman_prompt}]

    # Query the chairman model
    response = await query_model(chairman_model, messages)

    if response is None:
        # Fallback if chairman fails
        return {
            "model": chairman_model,
            "response": "Error: Unable to generate final synthesis."
        }

    return {
        "model": chairman_model,
        "response": response.get('content', '')
    }


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.

    Args:
        ranking_text: The full text response from the model

    Returns:
        List of response labels in ranked order
    """
    import re

    # Look for "FINAL RANKING:" section
    if "FINAL RANKING:" in ranking_text:
        # Extract everything after "FINAL RANKING:"
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            # Try to extract numbered list format (e.g., "1. Response A")
            # This pattern looks for: number, period, optional space, "Response X"
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
            if numbered_matches:
                # Extract just the "Response X" part
                return [re.search(r'Response [A-Z]', m).group() for m in numbered_matches]

            # Fallback: Extract all "Response X" patterns in order
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches

    # Fallback: try to find any "Response X" patterns in order
    matches = re.findall(r'Response [A-Z]', ranking_text)
    return matches


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        List of dicts with model name and average rank, sorted best to worst
    """
    from collections import defaultdict

    # Track positions for each model
    model_positions = defaultdict(list)

    for ranking in stage2_results:
        ranking_text = ranking['ranking']

        # Parse the ranking from the structured format
        parsed_ranking = parse_ranking_from_text(ranking_text)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x['average_rank'])

    return aggregate


async def generate_conversation_title(user_query: str) -> str:
    """
    Generate a short title for a conversation based on the first user message.

    Args:
        user_query: The first user message

    Returns:
        A short title (3-5 words)
    """
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]

    # Use gemini-2.5-flash for title generation (fast and cheap)
    response = await query_model("google/gemini-2.5-flash", messages, timeout=30.0)

    if response is None:
        # Fallback to a generic title
        return "New Conversation"

    title = response.get('content', 'New Conversation').strip()

    # Clean up the title - remove quotes, limit length
    title = title.strip('"\'')

    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + "..."

    return title


async def run_full_council(
    user_query: str,
    per_model_prompts: Optional[Dict[str, str]] = None,
    mode: str = "baseline",
    council_models: Optional[List[str]] = None,
    chairman_model: str = CHAIRMAN_MODEL,
) -> Tuple[List, List, Dict, Dict]:
    """
    Run the complete 3-stage council process.

    Args:
        user_query: The user's question

    Returns:
        Tuple of (stage1_results, stage2_results, stage3_result, metadata)
    """
    mode = (mode or "baseline").lower()
    runner = MODE_RUNNERS.get(mode, run_mode_baseline)
    return await runner(
        user_query=user_query,
        per_model_prompts=per_model_prompts,
        council_models=council_models,
        chairman_model=chairman_model,
    )


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------


async def run_mode_baseline(
    user_query: str,
    per_model_prompts: Optional[Dict[str, str]],
    council_models: Optional[List[str]],
    chairman_model: str,
):
    stage1_results = await stage1_collect_responses(user_query, per_model_prompts, council_models)
    if not stage1_results:
        return [], [], {
            "model": "error",
            "response": "All models failed to respond. Please try again."
        }, {"mode": "baseline"}

    stage2_results, label_to_model = await stage2_collect_rankings(user_query, stage1_results, council_models)
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    stage3_result = await stage3_synthesize_final(
        user_query,
        stage1_results,
        stage2_results,
        chairman_model=chairman_model,
    )

    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings,
        "mode": "baseline",
    }
    return stage1_results, stage2_results, stage3_result, metadata


async def run_mode_round_robin(
    user_query: str,
    per_model_prompts: Optional[Dict[str, str]],
    council_models: Optional[List[str]],
    chairman_model: str,
):
    models = council_models or COUNCIL_MODELS
    drafts: List[Dict[str, Any]] = []
    prior_text = ""
    for turn, model in enumerate(models, start=1):
        base_prompt = per_model_prompts.get(model, user_query) if per_model_prompts else user_query
        turn_prompt = (
            f"Turn {turn}/{len(models)}. You see the latest draft below. Improve accuracy and clarity; keep useful detail. "
            f"Original question: {user_query}\n\nLatest draft:\n{prior_text or '(none yet)'}"
        )
        messages = [{"role": "user", "content": f"{base_prompt}\n\n{turn_prompt}"}]
        resp = await query_model(model, messages)
        text = resp.get("content", "") if resp else ""
        drafts.append({"model": model, "response": text, "role": f"draft_turn_{turn}"})
        prior_text = text or prior_text

    if not drafts:
        return [], [], {"model": "error", "response": "Round Robin failed: no drafts produced."}, {"mode": "round_robin"}

    chair_prompt = (
        f"Final draft from round robin:\n{prior_text}\n\nOriginal question:\n{user_query}\n\n"
        "Produce the final answer building on the latest draft; fix any errors and cite context if present."
    )
    chair_resp = await query_model(chairman_model, [{"role": "user", "content": chair_prompt}])
    stage3_result = {
        "model": chairman_model,
        "response": chair_resp.get("content", "") if chair_resp else "No response from chairman."
    }

    metadata = {"mode": "round_robin", "steps": drafts}
    return drafts, [], stage3_result, metadata


async def run_mode_fight(
    user_query: str,
    per_model_prompts: Optional[Dict[str, str]],
    council_models: Optional[List[str]],
    chairman_model: str,
):
    models = council_models or COUNCIL_MODELS
    prompt_map = per_model_prompts or {}

    answers = await stage1_collect_responses(user_query, prompt_map, models)
    answers = [{"model": a["model"], "response": a["response"], "role": "answer"} for a in answers]

    critiques: List[Dict[str, Any]] = []
    for ans in answers:
        others = [a for a in answers if a["model"] != ans["model"]]
        critique_prompt = (
            f"Critique peers for question:\n{user_query}\n\nPeers:\n" +
            "\n\n".join([f"{o['model']}:\n{o['response']}" for o in others])
        )
        resp = await query_model(ans["model"], [{"role": "user", "content": critique_prompt}])
        critiques.append({
            "model": ans["model"],
            "response": resp.get("content", "") if resp else "",
            "role": "critique",
        })

    defenses: List[Dict[str, Any]] = []
    for ans in answers:
        peer_crits = [c for c in critiques if c["model"] != ans["model"]]
        defense_prompt = (
            f"Defend your answer to: {user_query}\nYour answer:\n{ans['response']}\nPeer critiques:\n" +
            "\n\n".join([f"{c['model']}:\n{c['response']}" for c in peer_crits])
        )
        resp = await query_model(ans["model"], [{"role": "user", "content": defense_prompt}])
        defenses.append({
            "model": ans["model"],
            "response": resp.get("content", "") if resp else "",
            "role": "defense",
        })

    chair_prompt = (
        f"Debate on: {user_query}\n\nAnswers:\n" +
        "\n\n".join([f"{a['model']}:\n{a['response']}" for a in answers]) +
        "\n\nCritiques:\n" +
        "\n\n".join([f"{c['model']}:\n{c['response']}" for c in critiques]) +
        "\n\nDefenses:\n" +
        "\n\n".join([f"{d['model']}:\n{d['response']}" for d in defenses]) +
        "\n\nSummarize consensus, disagreements, and provide the best combined answer."
    )
    chair_resp = await query_model(chairman_model, [{"role": "user", "content": chair_prompt}])
    stage3_result = {
        "model": chairman_model,
        "response": chair_resp.get("content", "") if chair_resp else "No response from chairman."
    }

    steps = answers + critiques + defenses
    metadata = {"mode": "fight", "steps": steps}
    return steps, [], stage3_result, metadata


async def run_mode_stacks(
    user_query: str,
    per_model_prompts: Optional[Dict[str, str]],
    council_models: Optional[List[str]],
    chairman_model: str,
):
    models = council_models or COUNCIL_MODELS
    if len(models) < 2:
        return [], [], {"model": "error", "response": "Stacks requires at least two models."}, {"mode": "stacks"}

    answers = await stage1_collect_responses(user_query, per_model_prompts, models[:2])
    answers = [{"model": a["model"], "response": a["response"], "role": "stacks_answer"} for a in answers]

    merge_prompt = (
        f"Merge two answers while preserving optionality. Cite context if needed.\n\nA:\n{answers[0]['response']}\n\nB:\n{answers[1]['response']}"
    )
    merged = await query_model(chairman_model, [{"role": "user", "content": merge_prompt}])
    merged_text = merged.get("content", "") if merged else ""
    merged_step = {"model": chairman_model, "response": merged_text, "role": "stacks_merge"}

    critics_models = models[2:] if len(models) > 2 else models[:2]
    critiques: List[Dict[str, Any]] = []
    for cm in critics_models:
        critique_prompt = (
            f"Critique the merged answer. Attack weak spots and missing context. Be concise.\n\nMerged:\n{merged_text}"
        )
        resp = await query_model(cm, [{"role": "user", "content": critique_prompt}])
        critiques.append({"model": cm, "response": resp.get("content", "") if resp else "", "role": "stacks_critique"})

    judge_prompt = (
        f"Judge the merged answer vs critiques. Note what holds and fails.\nMerged:\n{merged_text}\n\nCritiques:\n" +
        "\n\n".join([f"{c['model']}:\n{c['response']}" for c in critiques])
    )
    judge = await query_model(chairman_model, [{"role": "user", "content": judge_prompt}])
    judge_text = judge.get("content", "") if judge else ""
    judge_step = {"model": chairman_model, "response": judge_text, "role": "stacks_judge"}

    defenses: List[Dict[str, Any]] = []
    for ans in answers:
        defense_prompt = (
            f"Defend the merged answer vs critiques; fix valid issues briefly.\nMerged:\n{merged_text}\n\nCritiques:\n" +
            "\n\n".join([f"{c['model']}:\n{c['response']}" for c in critiques])
        )
        resp = await query_model(ans["model"], [{"role": "user", "content": defense_prompt}])
        defenses.append({"model": ans["model"], "response": resp.get("content", "") if resp else "", "role": "stacks_defense"})

    final_prompt = (
        f"Produce final report; present both sides; note judgment rationale.\nJudge:\n{judge_text}\n\nMerged:\n{merged_text}\n\nDefenses:\n" +
        "\n\n".join([f"{d['model']}:\n{d['response']}" for d in defenses])
    )
    final_resp = await query_model(chairman_model, [{"role": "user", "content": final_prompt}])
    final_text = final_resp.get("content", "") if final_resp else ""

    steps = answers + [merged_step] + critiques + [judge_step] + defenses
    metadata = {"mode": "stacks", "steps": steps}
    return steps, [], {"model": chairman_model, "response": final_text}, metadata


async def run_mode_complex_iterative(
    user_query: str,
    per_model_prompts: Optional[Dict[str, str]],
    council_models: Optional[List[str]],
    chairman_model: str,
):
    models = council_models or COUNCIL_MODELS
    if len(models) < 2:
        return [], [], {"model": "error", "response": "Complex Iterative needs at least two models."}, {"mode": "complex_iterative"}

    extract_model = models[0]
    expand_model = models[1]
    steps: List[Dict[str, Any]] = []
    summary = ""
    suggested = ""
    for hop in range(4):  # extract/expand twice
        if hop % 2 == 0:
            prompt = f"Extract: summarize intent and constraints; list key facts; propose the next prompt. Context:\n{user_query}\n\nPrior summary:\n{summary}\nPrior suggested:\n{suggested}"
            resp = await query_model(extract_model, [{"role": "user", "content": prompt}])
            text = resp.get("content", "") if resp else ""
            steps.append({"model": extract_model, "response": text, "role": "extract"})
            summary = text or summary
        else:
            prompt = f"Expand the prior extract; elaborate actionable detail and improve the suggested prompt.\nPrior summary:\n{summary}\nPrior suggested:\n{suggested}"
            resp = await query_model(expand_model, [{"role": "user", "content": prompt}])
            text = resp.get("content", "") if resp else ""
            steps.append({"model": expand_model, "response": text, "role": "expand"})
            suggested = text or suggested

    final_prompt = f"Use the latest extract/expand chain to answer the original question.\nOriginal question:\n{user_query}\n\nLatest summary:\n{summary}\nLatest expansion:\n{suggested}"
    final_resp = await query_model(chairman_model, [{"role": "user", "content": final_prompt}])
    final_text = final_resp.get("content", "") if final_resp else ""
    metadata = {"mode": "complex_iterative", "steps": steps}
    return steps, [], {"model": chairman_model, "response": final_text}, metadata


async def run_mode_complex_questioning(
    user_query: str,
    per_model_prompts: Optional[Dict[str, str]],
    council_models: Optional[List[str]],
    chairman_model: str,
):
    models = council_models or COUNCIL_MODELS
    answers = await stage1_collect_responses(user_query, per_model_prompts, models)
    answers = [{"model": a["model"], "response": a["response"], "role": "answer"} for a in answers]
    if not answers:
        return [], [], {"model": "error", "response": "Complex Questioning failed: no answers."}, {"mode": "complex_questioning"}

    questions: List[Dict[str, Any]] = []
    for ans in answers:
        peers = [a for a in answers if a["model"] != ans["model"]]
        question_prompt = (
            f"Re-read your answer through peers' lenses. Identify where you may be wrong or overconfident. Update briefly.\nYour answer:\n{ans['response']}\nPeers:\n" +
            "\n\n".join([f"{p['model']}:\n{p['response']}" for p in peers])
        )
        resp = await query_model(ans["model"], [{"role": "user", "content": question_prompt}])
        questions.append({
            "model": ans["model"],
            "response": resp.get("content", "") if resp else "",
            "role": "question_self",
        })

    brief_prompt = (
        f"Summarize convergences/divergences and produce a concise brief.\nQuestion:\n{user_query}\n\nAnswers:\n" +
        "\n\n".join([f"{a['model']}:\n{a['response']}" for a in answers]) +
        "\n\nReflections:\n" +
        "\n\n".join([f"{q['model']}:\n{q['response']}" for q in questions])
    )
    brief_resp = await query_model(chairman_model, [{"role": "user", "content": brief_prompt}])
    brief_text = brief_resp.get("content", "") if brief_resp else ""
    brief_step = {"model": chairman_model, "response": brief_text, "role": "brief"}

    muses: List[Dict[str, Any]] = []
    for ans in answers:
        muse_prompt = (
            f"Consider the brief alone (no original context). Add reflections or corrections; avoid inventing new facts.\nBrief:\n{brief_text}"
        )
        resp = await query_model(ans["model"], [{"role": "user", "content": muse_prompt}])
        muses.append({
            "model": ans["model"],
            "response": resp.get("content", "") if resp else "",
            "role": "muse",
        })

    final_prompt = (
        f"Produce final answer based on debate and muse round; cite from earlier context if needed.\nBrief:\n{brief_text}\n\nMuse:\n" +
        "\n\n".join([f"{m['model']}:\n{m['response']}" for m in muses])
    )
    final_resp = await query_model(chairman_model, [{"role": "user", "content": final_prompt}])
    final_text = final_resp.get("content", "") if final_resp else ""

    steps = answers + questions + muses + [brief_step]
    metadata = {"mode": "complex_questioning", "steps": steps}
    return steps, [], {"model": chairman_model, "response": final_text}, metadata


MODE_RUNNERS: Dict[str, Callable[..., Any]] = {
    "baseline": run_mode_baseline,
    "round_robin": run_mode_round_robin,
    "fight": run_mode_fight,
    "stacks": run_mode_stacks,
    "complex_iterative": run_mode_complex_iterative,
    "complex_questioning": run_mode_complex_questioning,
}
