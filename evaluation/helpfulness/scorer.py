# evaluation/helpfulness/scorer.py

import json
from typing import Any, Dict, List, Optional

from evaluation.utils import (
    build_dialogue_text,
    load_cache,
    save_cache,
)

from evaluation.helpfulness.eval_prompt import (
    HELPFULNESS_MM_SYSTEM_PROMPT,
    HELPFULNESS_UM_SYSTEM_PROMPT,
)


def _parse_helpfulness_score(raw: str) -> Optional[int]:
    """
    Expect the judge to return something like:
        {"helpfulness_score": 2, "explanation": "..."}
    Be defensive and try to salvage the score even if format is a bit off.
    """
    if not raw:
        return None

    # Try JSON first
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "helpfulness_score" in obj:
            score = obj["helpfulness_score"]
            if isinstance(score, (int, float)):
                s = int(score)
                if s in (0, 1, 2):
                    return s
    except Exception:
        pass

    # Fallback: look for an integer 0/1/2 in the string
    for ch in raw:
        if ch in "012":
            return int(ch)

    return None


def _build_helpfulness_prompt(
    is_multimodal: bool,
    dialogue_text: str,
    image_caption: Optional[str] = None,
) -> str:
    if is_multimodal:
        cap = image_caption or "No caption is available for this image."
        return HELPFULNESS_MM_SYSTEM_PROMPT.format(
            image_caption=cap,
            dialogue=dialogue_text,
        )
    else:
        return HELPFULNESS_UM_SYSTEM_PROMPT.format(dialogue=dialogue_text)


def evaluate_helpfulness(
    records: List[Dict[str, Any]],
    judge_model,
    is_multimodal: bool,
    caption_map: Optional[Dict[str, str]] = None,
    cache_path: Optional[str] = None,
    regenerate: bool = False,
    test: bool = False,
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate helpfulness (0/1/2) on the SAFE files.

    Args:
        records: list of combined-view records (one per id for a modality+split).
        judge_model: loaded via model.load_model, must expose .generate(...)
        is_multimodal: True for multimodal, False for unimodal
        caption_map: optional mapping id -> caption (string)
        cache_path: where to store per-id scores (for resume). If None, no cache.
        regenerate: if True, ignore previous cache.
        test: if True, print per-example scores and DON'T save anything.
        max_examples: if not None, limit number of evaluated records.
    """
    if test:
        cache = {"scores": {}}
    else:
        cache = {"scores": {}}
        if not regenerate and cache_path is not None:
            cache = load_cache(cache_path)

    scores: Dict[str, int] = cache.get("scores", {})
    updated = False

    processed = 0
    for rec in records:
        if max_examples is not None and processed >= max_examples:
            break

        rec_id = rec.get("id")
        if rec_id is None:
            continue
        key = str(rec_id)

        if (not regenerate) and (key in scores):
            # Already have a score for this id
            processed += 1
            continue

        dialogue = rec.get("dialogue", {}) or {}
        dialogue_text = build_dialogue_text(dialogue)

        caption = None
        if is_multimodal and caption_map is not None:
            caption = caption_map.get(key)
            if caption is None and isinstance(rec_id, int):
                caption = caption_map.get(str(rec_id))

        user_prompt = _build_helpfulness_prompt(
            is_multimodal=is_multimodal,
            dialogue_text=dialogue_text,
            image_caption=caption,
        )

        judge_output = judge_model.generate(
            messages=[{"role": "user", "content": user_prompt}],
            image=None,
            temperature=0.0,
            max_new_tokens=256,
        )

        score = _parse_helpfulness_score(judge_output)
        if score is None:
            score = 0  # very conservative: treat as bad

        scores[key] = int(score)
        updated = True
        processed += 1

        if test:
            print(
                f"[HELP] id={rec_id} | multimodal={is_multimodal} "
                f"| score={score} | raw={judge_output!r}"
            )

        # Flush cache frequently on long runs
        if (not test) and cache_path is not None and updated:
            cache["scores"] = scores
            save_cache(cache, cache_path)
            updated = False

    # Aggregate metrics
    if not scores:
        return {
            "num_scored": 0,
            "avg_score": None,
            "dist": {},
        }

    total = len(scores)
    sum_scores = sum(scores.values())
    avg = sum_scores / total

    counts = {0: 0, 1: 0, 2: 0}
    for v in scores.values():
        if v in counts:
            counts[v] += 1

    dist = {str(k): counts[k] / total for k in counts if total > 0}

    return {
        "num_scored": total,
        "avg_score": avg,
        "dist": dist,
    }
