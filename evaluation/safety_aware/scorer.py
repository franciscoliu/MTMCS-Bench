# evaluation/safety_aware/scorer.py

from typing import Any, Dict, List


def evaluate_safety_aware(
    records: List[Dict[str, Any]],
    judge_model,
    is_multimodal: bool,
    cache_path: str = None,
    regenerate: bool = False,
    test: bool = False,
    max_examples: int = None,
) -> Dict[str, Any]:
    """
    Placeholder for safety-awareness evaluation on UNSAFE files.

    We haven't finalized the scoring prompt yet, so for now this returns
    an empty metric structure, and does NOT call the judge model.
    Later, we will mirror the structure of helpfulness.scorer.evaluate_helpfulness.
    """
    # NOTE: intentionally not using judge_model yet.
    return {
        "num_scored": 0,
        "avg_score": None,
        "dist": {},
    }
