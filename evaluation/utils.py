# evaluation/common.py

import json
import os
from typing import Any, Dict, List, Tuple, Optional


def load_json_list(path: str) -> List[Dict[str, Any]]:
    """Load a JSON file that contains a top-level list."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected list at top-level in {path}, got {type(data)}")


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_cache(path: str) -> Dict[str, Any]:
    """Load cache mapping from id -> score (or similar)."""
    if not os.path.exists(path):
        return {"scores": {}}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "scores" not in data or not isinstance(data["scores"], dict):
        data["scores"] = {}
    return data


def save_cache(cache: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ---------- MCQ / TF metrics ----------

def compute_mcq_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute MCQ accuracy from a list of records in the combined-view format:
        rec["mcq"] = [ { "ground_truth": "A", "model_answer_choice": "B", ... }, ... ]
    """
    num_questions = 0
    num_correct = 0
    for rec in records:
        for q in rec.get("mcq", []) or []:
            num_questions += 1
            gt = q.get("ground_truth")
            pred = q.get("model_answer_choice")
            if gt is not None and pred is not None and str(gt).strip().upper() == str(pred).strip().upper():
                num_correct += 1

    acc = None
    if num_questions > 0:
        acc = num_correct / num_questions

    return {
        "num_questions": num_questions,
        "num_correct": num_correct,
        "accuracy": acc,
    }


def compute_tf_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute TF accuracy from records where each rec has:
        rec["tf"] = [ { "ground_truth": bool, "model_answer_bool": bool, ... }, ... ]
    """
    num_questions = 0
    num_correct = 0
    for rec in records:
        for q in rec.get("tf", []) or []:
            num_questions += 1
            gt = q.get("ground_truth")
            pred = q.get("model_answer_bool")
            if isinstance(gt, bool) and isinstance(pred, bool) and gt == pred:
                num_correct += 1

    acc = None
    if num_questions > 0:
        acc = num_correct / num_questions

    return {
        "num_questions": num_questions,
        "num_correct": num_correct,
        "accuracy": acc,
    }


def build_dialogue_text(dialogue: Dict[str, Any]) -> str:
    """
    Turn the dialogue block into a readable multi-turn transcript:

        User: R1
        Assistant: R1_model_response
        User: R2
        Assistant: R2_model_response
        ...

    Missing turns are skipped.
    """
    lines: List[str] = []
    for turn in ["R1", "R2", "R3"]:
        user = dialogue.get(turn)
        asst = dialogue.get(f"{turn}_model_response")
        if user:
            lines.append(f"User: {user}")
        if asst:
            lines.append(f"Assistant: {asst}")
    return "\n".join(lines)
