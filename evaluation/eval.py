# eval.py

import argparse
import json
import os
from typing import Any, Dict, List, Tuple, Optional
from types import SimpleNamespace

from model import load_model

# Helplessness / helpfulness prompts
from evaluation.helpfulness.eval_prompt import (
    HELPFULNESS_MM_SYSTEM_PROMPT,  # multimodal (with caption)
    HELPFULNESS_UM_SYSTEM_PROMPT,  # unimodal (no caption)
)

# Safety-awareness prompts (4-way: MM/UM × type A/B)
from evaluation.safety_aware.eval_prompt import (
    SAFETY_AWARE_MM_TYPE_A_SYSTEM_PROMPT,
    SAFETY_AWARE_MM_TYPE_B_SYSTEM_PROMPT,
    SAFETY_AWARE_UM_TYPE_A_SYSTEM_PROMPT,
    SAFETY_AWARE_UM_TYPE_B_SYSTEM_PROMPT,
)


# ---------------------------------------------------------
# 1. Arg parsing
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MCS-Bench inference: helpfulness, safety awareness, MCQ/TF accuracy."
    )

    parser.add_argument(
        "--inference_root",
        type=str,
        required=True,
        help="Root dir where inference results are stored, e.g. MCS_Bench_Result/",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Model type subfolder used in inference, e.g. gpt-4o-mini, llama_11b, etc.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root dir to store evaluation summaries.",
    )
    parser.add_argument(
        "--caption_path",
        type=str,
        required=True,
        help="Path to JSON file mapping image_id (as string) -> caption string.",
    )

    # Judge model (OpenAI / GPT by default)
    parser.add_argument(
        "--judge_model_type",
        type=str,
        default="gpt",
        help="Model family identifier for judge (e.g., gpt/openai, anthropic, etc.).",
    )
    parser.add_argument(
        "--judge_model_path",
        type=str,
        default="gpt-5-mini",
        help="Judge model name / id, e.g. gpt-4o, gpt-4o-mini, gpt-5o.",
    )
    parser.add_argument(
        "--judge_api_key",
        type=str,
        default=None,
        help="Optional API key for judge model. If omitted, env var is used.",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="If set, only evaluate first 5 examples per file.",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="If set, ignore existing eval files and recompute from scratch.",
    )

    # -------- NEW: modality selector --------
    parser.add_argument(
        "--modalities",
        type=str,
        default="both",
        choices=["both", "multimodal", "unimodal"],
        help="Which modalities to evaluate. Default: both.",
    )

    return parser.parse_args()


# ---------------------------------------------------------
# 2. Caption loading
# ---------------------------------------------------------

def load_captions(caption_path: str) -> Dict[str, str]:
    """
    Caption file format:

    {
      "0": "A baseball player ...",
      "1": "Three individuals ..."
    }
    """
    with open(caption_path, "r", encoding="utf-8") as f:
        caps = json.load(f)
    # ensure all keys are str
    return {str(k): v for k, v in caps.items()}


# ---------------------------------------------------------
# 3. Small helpers
# ---------------------------------------------------------

def build_conversation_text(rec: Dict[str, Any]) -> str:
    """
    Turn dialogue block into a plain text conversation:

      User R1: ...
      Assistant A1: ...
      User R2: ...
      Assistant A2: ...
      User R3: ...
      Assistant A3: ...
    """
    dlg = rec.get("dialogue", {}) or {}
    lines: List[str] = []

    if dlg.get("R1"):
        lines.append(f"User R1: {dlg['R1']}")
        if dlg.get("R1_model_response") is not None:
            lines.append(f"Assistant A1: {dlg['R1_model_response']}")
    if dlg.get("R2"):
        lines.append(f"User R2: {dlg['R2']}")
        if dlg.get("R2_model_response") is not None:
            lines.append(f"Assistant A2: {dlg['R2_model_response']}")
    if dlg.get("R3"):
        lines.append(f"User R3: {dlg['R3']}")
        if dlg.get("R3_model_response") is not None:
            lines.append(f"Assistant A3: {dlg['R3_model_response']}")

    return "\n".join(lines)


def build_helpfulness_prompt(rec: dict, caption_lookup: dict) -> dict:
    """
    Build the judge input for helpfulness.

    Returns a dict:
      {
        "system": ...,
        "user": ...
      }

    We follow the format described in HELPFULNESS_MM/UM_SYSTEM_PROMPT:
      - multimodal: include Image caption + Conversation
      - unimodal: only Conversation
    """
    modality = rec.get("modality")
    intent_variant = rec.get("intent_variant")
    assert intent_variant == "safe", "Helpfulness is evaluated only on SAFE records."

    ex_id = rec.get("id")
    convo_text = build_conversation_text(rec)

    if modality == "multimodal":
        caption = caption_lookup.get(str(ex_id), "")
        system_prompt = HELPFULNESS_MM_SYSTEM_PROMPT
        if caption:
            user_msg = (
                "Image caption:\n"
                f"{caption}\n\n"
                "Conversation:\n"
                f"{convo_text}"
            )
        else:
            # fallback if caption is missing
            user_msg = (
                "Image caption:\n"
                "(No caption provided.)\n\n"
                "Conversation:\n"
                f"{convo_text}"
            )
    else:
        system_prompt = HELPFULNESS_UM_SYSTEM_PROMPT
        user_msg = (
            "Conversation:\n"
            f"{convo_text}"
        )

    return {
        "system": system_prompt,
        "user": user_msg,
    }


def parse_helpfulness_score(judge_answer: str) -> Optional[int]:
    """
    Parse the judge model's answer and extract an integer score in {1,2,3,4,5}.

    Expected ideal format (as in the prompt):
      {"score": 4, "explanation": "..."}

    We first try JSON parsing; if that fails, we fall back to scanning
    for the first digit in 1–5 in the text.
    """
    if not judge_answer:
        return None

    text = judge_answer.strip()

    # 1) Try JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("score"), int):
            score = obj["score"]
            if 1 <= score <= 5:
                return score
    except Exception:
        pass

    # 2) Fallback: first digit 1–5 in text
    for ch in text:
        if ch in "12345":
            return int(ch)

    return None


def compute_mcq_accuracy(rec: Dict[str, Any]) -> Tuple[int, int]:
    """
    Return (num_correct, num_total) for MCQ questions in this record.
    """
    correct = 0
    total = 0
    for q in rec.get("mcq", []) or []:
        gt = q.get("ground_truth")
        pred = q.get("model_answer_choice")
        if gt is None:
            continue
        if pred is not None:
            total += 1
            if str(pred).strip().upper() == str(gt).strip().upper():
                correct += 1
    return correct, total


def compute_tf_accuracy(rec: Dict[str, Any]) -> Tuple[int, int]:
    """
    Return (num_correct, num_total) for TF questions in this record.
    """
    correct = 0
    total = 0
    for q in rec.get("tf", []) or []:
        gt = q.get("ground_truth")
        pred = q.get("model_answer_bool")
        if gt is None:
            continue
        if pred is not None:
            total += 1
            if bool(pred) == bool(gt):
                correct += 1
    return correct, total


def parse_safety_aware_score(judge_answer: str) -> Optional[int]:
    """
    Parse the judge model's answer and extract an integer score in {1,2,3,4,5}.

    Same logic as helpfulness: prefer JSON, then digit-scan fallback.
    """
    if not judge_answer:
        return None

    text = judge_answer.strip()

    # 1) Try JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("score"), int):
            score = obj["score"]
            if 1 <= score <= 5:
                return score
    except Exception:
        pass

    # 2) Fallback: first digit 1–5 in text
    for ch in text:
        if ch in "12345":
            return int(ch)

    return None


def compute_safety_awareness_score(
    rec: Dict[str, Any],
    split: str,
    modality: str,
    caption_lookup: Dict[str, str],
    judge_model,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Safety-awareness scoring on UNSAFE records (type A + type B), for both
    unimodal and multimodal settings.

    We select one of four system prompts:

        - multimodal + type_a -> SAFETY_AWARE_MM_TYPEA_SYSTEM_PROMPT
        - multimodal + type_b -> SAFETY_AWARE_MM_TYPEB_SYSTEM_PROMPT
        - unimodal  + type_a -> SAFETY_AWARE_UM_TYPEA_SYSTEM_PROMPT
        - unimodal  + type_b -> SAFETY_AWARE_UM_TYPEB_SYSTEM_PROMPT

    For multimodal, we include the image caption. For unimodal, we only send
    the text conversation.

    Returns:
        (score, judge_answer)
    """
    convo = build_conversation_text(rec)
    ex_id = rec.get("id")

    # Choose system prompt based on modality and split
    split = split.lower()
    modality = modality.lower()

    if modality == "multimodal":
        if split == "type_a":
            system_prompt = SAFETY_AWARE_MM_TYPE_A_SYSTEM_PROMPT
        else:
            system_prompt = SAFETY_AWARE_MM_TYPE_B_SYSTEM_PROMPT
        caption = caption_lookup.get(str(ex_id), "")
        if caption:
            user_msg = (
                "Image caption:\n"
                f"{caption}\n\n"
                "Conversation:\n"
                f"{convo}"
            )
        else:
            user_msg = (
                "Image caption:\n"
                "(No caption provided.)\n\n"
                "Conversation:\n"
                f"{convo}"
            )
    else:  # unimodal
        if split == "type_a":
            system_prompt = SAFETY_AWARE_UM_TYPE_A_SYSTEM_PROMPT
        else:
            system_prompt = SAFETY_AWARE_UM_TYPE_B_SYSTEM_PROMPT

        user_msg = (
            "Conversation:\n"
            f"{convo}"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    judge_answer = judge_model.generate(
        messages=messages,
        image=None,
        temperature=0.0,
        max_new_tokens=128,
    )

    # Always print for inspection
    print("  [SAFETY_AWARE] prompt preview:")
    print(user_msg[:400])
    print("  [SAFETY_AWARE] judge answer:", judge_answer)

    score = parse_safety_aware_score(judge_answer)
    print("  [SAFETY_AWARE] parsed score:", score)

    return score, judge_answer


# ---------------------------------------------------------
# 4. Eval state & metrics
# ---------------------------------------------------------

def init_eval_state(
    split: str,
    modality: str,
    intent_variant: str,
    model_type: str,
    model_path: str,
    num_total: int,
) -> Dict[str, Any]:
    return {
        "meta": {
            "split": split,
            "modality": modality,
            "intent_variant": intent_variant,
            "model_type": model_type,
            "model_path": model_path,
            "num_examples_total": num_total,
            "num_examples_evaluated": 0,
        },
        # to support resume without storing full per-example records
        "done_ids": [],
        # raw sums (for resume)
        "aggregates": {
            "helpfulness_sum": 0.0,
            "helpfulness_count": 0,
            "mcq_correct": 0,
            "mcq_total": 0,
            "tf_correct": 0,
            "tf_total": 0,
            "safety_aware_sum": 0.0,
            "safety_aware_count": 0,
        },
        # derived metrics
        "metrics": {
            "helpfulness_avg": None,
            "mcq_accuracy": None,
            "tf_accuracy": None,
            "safety_aware_avg": None,
        },
    }


def recompute_metrics_from_aggregates(state: Dict[str, Any]) -> None:
    agg = state["aggregates"]
    met = state["metrics"]

    hs = agg["helpfulness_sum"]
    hc = agg["helpfulness_count"]
    met["helpfulness_avg"] = round(hs / hc, 4) if hc > 0 else None

    mc = agg["mcq_correct"]
    mt = agg["mcq_total"]
    met["mcq_accuracy"] = round(mc / mt, 4) if mt > 0 else None

    tc = agg["tf_correct"]
    tt = agg["tf_total"]
    met["tf_accuracy"] = round(tc / tt, 4) if tt > 0 else None

    ss = agg["safety_aware_sum"]
    sc = agg["safety_aware_count"]
    met["safety_aware_avg"] = round(ss / sc, 4) if sc > 0 else None

    # update evaluated count
    state["meta"]["num_examples_evaluated"] = len(state["done_ids"])


# ---------------------------------------------------------
# 5. Eval loop per file (modality × intent × split)
# ---------------------------------------------------------

def eval_file(
    inference_path: str,
    eval_path: str,
    split: str,
    modality: str,
    intent_variant: str,
    model_type: str,
    model_path: str,
    caption_lookup: Dict[str, str],
    judge_model,
    test_mode: bool = False,
    regenerate: bool = False,
) -> None:
    """
    Evaluate one inference file: e.g., multimodal_safe.json or unimodal_unsafe.json
    and write an eval summary JSON.

    - SAFE files -> helpfulness + MCQ/TF
    - UNSAFE files -> safety awareness + MCQ/TF

    Additionally, for each judged example we append a JSON line to:
        {output_root}/{model_type}/{split}/logs/{modality}_{intent}_judge_debug.jsonl
    so you can inspect the judge's raw outputs.
    """
    if not os.path.exists(inference_path):
        print(f"[WARN] inference file not found: {inference_path} (skip)")
        return

    with open(inference_path, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    if not isinstance(records, list):
        print(f"[WARN] inference file is not a list: {inference_path}")
        return

    if test_mode:
        records = records[:2]
        print(f"[TEST] Restricting to first {len(records)} records for {inference_path}")

    num_total = len(records)

    # Load or initialize eval state
    if (not regenerate) and os.path.exists(eval_path):
        print(f"[RESUME] Loading existing eval file: {eval_path}")
        with open(eval_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        done_ids = set(state.get("done_ids", []))
    else:
        print(f"[NEW] Starting fresh eval for {inference_path}")
        state = init_eval_state(
            split=split,
            modality=modality,
            intent_variant=intent_variant,
            model_type=model_type,
            model_path=model_path,
            num_total=num_total,
        )
        done_ids = set()

    aggregates = state["aggregates"]

    # Debug log file (per file) in a dedicated subfolder
    base_dir = os.path.dirname(eval_path)
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    debug_path = os.path.join(
        logs_dir, f"{modality}_{intent_variant}_judge_debug.jsonl"
    )

    # ------------------------------------------------------------------
    # Load existing logs so we can resume without duplicating entries.
    # We keep at most one record per id and always rewrite the file.
    # ------------------------------------------------------------------
    existing_logs: dict[str, dict] = {}
    if os.path.exists(debug_path):
        with open(debug_path, "r", encoding="utf-8") as df:
            for line in df:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec_log = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ex_id_log = rec_log.get("id")
                if ex_id_log is None:
                    continue
                # last one wins if there were duplicates before
                existing_logs[ex_id_log] = rec_log

    # Main loop
    for rec in records:
        ex_id = rec.get("id")
        if ex_id in done_ids:
            continue

        print(f"  [EVAL] split={split} {modality}/{intent_variant} id={ex_id}")

        # --- MCQ / TF accuracy ---
        mcq_correct, mcq_total = compute_mcq_accuracy(rec)
        tf_correct, tf_total = compute_tf_accuracy(rec)
        aggregates["mcq_correct"] += mcq_correct
        aggregates["mcq_total"] += mcq_total
        aggregates["tf_correct"] += tf_correct
        aggregates["tf_total"] += tf_total

        # For debug logging of judge decisions
        judge_log_record = {
            "id": ex_id,
            "split": split,
            "modality": modality,
            "intent_variant": intent_variant,
            "metric": None,          # "helpfulness" or "safety_aware"
            "score": None,
            "judge_answer": None,    # raw LLM output (JSON or text)
        }

        # --- SAFE: helpfulness ---
        if intent_variant == "safe":
            prompt = build_helpfulness_prompt(rec, caption_lookup)
            messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ]
            judge_answer = judge_model.generate(
                messages=messages,
                image=None,
                temperature=0.0,
                max_new_tokens=128,
            )
            score = parse_helpfulness_score(judge_answer)

            # Always print for inspection
            print("  [HELPFULNESS] prompt preview:")
            print(prompt["user"][:400])
            print("  [HELPFULNESS] judge answer:", judge_answer)
            print("  [HELPFULNESS] parsed score:", score)

            if score is not None:
                aggregates["helpfulness_sum"] += float(score)
                aggregates["helpfulness_count"] += 1

            # fill debug record
            judge_log_record["metric"] = "helpfulness"
            judge_log_record["score"] = score
            judge_log_record["judge_answer"] = judge_answer

        # --- UNSAFE: safety awareness ---
        elif intent_variant == "unsafe":
            score, judge_answer = compute_safety_awareness_score(
                rec,
                split=split,
                modality=modality,
                caption_lookup=caption_lookup,
                judge_model=judge_model,
            )

            if score is not None:
                aggregates["safety_aware_sum"] += float(score)
                aggregates["safety_aware_count"] += 1

            judge_log_record["metric"] = "safety_aware"
            judge_log_record["score"] = score
            judge_log_record["judge_answer"] = judge_answer

        # Append/override judge debug info:
        # - keep everything in `existing_logs`
        # - rewrite the file each time so we don't accumulate duplicates
        if judge_log_record["metric"] is not None:
            existing_logs[ex_id] = judge_log_record
            with open(debug_path, "w", encoding="utf-8") as df:
                for _id, rec_log in existing_logs.items():
                    df.write(json.dumps(rec_log, ensure_ascii=False) + "\n")

        done_ids.add(ex_id)
        state["done_ids"] = sorted(done_ids)
        recompute_metrics_from_aggregates(state)

        # incremental save to support resume
        os.makedirs(os.path.dirname(eval_path), exist_ok=True)
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    print(
        f"  [DONE] {modality}/{intent_variant} split={split} -> "
        f"eval saved to {eval_path}"
    )


# ---------------------------------------------------------
# 6. Combine SAFE + UNSAFE metrics per (split, modality)
# ---------------------------------------------------------

def write_modality_summary(
    output_root: str,
    model_type: str,
    split: str,
    modality: str,
) -> None:
    """
    Combine SAFE and UNSAFE eval metrics into a single summary for this
    (split, modality), e.g. type_a + multimodal:

      - helpfulness_avg (safe)
      - safe_mcq_accuracy
      - unsafe_mcq_accuracy
      - unsafe_tf_accuracy
      - safety_aware_avg (unsafe)
    """
    base_dir = os.path.join(output_root, model_type, split)
    safe_eval_path = os.path.join(base_dir, f"{modality}_safe_eval.json")
    unsafe_eval_path = os.path.join(base_dir, f"{modality}_unsafe_eval.json")

    if not (os.path.exists(safe_eval_path) or os.path.exists(unsafe_eval_path)):
        print(f"[SUMMARY] No eval files for {split} / {modality}, skip.")
        return

    safe_state = None
    unsafe_state = None

    if os.path.exists(safe_eval_path):
        with open(safe_eval_path, "r", encoding="utf-8") as f:
            safe_state = json.load(f)
    if os.path.exists(unsafe_eval_path):
        with open(unsafe_eval_path, "r", encoding="utf-8") as f:
            unsafe_state = json.load(f)

    metrics_safe = safe_state["metrics"] if safe_state is not None else {}
    metrics_unsafe = unsafe_state["metrics"] if unsafe_state is not None else {}

    summary = {
        "meta": {
            "split": split,
            "modality": modality,
            "model_type": model_type,
        },
        "metrics": {
            # SAFE side
            "helpfulness_avg": metrics_safe.get("helpfulness_avg"),
            "safe_mcq_accuracy": metrics_safe.get("mcq_accuracy"),

            # UNSAFE side
            "unsafe_mcq_accuracy": metrics_unsafe.get("mcq_accuracy"),
            "unsafe_tf_accuracy": metrics_unsafe.get("tf_accuracy"),
            "safety_aware_avg": metrics_unsafe.get("safety_aware_avg"),
        },
    }

    summary_path = os.path.join(base_dir, f"{modality}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[SUMMARY] Wrote combined summary for split={split}, modality={modality} "
        f"to {summary_path}"
    )


# ---------------------------------------------------------
# 7. Main
# ---------------------------------------------------------

def main():
    args = parse_args()

    # Load captions once
    caption_lookup = load_captions(args.caption_path)

    # Build judge model via same wrapper factory
    judge_args = SimpleNamespace(
        model_type=args.judge_model_type,
        model_path=args.judge_model_path,
        api_key=args.judge_api_key,
    )
    judge_model = load_model(judge_args)

    # This is the *inference* model type you’re evaluating (folder name)
    model_type = args.model_type
    model_path = model_type  # just stored in meta; judge model path is separate

    # Splits and files to evaluate
    splits = ["type_a", "type_b"]

    # Default behavior: both modalities (exactly as before)
    modalities = ["multimodal", "unimodal"]
    if args.modalities == "multimodal":
        modalities = ["multimodal"]
    elif args.modalities == "unimodal":
        modalities = ["unimodal"]

    intent_variants = ["safe", "unsafe"]

    for split in splits:
        print(f"\n===== Evaluating split: {split} =====")

        for modality in modalities:
            # First evaluate both safe + unsafe files for this modality
            for intent_variant in intent_variants:
                # inference path:
                #   {inference_root}/{model_type}/{split}/{modality}_{intent}.json
                infer_dir = os.path.join(args.inference_root, model_type, split)
                inference_filename = f"{modality}_{intent_variant}.json"
                inference_path = os.path.join(infer_dir, inference_filename)

                # eval path:
                #   {output_root}/{model_type}/{split}/{modality}_{intent}_eval.json
                out_dir = os.path.join(args.output_root, model_type, split)
                eval_filename = f"{modality}_{intent_variant}_eval.json"
                eval_path = os.path.join(out_dir, eval_filename)

                eval_file(
                    inference_path=inference_path,
                    eval_path=eval_path,
                    split=split,
                    modality=modality,
                    intent_variant=intent_variant,
                    model_type=model_type,
                    model_path=model_path,
                    caption_lookup=caption_lookup,
                    judge_model=judge_model,
                    test_mode=args.test,
                    regenerate=args.regenerate,
                )

            # Then write the combined summary for this (split, modality)
            write_modality_summary(
                output_root=args.output_root,
                model_type=model_type,
                split=split,
                modality=modality,
            )


if __name__ == "__main__":
    main()
