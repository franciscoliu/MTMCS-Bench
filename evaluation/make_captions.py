# evaluation/make_captions.py

import argparse
import json
import os
from typing import Dict, Optional

from datasets import load_dataset
from PIL import Image

from model import load_model


CAPTION_SYSTEM_PROMPT = (
    "You are an assistant that writes neutral, factual image captions.\n"
    "Given an image, describe what is visibly present in 1–2 sentences.\n"
    "Do NOT speculate about hidden intent, identities, or future actions.\n"
    "Do NOT mention any safety or policy judgments; just describe what you see."
)

CAPTION_USER_PROMPT = (
    "Please provide a concise, objective caption for this image. "
    "1–2 sentences, factual only."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate image captions for MCS-Bench and save id->caption mapping."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt",
        help=(
            "Caption model family (e.g., gpt, openai, claude). "
            "Default: gpt (OpenAI wrapper)."
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="gpt-4o-mini",
        help=(
            "Caption model name / id (OpenAI API model), e.g. gpt-4o, gpt-4o-mini. "
            "Default: gpt-4o-mini."
        ),
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help=(
            "Optional API key for the caption model. If omitted, "
            "wrappers use environment variables (e.g., OPENAI_API_KEY)."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["both", "type_a", "type_b"],
        help="Which split(s) to caption (type_a, type_b, or both).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="DATASET_NAME",
        help="HF dataset name." ,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save captions JSON,",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="If set, ignore existing caption file and recompute everything.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Debug mode: caption only first 5 examples per split, print, and do not save.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=96,
        help="Max new tokens for each caption generation.",
    )
    return parser.parse_args()


# A tiny args wrapper so we can reuse model.load_model
class CaptionArgs:
    def __init__(self, model_type: str, model_path: str, api_key: Optional[str]):
        self.model_type = model_type
        self.model_path = model_path
        self.api_key = api_key
        self.organization = None  # for OpenAI if you ever need it


def load_existing_captions(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Existing caption file at {path} is not a dict.")
    # normalize keys to str
    return {str(k): str(v) for k, v in data.items()}


def save_captions(captions: Dict[str, str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)


def caption_image(chat_model, image: Image.Image, args) -> str:
    """
    Call your VLM wrapper to caption a single image.
    """
    messages = [
        {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
        {"role": "user", "content": CAPTION_USER_PROMPT},
    ]
    resp = chat_model.generate(
        messages=messages,
        image=image,
        temperature=0.0,
        max_new_tokens=args.max_new_tokens,
    )
    return resp.strip()


def main():
    args = parse_args()

    # Load caption model
    cargs = CaptionArgs(
        model_type=args.model_type,
        model_path=args.model_path,
        api_key=args.api_key,
    )
    chat_model = load_model(cargs)

    # Load or initialize caption map
    if args.test:
        captions: Dict[str, str] = {}
    else:
        if args.regenerate:
            captions = {}
        else:
            if os.path.exists(args.output_path):
                print(f"Resuming from existing caption file: {args.output_path}")
                captions = load_existing_captions(args.output_path)
            else:
                captions = {}

    # Decide splits
    if args.split == "both":
        splits = ["type_a", "type_b"]
    else:
        splits = [args.split]

    for split in splits:
        print(f"\n=== Generating captions for split: {split} ===")
        ds = load_dataset(args.dataset_name, split=split)
        total = len(ds)
        limit = 5 if args.test else total

        for i in range(limit):
            ex = ds[i]
            ex_id = ex.get("id")
            if ex_id is None:
                continue
            key = str(ex_id)

            if (not args.regenerate) and (key in captions):
                # Already have caption for this image id
                continue

            image = ex.get("image", None)
            if not isinstance(image, Image.Image):
                # dataset should have PIL Images; if not, skip
                print(f"  [WARN] Example id={key} has no PIL image; skipping.")
                continue

            print(f"  [{split}] Example {i+1}/{limit} (id={key}) -> captioning...")
            cap = caption_image(chat_model, image, args)

            captions[key] = cap

            if args.test:
                print(f"    Caption: {cap!r}")

            # Save incrementally on non-test runs
            if (not args.test) and (i % 20 == 0):
                save_captions(captions, args.output_path)
                print(f"    [checkpoint] Saved captions up to idx={i}")

    if args.test:
        print("\n[TEST] Finished captioning first few samples (not saved).")
    else:
        save_captions(captions, args.output_path)
        print(f"\nAll captions saved to: {args.output_path}")
        print(f"Total unique ids with captions: {len(captions)}")


if __name__ == "__main__":
    main()
