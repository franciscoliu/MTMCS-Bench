# model/instructionblip_wrapper.py

from typing import List, Dict, Optional, Union

import torch
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

from .base_wrapper import BaseChatModel


class InstructBlipModel(BaseChatModel):
    """
    Wrapper around Salesforce/instructblip-vicuna-7b that matches your
    BaseChatModel interface:

      generate(messages, image=None, temperature=..., max_new_tokens=...)

    It supports:
      - messages: str
      - messages: list[str]
      - messages: list[{"role": ..., "content": ...}]
      - image: PIL.Image.Image or None
    """

    def __init__(
        self,
        model_name: str,
        api_keys: Optional[str] = None,   # unused, for interface parity
        api_key: Optional[str] = None,    # unused, for interface parity
        generation_config=None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = torch.float16,
        max_input_length: int = 256,      # <--- NEW: hard cap on text length
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = InstructBlipProcessor.from_pretrained(model_name)
        self._dummy_image = None  # for text-only / unimodal mode
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        ).to(self.device)
        self.model.eval()

        self.generation_config = generation_config if generation_config is not None else {}
        self.seed = 42
        self.inputs: List[Dict] = []

        # NEW: store the max text length we allow into the Q-Former
        # (must be <= the underlying max_position_embeddings, typically 512)
        self.max_input_length = max_input_length

        # For efficiency tracking compatibility (optional)
        self._last_usage: Dict[str, int] = {}

        self.hf_model = self.model
        self.hf_processor = self.processor
        self.model_family = "instructblip"

    # ----------------- helper: format messages -> plain prompt -----------------
    def _ensure_image(self, image: Optional[Union[Image.Image, str]]) -> Image.Image:
        """
        InstructBLIP requires pixel_values; for text-only calls we feed a
        small dummy image so the model always gets something.
        """
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
            return img

        # No image provided: use (or create) a black 224x224 dummy
        if self._dummy_image is None:
            self._dummy_image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        return self._dummy_image

    def _format_history_to_prompt(self, messages) -> str:
        """
        Turn your history format into a single text prompt.

        Keeps behavior consistent with your other wrappers:
          - str -> single user turn
          - list[str] -> each string is a user turn
          - list[dict] with role/content
        """

        if isinstance(messages, str):
            # single user message
            return messages

        if isinstance(messages, list):
            if len(messages) == 0:
                return ""

            # list[dict] with "role" / "content"
            if isinstance(messages[0], dict) and "content" in messages[0]:
                lines = []
                for m in messages:
                    role = m.get("role", "user")
                    text = m.get("content", "")
                    # keep simple "User: ..." / "Assistant: ..." style
                    if role == "user":
                        prefix = "User"
                    elif role == "assistant":
                        prefix = "Assistant"
                    else:
                        prefix = role.capitalize()
                    lines.append(f"{prefix}: {text}")
                return "\n".join(lines)

            # list[str]
            return "\n".join(str(m) for m in messages)

        raise TypeError(f"Unsupported messages type for InstructionBlipModel: {type(messages)}")

    # ----------------- core generate -----------------

    def generate(
        self,
        messages,
        image: Optional[Union[Image.Image, str]] = None,
        clear_old_history: bool = True,  # ignored; kept for interface
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> str:
        """
        Generate a response given messages and an optional image.
        If image is None, we feed a dummy image so pixel_values is always present.
        """
        prompt = self._format_history_to_prompt(messages)

        # Always ensure we have an image for InstructBLIP
        viz = self._ensure_image(image)

        inputs = self.processor(
            images=viz,
            text=prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,              # prevents Q-Former pos-emb length crash
            max_length=self.max_input_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Store what we actually fed in (for debugging)
        self.inputs = {"prompt": prompt}

        do_sample = temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-5)

        gen_kwargs.update(self.generation_config)
        gen_kwargs.update(kwargs)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        # Decode whole sequence (prompt may be partially echoed; we can refine later if needed)
        text = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()

        # Rough token usage for EfficiencyTracker
        try:
            tokenized = self.processor.tokenizer(
                prompt,
                return_attention_mask=False,
                return_tensors=None,
            )
            num_in = len(tokenized["input_ids"])
            num_out = len(output_ids[0]) - num_in
            if num_out < 0:
                num_out = 0
            self._last_usage = {
                "input_tokens": num_in,
                "output_tokens": num_out,
            }
        except Exception:
            self._last_usage = {}

        return text



# Alias, matching your other wrappers
InstructionBlipWrapper = InstructBlipModel
