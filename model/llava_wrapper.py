import copy
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from .base_wrapper import BaseChatModel


class LlavaModel(BaseChatModel):
    """
    Wrapper for LLaVA / LLaVA-NeXT models (HF):

      - llava-hf/llava-v1.6-mistral-7b-hf
      - llava-hf/llava-next-72b-hf
      - and other LlavaNext* checkpoints

    Interface is compatible with your inference.py:

        generate(messages, image=None, temperature=..., max_new_tokens=...)

    `messages` formats supported:
      - str
      - list[str]
      - list[{"role": ..., "content": ...}] (OpenAI-style)
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
        **hf_kwargs: Any,
    ):
        """
        Args:
            model_name: HF model id, e.g. "llava-hf/llava-v1.6-mistral-7b-hf".
            device: e.g. "cuda", "cuda:0", "cpu". If None, we use model.device.
            torch_dtype: dtype for weights (default float16).
            hf_kwargs: extra kwargs passed to from_pretrained (e.g. device_map="auto").
        """
        self.model_name = model_name

        # Processor handles both text & image and knows the chat template.
        self.processor = LlavaNextProcessor.from_pretrained(model_name)

        # For big models (72B), you probably want device_map="auto".
        # You can pass that via hf_kwargs from load_model if needed.
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            **hf_kwargs,
        )

        # Move to device if explicitly specified; otherwise rely on model.device
        if device is not None:
            self.device = torch.device(device)
            self.model.to(self.device)
        else:
            self.device = self.model.device

        # Optional compatibility fields used elsewhere in your code
        self.seed = 42
        self.inputs: List[Dict[str, Any]] = []  # last processed conversation
        self._last_usage: Optional[Dict[str, Any]] = None  # (we'll keep as None)

        self.hf_model = self.model
        self.hf_processor = self.processor
        self.model_family = "llava"

    # -------------------------------------------------------
    # Core generate
    # -------------------------------------------------------
    def generate(
        self,
        messages,
        image: Optional[Union[Image.Image, str]] = None,
        clear_old_history: bool = True,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        **gen_kwargs: Any,
    ) -> str:
        """
        Main entry point called by inference.py.

        `image`:
          - None  -> text-only chat.
          - PIL.Image.Image (or path string) -> attached to the last user message.
        """

        # -------- 1) Normalize messages into list[dict] --------
        if isinstance(messages, str):
            msg_list = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            if not messages:
                msg_list = []
            elif isinstance(messages[0], dict) and "content" in messages[0]:
                # Already [{role, content}, ...]
                msg_list = copy.deepcopy(messages)
            else:
                # Assume list[str] -> each is user turn
                msg_list = [{"role": "user", "content": m} for m in messages]
        else:
            raise TypeError(f"LlavaModel.generate: unsupported messages type {type(messages)}")

        # -------- 2) Build LLaVA-style conversation --------
        # LLaVA expects: [{ "role": "user"/"assistant",
        #                   "content": [ {"type": "text", "text": ...}, {"type": "image"}, ... ] }]
        conversation: List[Dict[str, Any]] = []
        for m in msg_list:
            role = m.get("role", "user")
            if role not in ("user", "assistant"):
                role = "user"

            text = str(m.get("content", ""))
            blocks = []
            if text:
                blocks.append({"type": "text", "text": text})

            conversation.append({"role": role, "content": blocks})

        # Attach image to last user message (if provided)
        if image is not None:
            last_user_idx = None
            for i in range(len(conversation) - 1, -1, -1):
                if conversation[i]["role"] == "user":
                    last_user_idx = i
                    break

            if last_user_idx is None:
                # No user message yet: create one with just the image
                conversation.append(
                    {
                        "role": "user",
                        "content": [{"type": "image"}],
                    }
                )
            else:
                conversation[last_user_idx]["content"].append({"type": "image"})

        # Save for debugging parity with other wrappers
        self.inputs = conversation

        # -------- 3) Build prompt & model inputs via chat template --------
        # Pattern recommended in HF model card for LLaVA-NeXT.
        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )

        if image is not None:
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=prompt,
                return_tensors="pt",
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # -------- 4) Run generation --------
        # Decide sampling vs greedy based on temperature
        do_sample = temperature > 0.0

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                **gen_kwargs,
            )

        # -------- 5) Decode only the newly generated tokens --------
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][input_len:]
        text = self.processor.decode(generated_ids, skip_special_tokens=True)

        # We don't have real usage stats from HF here; keep _last_usage as None
        self._last_usage = None

        return text.strip()

    # For EfficiencyTracker; will just return None so it skips token accounting.
    def get_last_usage(self) -> Optional[Dict[str, Any]]:
        return self._last_usage

    # Simple batch helper mirroring Anthropic/OpenAI wrappers
    def batch_generate(self, conversations, images, **kwargs):
        outputs = []
        for msgs, img in zip(conversations, images):
            outputs.append(self.generate(msgs, img, **kwargs))
        return outputs


# Backwards-compatible alias if you want to mirror Anthropic/OpenAI naming
LlavaWrapper = LlavaModel
