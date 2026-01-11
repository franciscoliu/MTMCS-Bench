# model/idefics3_wrapper.py

from typing import List, Dict, Optional, Union

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

from .base_wrapper import BaseChatModel


class Idefics3Model(BaseChatModel):
    """
    Local wrapper for HuggingFaceM4/Idefics3-8B-Llama3.

    Interface is kept aligned with your other wrappers:

      - __init__(model_name, ...)
      - generate(messages, image=None, temperature=..., max_new_tokens=...)
      - batch_generate(conversations, batches, ...)

    Assumptions (to match your MCS code):
      * At most ONE image per call (passed via `image` arg).
      * That image is conceptually associated with the LAST user turn.
      * When `image is None`, we just run pure-text chat.
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceM4/Idefics3-8B-Llama3",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        processor_kwargs: Optional[Dict] = None,
        model_kwargs: Optional[Dict] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")


        processor_kwargs = processor_kwargs or {}
        model_kwargs = model_kwargs or {}

        # You can control resolution via `size={"longest_edge": N*364}` in processor_kwargs if needed.
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            **processor_kwargs,
        )

        if torch_dtype is not None:
            model_kwargs.setdefault("torch_dtype", torch_dtype)

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            **model_kwargs,
        ).to(self.device)
        self.model.eval()

        # For compatibility with EfficiencyTracker (no token usage here)
        self.last_usage: Optional[Dict] = None

        # ============================
        # NEW: attributes for Immune
        # ============================
        # Immune wants direct access to HF model, processor, tokenizer, and a model_family string
        self.hf_model = self.model
        self.hf_processor = self.processor
        self.model_family = "idefics"  # <- string that immune_utils expects for this family

    # ---------------------------
    # Core generate
    # ---------------------------

    def _normalize_messages(self, messages) -> List[Dict[str, str]]:
        """
        Normalize incoming messages into a list[{'role', 'content'}]
        compatible with your OpenAI / Anthropic wrappers.
        """
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        if isinstance(messages, list):
            if not messages:
                return []
            # Already [{role, content}, ...] ?
            if isinstance(messages[0], dict) and "content" in messages[0]:
                return messages
            # Otherwise assume list[str]
            return [{"role": "user", "content": m} for m in messages]

        raise TypeError(f"Unsupported messages type for Idefics3Model: {type(messages)}")

    def generate(
        self,
        messages,
        image: Optional[Union[Image.Image, str]] = None,
        clear_old_history: bool = True,  # kept for API compatibility, unused
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> str:
        """
        Main entry point used by inference.py.

        - `messages`: str | list[str] | list[{'role','content'}]
        - `image`: single PIL.Image or path, attached to LAST user turn
        """
        msg_objs = self._normalize_messages(messages)

        # Build Idefics3-style chat messages:
        # each item: {"role": "user"/"assistant", "content": [ {"type": "text"/"image", ...}, ... ]}
        idefics_messages: List[Dict] = []

        # Index of last user message (for attaching the single image, if any)
        last_user_idx = None
        for i, m in enumerate(msg_objs):
            role = m.get("role", "user")
            if role not in ("user", "assistant", "system"):
                role = "user"
            # Treat system as user-style instruction (no dedicated system field in template)
            if role == "system":
                role = "user"

            text = m.get("content", "")
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            # We *do* allow empty text here (e.g., if you ever want an image-only prompt),
            # but we won't add an empty text block.

            content_blocks = []
            if text.strip():
                content_blocks.append({"type": "text", "text": text})

            idefics_messages.append({"role": role, "content": content_blocks})

            if role == "user":
                last_user_idx = i

        # Attach image (if provided) as a separate block on the last user message
        images_arg = None
        if image is not None and last_user_idx is not None:
            # Ensure the corresponding content list exists
            content_blocks = idefics_messages[last_user_idx]["content"]
            # Convention from the README: image block first, then text.
            content_blocks.insert(0, {"type": "image"})
            images_arg = [image]  # single image, single placeholder

        # Turn chat into a prompt string using the built-in template
        prompt = self.processor.apply_chat_template(
            idefics_messages,
            add_generation_prompt=True,
        )

        # Build model inputs
        if images_arg is not None:
            inputs = self.processor(
                text=prompt,
                images=images_arg,
                return_tensors="pt",
            )
        else:
            # pure text
            inputs = self.processor(
                text=prompt,
                return_tensors="pt",
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Sampling / decoding settings
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
        )
        if temperature is not None and temperature > 0.0:
            gen_kwargs.update(
                dict(
                    do_sample=True,
                    temperature=float(temperature),
                )
            )
        else:
            # deterministic
            gen_kwargs.update(
                dict(
                    do_sample=False,
                )
            )

        gen_kwargs.update(kwargs)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        # generated_ids = [prompt_tokens + new_tokens]
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        # Keep only the newly generated part
        gen_only_ids = generated_ids[:, prompt_len:]

        # If, for some reason, nothing new was generated, fall back to last token
        if gen_only_ids.shape[1] == 0:
            gen_only_ids = generated_ids[:, -1:]

        texts = self.processor.batch_decode(
            gen_only_ids,
            skip_special_tokens=True,
        )
        out = texts[0].strip() if texts else ""
        self.last_usage = None
        return out

    def batch_generate(self, conversations, batches, **kwargs):
        """
        Simple loop over generate(), mirroring other wrappers.

        `conversations`: iterable of messages (each accepted by generate())
        `batches`: iterable of images (each None | PIL | path)
        """
        responses = []
        for conv, img in zip(conversations, batches):
            responses.append(self.generate(conv, image=img, **kwargs))
        return responses

    # Optional hook for EfficiencyTracker (returns None → tokens N/A)
    def get_last_usage(self) -> Optional[Dict]:
        return self.last_usage


# Alias for consistency with your naming pattern, if you want it
Idefics3Wrapper = Idefics3Model
