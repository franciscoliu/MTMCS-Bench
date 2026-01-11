# model/qwen3_wrapper.py

from typing import List, Dict, Optional, Union

from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from .base_wrapper import BaseChatModel


class Qwen3VLModel(BaseChatModel):
    """
    Wrapper for Qwen3-VL models (e.g.,
      - Qwen/Qwen3-VL-8B-Instruct
      - Qwen/Qwen3-VL-32B-Instruct

    Interface is aligned with your other wrappers:

      - __init__(model_name, api_keys/api_key, generation_config)
      - generate(messages, image=None, temperature=..., max_new_tokens=...)
      - batch_generate(conversations, batches, ...)

    Supported `messages` formats:
      - str
      - list[str]
      - list[{"role": ..., "content": ...}]

    `image` can be:
      - None
      - PIL.Image.Image
      - local path (str)
    """

    def __init__(
        self,
        model_name: str,
        api_keys: Optional[str] = None,   # kept for API-compat; unused
        api_key: Optional[str] = None,    # kept for API-compat; unused
        generation_config: Optional[Dict] = None,
        **model_kwargs,
    ):
        # Load model & processor using the official HF pattern
        self.model_name = model_name
        self.device = "cuda"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype="auto",      # or torch.bfloat16 if you prefer
            # device_map="auto",
            device_map={"": self.device},
            **model_kwargs,
        ).eval()
        # self.processor = AutoProcessor.from_pretrained(model_name)

        min_pixels = 256 * 28 * 28
        max_pixels = 256 * 28 * 28

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True,
        )

        self.generation_config = generation_config if generation_config is not None else {}

        # for parity with other wrappers
        self.seed = 42
        self.inputs: List[Dict] = []
        self._last_usage: Optional[Dict[str, int]] = None

        self.hf_model = self.model
        self.hf_processor = self.processor
        self.model_family = "qwen"

    # ---------- core generate API ----------

    def _normalize_messages(
        self, messages
    ) -> List[Dict[str, List[Dict[str, str]]]]:
        """
        Convert your generic messages format into Qwen3-VL chat format:

        [
          {"role": "user" | "assistant" | "system",
           "content": [{"type":"text","text": ...}, ...]},
          ...
        ]
        """
        if isinstance(messages, str):
            msg_objs = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            if len(messages) == 0:
                msg_objs = []
            elif isinstance(messages[0], dict) and "content" in messages[0]:
                msg_objs = messages
            else:
                # list[str] -> each is a user turn
                msg_objs = [{"role": "user", "content": m} for m in messages]
        else:
            raise TypeError(f"Unsupported messages type: {type(messages)}")

        qwen_msgs: List[Dict] = []
        for m in msg_objs:
            role = m.get("role", "user")
            text = m.get("content", "")

            if role not in ("user", "assistant", "system"):
                role = "user"

            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            if not text.strip():
                # skip empty turns
                continue

            qwen_msgs.append(
                {
                    "role": role,
                    "content": [
                        {"type": "text", "text": text},
                    ],
                }
            )

        return qwen_msgs

    def _attach_image(
        self,
        qwen_msgs: List[Dict],
        image: Optional[Union[Image.Image, str]],
    ) -> List[Dict]:
        """
        Attach image block to the last user message (or create one if none).
        Qwen3-VL processor accepts PIL images or paths/URLs in the 'image' field.
        """
        if image is None:
            return qwen_msgs

        # find last user message
        last_user_idx = None
        for i in reversed(range(len(qwen_msgs))):
            if qwen_msgs[i]["role"] == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            last_user_idx = len(qwen_msgs)
            qwen_msgs.append({"role": "user", "content": []})

        qwen_msgs[last_user_idx]["content"].insert(
            0,
            {
                "type": "image",
                "image": image,  # PIL or path; processor will handle it
            },
        )
        return qwen_msgs

    def generate(
        self,
        messages,
        image=None,
        clear_old_history: bool = True,
        temperature: float = 0.0,
        max_new_tokens: int = 128,
        **kwargs,
    ) -> str:
        """
        Main generate entry point used by your inference.py.
        """
        # 1) Normalize messages to Qwen3-VL format
        qwen_msgs = self._normalize_messages(messages)
        qwen_msgs = self._attach_image(qwen_msgs, image)

        # Keep for debugging parity
        self.inputs = qwen_msgs

        # 2) Build model inputs via chat template
        inputs = self.processor.apply_chat_template(
            qwen_msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        do_sample = temperature > 0.0

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": max(temperature, 1e-5) if do_sample else None,
        }
        # prune None so transformers doesn't complain
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        gen_kwargs.update(self.generation_config)
        gen_kwargs.update(kwargs)

        # 3) Generate
        generated_ids = self.model.generate(**inputs, **gen_kwargs)

        # 4) Slice off the prompt part (same as HF example)
        in_len = inputs["input_ids"].shape[1]
        out_ids = generated_ids[0]
        gen_only_ids = out_ids[in_len:]

        # 5) Decode
        output_text = self.processor.decode(
            gen_only_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # 6) Track token usage for EfficiencyTracker
        self._last_usage = {
            "input_tokens": int(in_len),
            "output_tokens": int(gen_only_ids.shape[0]),
        }

        return output_text

    def get_last_usage(self) -> Optional[Dict[str, int]]:
        """
        Exposed for EfficiencyTracker._maybe_get_usage().
        """
        return self._last_usage

    def batch_generate(self, conversations, batches, **kwargs):
        """
        Simple loop over generate(), mirroring your other wrappers.
        `conversations` = iterable of messages; `batches` = iterable of images.
        """
        responses = []
        for conv, img in zip(conversations, batches):
            responses.append(self.generate(conv, image=img, **kwargs))
        return responses


# Alias for symmetry with other wrappers if you want
Qwen3VLWrapper = Qwen3VLModel
