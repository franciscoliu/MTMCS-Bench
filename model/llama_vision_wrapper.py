import math
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

from .base_wrapper import BaseChatModel


class LlamaVisionModel(BaseChatModel):
    """
    Wrapper for:
      - meta-llama/Llama-3.2-11B-Vision-Instruct
      - meta-llama/Llama-3.2-90B-Vision-Instruct

    API is compatible with your other wrappers:

      generate(
          messages,               # str | list[str] | list[{"role","content"}]
          image=None,             # optional PIL.Image or path handled upstream
          temperature=0.0,
          max_new_tokens=256,
          ...
      )
    """

    def __init__(
        self,
        model_name: str,
        api_keys: Optional[str] = None,   # unused but kept for API parity
        api_key: Optional[str] = None,    # unused
        generation_config: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.generation_config = generation_config or {}

        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

        # for parity with other wrappers / EfficiencyTracker
        self.seed = 42
        self.inputs: List[Dict[str, Any]] = []
        self._last_usage: Optional[Dict[str, Any]] = None

        self.hf_model = self.model
        self.hf_processor = self.processor
        self.model_family = "llama"

    # ---------- message normalization ----------

    def _normalize_messages(
        self, messages: Union[str, List[str], List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Normalize to HF-style messages:
          [{"role": "user"|"assistant","content":[{"type":"text","text": ...}, ...]}, ...]
        (image block will be injected later if needed.)
        """
        if isinstance(messages, str):
            msg_objs = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            if not messages:
                msg_objs = []
            elif isinstance(messages[0], dict) and "content" in messages[0]:
                msg_objs = messages  # already [{role, content}, ...]
            else:
                # list[str] -> each is a user turn
                msg_objs = [{"role": "user", "content": m} for m in messages]
        else:
            raise TypeError(f"Unsupported messages type: {type(messages)}")

        hf_messages: List[Dict[str, Any]] = []
        for m in msg_objs:
            role = m.get("role", "user")
            text = m.get("content", "")

            if role not in ("user", "assistant"):
                role = "user"

            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            # allow empty assistant turns? they don't really make sense; skip them
            if not text.strip():
                continue

            hf_messages.append(
                {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                        }
                    ],
                }
            )
        return hf_messages

    def _inject_image_block(
        self, messages: List[Dict[str, Any]], image: Optional[Image.Image]
    ) -> List[Dict[str, Any]]:
        """
        Insert an {"type": "image"} block into the last user message's content,
        matching the HF example:
            {"role": "user", "content": [{"type":"image"}, {"type":"text","text": "..."}]}
        """
        if image is None or not messages:
            return messages

        last_user_idx = None
        for i in reversed(range(len(messages))):
            if messages[i]["role"] == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            # no user message; create one
            messages.append({"role": "user", "content": []})
            last_user_idx = len(messages) - 1

        content = messages[last_user_idx].get("content", [])
        if not isinstance(content, list):
            content = []

        # put image block first, then any text
        content = [{"type": "image"}] + content
        messages[last_user_idx]["content"] = content
        return messages

    # ---------- token usage helper ----------

    def get_last_usage(self) -> Optional[Dict[str, Any]]:
        """
        Exposed for EfficiencyTracker._maybe_get_usage().
        Returns:
          {
            "input_tokens": int,
            "output_tokens": int
          }
        """
        return self._last_usage

    # ---------- core generate API ----------

    def generate(
        self,
        messages,
        image: Optional[Image.Image] = None,
        clear_old_history: bool = True,  # kept for API compatibility; unused
        temperature: float = 0.0,
        max_new_tokens: int = 128,
        **kwargs,
    ) -> str:
        # 1) normalize messages -> HF multimodal format
        hf_messages = self._normalize_messages(messages)

        # 2) inject image token if any
        hf_messages = self._inject_image_block(hf_messages, image)

        # store what we actually send (for debugging / parity)
        self.inputs = hf_messages

        # 3) build chat template text
        input_text = self.processor.apply_chat_template(
            hf_messages,
            add_generation_prompt=True,
        )

        # 4) tokenize + (optionally) add image
        if image is not None:
            inputs = self.processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.model.device)
        else:
            inputs = self.processor(
                text=input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.model.device)

        # 5) generation args: map temperature -> do_sample
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
        }
        if temperature is not None and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(temperature)
        else:
            gen_kwargs["do_sample"] = False

        # allow caller-provided generation_config + **kwargs to override defaults
        gen_kwargs.update(self.generation_config)
        gen_kwargs.update(kwargs)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        # inputs["input_ids"] includes the prompt; we want to decode *only* new tokens
        in_len = inputs["input_ids"].shape[1]
        out_ids = generated_ids[0]
        gen_only_ids = out_ids[in_len:]  # strip prompt

        self._last_usage = {
            "input_tokens": int(in_len),
            "output_tokens": int(gen_only_ids.shape[0]),
        }

        text = self.processor.decode(gen_only_ids, skip_special_tokens=True)
        return text

    def batch_generate(self, conversations, batches, **kwargs):
        """
        Simple loop over generate(), mirroring other wrappers.
        conversations: list of messages (same format as generate.messages)
        batches:       list of images (PIL.Image or None)
        """
        responses = []
        for convo, image in zip(conversations, batches):
            responses.append(self.generate(convo, image, **kwargs))
        return responses
