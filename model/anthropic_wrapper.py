import base64
import io
from typing import List, Dict, Optional, Union

from PIL import Image
from anthropic import Anthropic, APIError

from .base_wrapper import BaseChatModel


class AnthropicModel(BaseChatModel):
    """
    Anthropic (Claude) VLM wrapper using the Messages API.

    Interface is made parallel to OpenaiModel:

      - __init__(model_name, api_keys/api_key, generation_config)
      - generate(messages, image=None, temperature=..., max_new_tokens=...)
      - batch_generate(conversations, batches, ...)

    Supported inputs:
      - messages:
          * str
          * list[str]
          * list[{"role": ..., "content": ...}]
      - image:
          * None
          * PIL.Image.Image
          * local file path (str)
    """

    def __init__(
        self,
        model_name: str,
        api_keys: Optional[str] = None,
        api_key: Optional[str] = None,
        generation_config=None,
    ):
        """
        Args:
            model_name: Anthropic model name, e.g. "claude-3-5-sonnet-20241022".
            api_keys / api_key: either name is accepted.
            generation_config: optional dict passed through to messages.create().
        """
        key = api_keys if api_keys is not None else api_key
        if key is not None:
            self.client = Anthropic(api_key=key)
        else:
            # fallback to ANTHROPIC_API_KEY env var
            self.client = Anthropic()

        self.model_name = model_name
        self.generation_config = generation_config if generation_config is not None else {}

        # for compatibility with your OpenAI wrapper
        self.seed = 42
        self.inputs: List[Dict] = []

        # ------------------------------
        # ADDED: storage for token usage
        # ------------------------------
        self._last_usage: Optional[Dict] = None


    # ---------- image helpers ----------
    # ---------- image helpers ----------

    def _shrink_and_compress(
        self,
        image: Image.Image,
        max_side: int = 1024,
        quality: int = 85,
    ) -> bytes:
        """
        Downscale and JPEG-compress an image so it stays well under Anthropic's 5MB limit.
        """
        # Ensure RGB (no alpha)
        img = image.convert("RGB")

        # Downscale if needed
        w, h = img.size
        longest = max(w, h)
        if longest > max_side:
            scale = max_side / float(longest)
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def encode_image_from_pil(self, image: Image.Image) -> (str, str):
        """
        Encode a PIL image to base64 string and return (b64, media_type).
        We resize + JPEG-compress to avoid exceeding Anthropic's 5MB limit.
        """
        data_bytes = self._shrink_and_compress(image)
        data_b64 = base64.b64encode(data_bytes).decode("utf-8")
        media_type = "image/jpeg"
        return data_b64, media_type

    def encode_image_from_path(self, image_path: str) -> (str, str):
        """
        Encode an image from disk to base64 string and return (b64, media_type),
        with resizing/compression.
        """
        with Image.open(image_path) as img:
            data_bytes = self._shrink_and_compress(img)
        data_b64 = base64.b64encode(data_bytes).decode("utf-8")
        media_type = "image/jpeg"
        return data_b64, media_type

    def encode_image(self, image: Union[Image.Image, str]) -> (str, str):
        """
        Accept either a PIL image or a local path, return (base64, media_type).
        (Remote URLs are not handled here because your pipeline passes PIL images.)
        """
        if isinstance(image, Image.Image):
            return self.encode_image_from_pil(image)
        elif isinstance(image, str):
            return self.encode_image_from_path(image)
        else:
            raise TypeError(f"Unsupported image type for encode_image: {type(image)}")

    # ---------- core generate API ----------

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
        Unified generate function compatible with your inference.py:

            generate(messages=list[dict], image=PIL.Image.Image, ...)

        Also supports:
            - messages: str
            - messages: list[str]

        `image` is an optional single image (PIL.Image or path) that
        will be attached to the **last user message**, mirroring OpenaiModel.
        """
        # ---- normalize messages into Anthropic format ----
        # Anthropic expects:
        #   messages = [
        #     {"role": "user" | "assistant", "content": [ {...}, ... ]},
        #     ...
        #   ]
        core_msgs: List[Dict] = []

        # Convert incoming messages to a list of dicts with role/content
        if isinstance(messages, str):
            # single turn
            msg_objs = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            if len(messages) == 0:
                msg_objs = []
            elif isinstance(messages[0], dict) and "content" in messages[0]:
                # already [{role, content}, ...]
                msg_objs = messages
            else:
                # assume list[str] -> each becomes user turn
                msg_objs = [{"role": "user", "content": m} for m in messages]
        else:
            raise TypeError(f"Unsupported messages type: {type(messages)}")


        # Build Anthropic-style messages list
            # Build Anthropic-style messages list
        for m in msg_objs:
            role = m.get("role", "user")
            text = m.get("content", "")

            # Normalize role (we don't use top-level `system` field anymore)
            if role not in ("user", "assistant"):
                role = "user"
            if role == "system":
                role = "user"

            # Anthropic requires non-empty text in text blocks.
            # Skip messages whose content is empty/whitespace (e.g. empty assistant reply).
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            if not text.strip():
                # no text; we skip this turn entirely
                continue

            content_blocks = [{"type": "text", "text": text}]
            core_msgs.append({"role": role, "content": content_blocks})

        # ---- attach image (if any) to the last user message ----
        if image is not None and len(core_msgs) > 0:
            # find index of last user message
            last_user_idx = None
            for i in reversed(range(len(core_msgs))):
                if core_msgs[i]["role"] == "user":
                    last_user_idx = i
                    break

            if last_user_idx is None:
                # no user message -> attach as a new user message
                last_user_idx = len(core_msgs)
                core_msgs.append({"role": "user", "content": []})

            b64, media_type = self.encode_image(image)
            image_block = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            }
            core_msgs[last_user_idx]["content"].append(image_block)

        # store what we actually send (for debugging parity with OpenaiModel)
        self.inputs = core_msgs

        # ---- call Anthropic Messages API ----
        resp = self.client.messages.create(
            model=self.model_name,
            messages=self.inputs,  # already a proper list
            max_tokens=max_new_tokens,
            temperature=temperature,
            # NOTE: no `system=` here, to avoid the "system must be list" error
            **self.generation_config,
            **kwargs,
        )

        # -------------------------------------
        # ADDED: collect usage for EfficiencyTracker
        # -------------------------------------
        if getattr(resp, "usage", None) is not None:
            self._last_usage = {
                "input_tokens": getattr(resp.usage, "input_tokens", None),
                "output_tokens": getattr(resp.usage, "output_tokens", None),
            }
        else:
            self._last_usage = None

        # Anthropic returns content as a list of blocks
        # We take the first text block
        for block in resp.content:
            if block.type == "text":
                return block.text

        # fallback if no text block
        return ""

    def batch_generate(self, conversations, batches, **kwargs):
        """
        Mirror OpenaiModel: simple loop over generate().
        """
        responses = []
        for conversation, image in zip(conversations, batches):
            responses.append(self.generate(conversation, image, **kwargs))
        return responses

    # --------------------------------------
    # ADDED: expose usage to EfficiencyTracker
    # --------------------------------------
    def get_last_usage(self) -> Optional[Dict]:
        return self._last_usage


# Backwards-compatible alias, like OpenAIWrapper
AnthropicWrapper = AnthropicModel
