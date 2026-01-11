import base64
import warnings
from typing import List, Dict, Optional, Union

from PIL import Image
from openai import OpenAI, BadRequestError

from .base_wrapper import BaseChatModel


class OpenaiModel(BaseChatModel):
    """
    OpenAI VLM wrapper using Chat Completions API (for gpt-4o, gpt-4o-mini, etc.)
    and Responses API (for gpt-5*).

    It supports:
      - messages as list[{"role": ..., "content": ...}] (MCS-Bench inference.py)
      - image as PIL.Image.Image OR string path/URL
      - or old style: messages as str / list[str], images as paths
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
            model_name: OpenAI model name, e.g. "gpt-4o" or "gpt-5-mini".
            api_keys / api_key: either name is accepted, for robustness.
            generation_config: optional dict, e.g. {"organization": "..."}.
        """
        # accept both api_keys and api_key for safety
        key = api_keys if api_keys is not None else api_key

        if key is not None:
            if generation_config and "organization" in generation_config:
                self.client = OpenAI(api_key=key, organization=generation_config["organization"])
            else:
                self.client = OpenAI(api_key=key)
        else:
            # Fallback to env vars
            if generation_config and "organization" in generation_config:
                self.client = OpenAI(organization=generation_config["organization"])
            else:
                self.client = OpenAI()

        self.model_name = model_name
        self.generation_config = generation_config if generation_config is not None else {}
        self.seed = 42

        # track last usage for efficiency logging
        self._last_usage: Optional[Dict[str, Optional[int]]] = None

    # ---------- image helpers ----------

    def encode_image_from_pil(self, image: Image.Image, fmt: str = "PNG") -> str:
        import io
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def encode_image_from_path(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def encode_image(self, image: Union[Image.Image, str]) -> str:
        """
        Accept either a PIL image or a path, and return base64 string.
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
        max_new_tokens: int = 256,
        **kwargs,
    ) -> str:
        """
        Unified generate function compatible with:
          - MCS_Bench inference.py:
                generate(messages=list[dict], image=PIL.Image.Image, ...)
          - old style:
                generate(messages=str or list[str], images=path or list[path])
        """
        # reset usage for this call
        self._last_usage = None

        # ---- normalize messages ----
        msg_list: List[str] = []
        if isinstance(messages, str):
            msg_list = [messages]
        elif isinstance(messages, list):
            if len(messages) == 0:
                msg_list = []
            elif isinstance(messages[0], dict) and "content" in messages[0]:
                # new style: list[{"role": ..., "content": ...}]
                msg_list = [m["content"] for m in messages]
            else:
                # assume list[str]
                msg_list = list(messages)
        else:
            raise TypeError(f"Unsupported messages type: {type(messages)}")

        # ---- normalize images ----
        if isinstance(image, list):
            img_list = image
        else:
            img_list = [image] if image is not None else [None]

        self.inputs = []
        n = len(msg_list)
        for idx, message_text in enumerate(msg_list):
            msg = {
                "role": "user",
                "content": [],
            }

            # text part
            text_conv = {"type": "text", "text": message_text}
            msg["content"].append(text_conv)

            # choose which image (if any) to attach
            img_to_use = None
            if len(img_list) == n:          # old style 1:1 messages/images
                img_to_use = img_list[idx]
            elif len(img_list) == 1 and idx == n - 1:
                img_to_use = img_list[0]

            # NOTE: GPT-5-mini is text-only; we still keep image code here
            # for other models (gpt-4o, etc.).
            if img_to_use is not None:
                if isinstance(img_to_use, str) and img_to_use.startswith("http"):
                    image_conv = {"type": "image_url", "image_url": {"url": img_to_use}}
                else:
                    base64_image = self.encode_image(img_to_use)
                    image_conv = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                msg["content"].append(image_conv)

            self.inputs.append(msg)

        # ---- call OpenAI with retry ----
        num_attempts = 0
        while num_attempts < 5:
            num_attempts += 1
            try:
                # Merge extra kwargs + generation_config
                merged_cfg = {**kwargs, **self.generation_config}

                # -----------------------------
                # PATH 1: GPT-5 family -> Responses API
                # -----------------------------
                if self.model_name.startswith("gpt-5"):
                    # GPT-5 mini etc. do NOT support temperature/top_p/logprobs/max_tokens
                    for bad in ("temperature", "top_p", "logprobs", "max_tokens", "max_completion_tokens"):
                        merged_cfg.pop(bad, None)

                    # Flatten conversation into a single text block
                    convo_lines = []
                    for m in self.inputs:
                        role = m.get("role", "user")
                        parts = m.get("content", [])
                        texts = []
                        for p in parts:
                            if p.get("type") == "text":
                                texts.append(p.get("text", ""))
                        if not texts:
                            continue
                        convo_lines.append(f"{role.capitalize()}: " + "\n".join(texts))
                    full_text = "\n".join(convo_lines) if convo_lines else ""

                    # DEBUG: show prompt preview
                    print("=" * 80)
                    print(f"[DEBUG][gpt-5] model_name: {self.model_name}")
                    print("[DEBUG][gpt-5] full_text preview (first 400 chars):")
                    print("    " + full_text[:400].replace("\n", " ") + ("..." if len(full_text) > 400 else ""))
                    print("-" * 80)

                    # Responses API call
                    response = self.client.responses.create(
                        model=self.model_name,
                        input=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": full_text}
                                ],
                            }
                        ],
                        text={"format": {"type": "text"}},
                        reasoning={"effort": "low"},
                        max_output_tokens=256,
                        **merged_cfg,
                    )

                    # # DEBUG: inspect raw response object
                    # print("[DEBUG][gpt-5] responses.create returned.")
                    # print("[DEBUG][gpt-5] type(response):", type(response))
                    # print("[DEBUG][gpt-5] dir(response):", dir(response))
                    # print("[DEBUG][gpt-5] response.output_text:", repr(getattr(response, "output_text", None)))
                    # print("[DEBUG][gpt-5] response.output:", repr(getattr(response, "output", None)))

                    # usage tracking for Responses API
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        in_tok = getattr(usage, "input_tokens", None)
                        out_tok = getattr(usage, "output_tokens", None)
                        total_tok = getattr(usage, "total_tokens", None)
                        self._last_usage = {
                            "input_tokens": in_tok,
                            "output_tokens": out_tok,
                            "total_tokens": total_tok,
                        }
                    else:
                        self._last_usage = None

                    # Extract output text
                    raw_cleaned = None

                    # 1) Try high-level helper if present
                    output_text = getattr(response, "output_text", None)
                    if isinstance(output_text, str) and output_text.strip():
                        raw_cleaned = output_text.strip()
                        # print("[DEBUG][gpt-5] using response.output_text.")
                    else:
                        # 2) Fallback: scan output blocks
                        output_field = getattr(response, "output", None)
                        if output_field:
                            for i, item in enumerate(output_field):
                                # print(f"[DEBUG][gpt-5] output[{i}] type:", getattr(item, "type", None))
                                content = getattr(item, "content", None)
                                # print(f"[DEBUG][gpt-5] output[{i}].content:", repr(content))
                                # Some SDKs: item.text directly
                                if hasattr(item, "text") and isinstance(item.text, str) and item.text.strip():
                                    raw_cleaned = item.text.strip()
                                    # print("[DEBUG][gpt-5] found text on item.text.")
                                    break
                                # Or blocks with type="output_text"
                                if content:
                                    for j, c in enumerate(content):
                                        c_type = getattr(c, "type", None)
                                        c_text = getattr(c, "text", None)
                                        # print(f"[DEBUG][gpt-5]   content[{j}] type:", c_type)
                                        if c_type == "output_text" and isinstance(c_text, str) and c_text.strip():
                                            raw_cleaned = c_text.strip()
                                            # print("[DEBUG][gpt-5] found text on content block with type='output_text'.")
                                            break
                                    if raw_cleaned is not None:
                                        break

                    if raw_cleaned is None:
                        print("[DEBUG][gpt-5] !!! FAILED TO EXTRACT TEXT FROM RESPONSES OBJECT !!!")
                        print("[DEBUG][gpt-5] full response repr:")
                        print(response)
                        print("=" * 80)
                        raise RuntimeError("⚠️ No valid response content from OpenAI model (gpt-5 path).")

                    # print("[DEBUG][gpt-5] raw_cleaned preview (first 200 chars):")
                    print("    " + raw_cleaned[:200].replace("\n", " ") + ("..." if len(raw_cleaned) > 200 else ""))
                    print("=" * 80)

                    return raw_cleaned

                # -----------------------------
                # PATH 2: non-GPT-5 -> Chat Completions (original behavior)
                # -----------------------------
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=self.inputs,
                        seed=self.seed,
                        temperature=temperature,
                        max_tokens=max_new_tokens,
                        **merged_cfg,
                    )

                    # store usage for efficiency logging
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        prompt_tokens = getattr(usage, "prompt_tokens", None)
                        completion_tokens = getattr(usage, "completion_tokens", None)
                        total_tokens = getattr(usage, "total_tokens", None)

                        self._last_usage = {
                            "input_tokens": prompt_tokens,
                            "output_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                        }
                    else:
                        self._last_usage = None

                    return response.choices[0].message.content or ""

            except BadRequestError as e:
                print(f"OpenAI BadRequestError: {e}")
                # If it's a parameter error, no point retrying same thing
                continue
            except Exception as e:
                print(f"OpenAI server offers this error: {e}")
                continue

        # If all retries failed, return empty string
        return ""

    def batch_generate(self, conversations, batches, **kwargs):
        """
        Same as your original style: just a simple wrapper that calls generate.
        """
        responses = []
        for conversation, image in zip(conversations, batches):
            if isinstance(conversation, str):
                warnings.warn(
                    "For batch generation based on several conversations, provide a list[str] "
                    "for each conversation. Using list[list[str]] will avoid this warning."
                )
            responses.append(self.generate(conversation, image, **kwargs))
        return responses

    # ---------- expose last usage to efficiency tracker ----------

    def get_last_usage(self) -> Optional[Dict[str, Optional[int]]]:
        """
        Return token usage for the *last* successful generate() call.

        Example return value:
          {
            "input_tokens": 123,
            "output_tokens": 45,
            "total_tokens": 168,
          }
        or None if not available.
        """
        return self._last_usage


# Backwards-compatible alias, in case anything imports OpenAIWrapper
OpenAIWrapper = OpenaiModel
