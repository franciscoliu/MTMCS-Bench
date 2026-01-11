# model/base_wrapper.py

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from PIL import Image


class BaseChatModel(ABC):
    """
    Unified interface that inference.py will use for all models.

    Each concrete model (OpenAI, Claude, Gemini, Qwen, LLaVA, etc.)
    should subclass this and implement `generate`.
    """

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        image: Optional[Image.Image] = None,
        temperature: float = 0.1,
        max_new_tokens: int = 256,
    ) -> str:
        """
        Args:
            messages: a list of {"role": "user" | "assistant" | "system",
                                 "content": "<text>"}
            image: optional PIL image for vision-language models (None for text-only).
        Returns:
            Model's text response.
        """
        ...
