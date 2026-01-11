import os
import torch

# ---- HF imports ----
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForVision2Seq,
    GenerationConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    IdeficsForVisionText2Text,
    MllamaForConditionalGeneration,
)

# Qwen family (Qwen2.5-VL + Qwen3-VL)
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

# Idefics3 (vision2seq)
# uses AutoModelForVision2Seq + AutoProcessor

# DeepSeek-VL2
try:
    from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
except ImportError:
    DeepseekVLV2Processor = None
    DeepseekVLV2ForCausalLM = None

# ---- API clients (optional deps) ----
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


def _check_package(name, obj):
    if obj is None:
        raise ImportError(
            f"`{name}` package not available. Please install it before using this model type."
        )


def load_model(args):
    """
    Unified loader for all models.

    Returns:
        (model_or_client, processor, tokenizer)

    - For local HF VLMs:
        model_or_client: HF model instance
        processor:       HF processor
        tokenizer:       tokenizer (if available)

    - For API models (gpt / claude / gemini):
        model_or_client: API client object (OpenAI / Anthropic / GenerativeModel)
        processor:       None
        tokenizer:       None

    You will branch on args.model_type in inference.py to call them correctly.
    """
    model_type = args.model_type.lower()
    model_path = args.model_path
    # default; some models will override to bfloat16
    dtype = torch.float16

    # -----------------------------
    # 1) LLaVA-NeXT family
    #    (covers llava-v1.6-mistral-7b-hf, llava-next-72b-hf, etc.)
    # -----------------------------
    if model_type in {"llava", "llava_next", "llava_v1_6"}:
        print(f"Loading LLaVA-NeXT model from {model_path} ...")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=dtype,
        ).eval()

        processor = LlavaNextProcessor.from_pretrained(model_path)
        tokenizer = processor.tokenizer
        return model, processor, tokenizer

    # -----------------------------
    # 2) Qwen VL (Qwen2.5-VL + Qwen3-VL)
    # -----------------------------
    elif model_type in {"qwen", "qwen_vl", "qwen3_vl"}:
        print(f"Loading Qwen-VL model from {model_path} ...")

        min_pixels = 256 * 28 * 28
        max_pixels = 256 * 28 * 28

        model_path_lower = model_path.lower()

        # Qwen3-VL branch
        if "qwen3" in model_path_lower:
            _check_package("transformers (Qwen3VLForConditionalGeneration)", Qwen3VLForConditionalGeneration)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).eval()

        # Qwen2.5-VL branch
        else:
            _check_package("transformers (Qwen2_5_VLForConditionalGeneration)", Qwen2_5_VLForConditionalGeneration)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            ).eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        return model, processor, tokenizer

    # -----------------------------
    # 3) InstructBLIP-7B
    # -----------------------------
    elif model_type in {"instructionblip", "instructblip"}:
        print(f"Loading InstructBLIP model from {model_path} ...")
        processor = InstructBlipProcessor.from_pretrained(model_path)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=dtype,
        ).eval()
        tokenizer = processor.tokenizer
        return model, processor, tokenizer

    # -----------------------------
    # 4) Idefics (v1 / v2)
    # -----------------------------
    elif model_type in {"idefics", "idefics2"}:
        print(f"Loading IDEFICS model from {model_path} ...")
        processor = AutoProcessor.from_pretrained(model_path)
        model = IdeficsForVisionText2Text.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()
        model.generation_config = GenerationConfig.from_pretrained(model_path)
        tokenizer = processor.tokenizer
        return model, processor, tokenizer

    # -----------------------------
    # 5) Idefics3-8B-Llama3
    #     (AutoModelForVision2Seq + AutoProcessor)
    # -----------------------------
    elif model_type in {"idefics3", "idefics3_llama3"}:
        print(f"Loading IDEFICS3 model from {model_path} ...")
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()
        tokenizer = processor.tokenizer
        return model, processor, tokenizer

    # -----------------------------
    # 6) Llama-3.2 Vision (11B / 90B)
    #     meta-llama/Llama-3.2-11B-Vision-Instruct
    #     meta-llama/Llama-3.2-90B-Vision-Instruct
    # -----------------------------
    elif model_type in {"llama_vision", "llama3_vision", "llama_3_2_vision"}:
        print(f"Loading Llama-3.2 Vision model from {model_path} ...")
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else None
        return model, processor, tokenizer


    # -----------------------------
    # 7) GPT (OpenAI) – GPT-5, GPT-4o, etc.
    #     model_path is the actual API model id, e.g.:
    #       --model_type gpt --model_path gpt-5.1
    #       --model_type gpt --model_path gpt-4o
    # -----------------------------
    elif model_type in {"gpt", "openai"}:
        _check_package("openai", OpenAI)
        print(f"Creating OpenAI client for model '{model_path}' ...")
        client = OpenAI()  # uses OPENAI_API_KEY env var
        # For API models we return client and no processor/tokenizer
        return client, None, None

    # -----------------------------
    # 8) Claude (Anthropic) – Sonnet / Haiku / Opus
    #     model_path is API id, e.g.:
    #       --model_type claude --model_path claude-3-5-sonnet-20241022
    #       --model_type claude --model_path claude-3-opus-20240229
    # -----------------------------
    elif model_type in {"claude", "anthropic"}:
        _check_package("anthropic", anthropic)
        print(f"Creating Anthropic client for model '{model_path}' ...")
        client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
        return client, None, None

    # -----------------------------
    # 10) Fallback
    # -----------------------------
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            "Supported types include: "
            "llava, qwen, instructionblip, idefics, idefics3, "
            "llama_vision, deepseek_vl2, gpt, claude, gemini."
        )
