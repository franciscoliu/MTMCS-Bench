# model/__init__.py
import torch
from .base_wrapper import BaseChatModel
from .openai_wrapper import OpenaiModel  # OpenAIWrapper alias also exists
from .anthropic_wrapper import AnthropicModel  # new: Claude / Anthropic wrapper
from .llava_wrapper import LlavaModel
from .idefics_wrapper import Idefics3Model
from .qwen3_wrapper import Qwen3VLModel
from .instructionblip_wrapper import InstructBlipModel   # <-- NEW
from .llama_vision_wrapper import LlamaVisionModel  # <-- NEW
# from .deepseek_vl_wrapper import DeepseekVL2Model  # <--- NEW

def load_model(args) -> BaseChatModel:
    """
    Factory for MCS-Bench.

    Currently supports:
        --model_type gpt / openai       -> OpenaiModel (OpenAI API, incl. VLMs)
        --model_type claude / anthropic -> AnthropicModel (Claude models)

    You can extend this further for Qwen/LLaVA etc.
    """
    model_type = args.model_type.lower()
    model_path = args.model_path  # e.g., "gpt-4o" or "claude-3-5-sonnet-20241022"

    # -------- OpenAI / GPT family --------
    if "gpt" in model_type or "openai" in model_type:
        api_key = getattr(args, "api_key", None)
        org = getattr(args, "organization", None)
        generation_config = {}
        if org is not None:
            generation_config["organization"] = org

        return OpenaiModel(
            model_name=model_path,
            api_keys=api_key,
            generation_config=generation_config if generation_config else None,
        )

    # -------- Anthropic / Claude family --------
    elif "claude" in model_type or "anthropic" in model_type:
        api_key = getattr(args, "api_key", None)
        # You can extend generation_config later if needed (e.g., extra params)
        generation_config = {}

        return AnthropicModel(
            model_name=model_path,
            api_keys=api_key,
            generation_config=generation_config if generation_config else None,
        )

    elif "llava" in model_type:
        # Example: --model_type llava --model_path llava-hf/llava-v1.6-mistral-7b-hf
        return LlavaModel(
            model_name=model_path,
            # For 72B you probably want device_map="auto":
            # hf_kwargs forwarded inside wrapper
            torch_dtype=torch.float16,
            device_map="auto" if "72b" in model_path else "auto",
        )

    elif ("idefics3" in model_type) or ("idefics" in model_type):
        # Example:
        #   --model_type idefics3
        #   --model_path HuggingFaceM4/Idefics3-8B-Llama3
        return Idefics3Model(
            model_name=model_path,
            # defaults: device = cuda if available, dtype = bfloat16
        )

    elif ("qwen3_32b" in model_type) or ("qwen3_8b" in model_type):
        return Qwen3VLModel(model_name=model_path)

    elif "instructblip_7b" in model_type:
        # local HF model: Salesforce/instructblip-vicuna-7b (or a fine-tuned variant)
        return InstructBlipModel(model_name=model_path)

    elif "llama" in model_type or "llama_vision" in model_type:
        return LlamaVisionModel(
            model_name=model_path,
        )


    raise ValueError(f"Unsupported model_type: {args.model_type}")
