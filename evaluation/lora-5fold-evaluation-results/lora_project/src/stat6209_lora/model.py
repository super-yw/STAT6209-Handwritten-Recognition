from __future__ import annotations

import importlib
from typing import Optional

import torch

try:
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
except ImportError as exc:  # pragma: no cover - runtime dependency
    LoraConfig = PeftModel = get_peft_model = prepare_model_for_kbit_training = None
    _PEFT_IMPORT_ERROR = exc
else:
    _PEFT_IMPORT_ERROR = None

try:
    from transformers import AutoProcessor, BitsAndBytesConfig
except ImportError as exc:  # pragma: no cover - runtime dependency
    AutoProcessor = BitsAndBytesConfig = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


def _ensure_runtime_imports() -> None:
    if _TRANSFORMERS_IMPORT_ERROR is not None:
        raise ImportError(
            "transformers is required for this project. Install lora_project/requirements.txt first."
        ) from _TRANSFORMERS_IMPORT_ERROR
    if _PEFT_IMPORT_ERROR is not None:
        raise ImportError(
            "peft is required for this project. Install lora_project/requirements.txt first."
        ) from _PEFT_IMPORT_ERROR


def parse_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[key]


def parse_target_modules(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_model_class():
    _ensure_runtime_imports()
    transformers_module = importlib.import_module("transformers")
    for class_name in (
        "Qwen3VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ):
        model_class = getattr(transformers_module, class_name, None)
        if model_class is not None:
            return model_class
    raise ImportError("Could not find a compatible Qwen3-VL model class in transformers.")


def load_processor(
    model_name_or_path: str,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
):
    _ensure_runtime_imports()
    kwargs = {"trust_remote_code": True}
    if min_pixels is not None:
        kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        kwargs["max_pixels"] = max_pixels
    return AutoProcessor.from_pretrained(model_name_or_path, **kwargs)


def _build_quantization_config(load_in_4bit: bool, compute_dtype: str):
    if not load_in_4bit:
        return None
    _ensure_runtime_imports()
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=parse_torch_dtype(compute_dtype),
    )


def _build_model_load_kwargs(
    torch_dtype: str,
    load_in_4bit: bool,
    bnb_4bit_compute_dtype: str,
):
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": parse_torch_dtype(torch_dtype),
    }
    quantization_config = _build_quantization_config(load_in_4bit, bnb_4bit_compute_dtype)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    return model_kwargs


def load_model_for_training(
    model_name_or_path: str,
    torch_dtype: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: str,
    gradient_checkpointing: bool,
    load_in_4bit: bool,
    bnb_4bit_compute_dtype: str,
):
    _ensure_runtime_imports()
    model_class = resolve_model_class()
    model_kwargs = _build_model_load_kwargs(torch_dtype, load_in_4bit, bnb_4bit_compute_dtype)
    model = model_class.from_pretrained(model_name_or_path, **model_kwargs)
    model.config.use_cache = False

    if load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )
    elif gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=parse_target_modules(lora_target_modules),
    )
    model = get_peft_model(model, peft_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def load_model_for_inference(
    model_name_or_path: str,
    adapter_path: Optional[str],
    torch_dtype: str,
    load_in_4bit: bool,
    bnb_4bit_compute_dtype: str,
):
    _ensure_runtime_imports()
    model_class = resolve_model_class()
    model_kwargs = _build_model_load_kwargs(torch_dtype, load_in_4bit, bnb_4bit_compute_dtype)
    model = model_class.from_pretrained(model_name_or_path, **model_kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def infer_model_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device
