from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from .data import (
    DEFAULT_BASE_MODEL,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROCESSED_ROOT,
    REPO_ROOT,
    Sample,
    build_user_messages,
    load_fold_samples,
    normalize_output,
)
from .model import infer_model_device, load_model_for_inference, load_processor


def _load_transformers_set_seed():
    try:
        from transformers import HfArgumentParser, set_seed
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "transformers is required for evaluation. Install lora_project/requirements.txt first."
        ) from exc
    return HfArgumentParser, set_seed


@dataclass
class EvalConfig:
    fold: int
    repo_root: str = str(REPO_ROOT)
    processed_root: str = str(DEFAULT_PROCESSED_ROOT)
    model_name_or_path: str = str(DEFAULT_BASE_MODEL)
    adapter_path: Optional[str] = None
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    output_path: Optional[str] = None
    report_name: Optional[str] = None
    split: str = "eval"
    use_base_model: bool = False
    torch_dtype: str = "bfloat16"
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    max_new_tokens: int = 16
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    seed: int = 42


def _open_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def generate_prediction(model, processor, sample: Sample, max_new_tokens: int) -> str:
    device = infer_model_device(model)
    messages = build_user_messages(sample.prompt_text)
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image = _open_image(sample.image_path)
    inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    prompt_length = int(inputs["attention_mask"].sum().item())
    generated_text = processor.batch_decode(
        generated_ids[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return normalize_output(generated_text)


def build_generation_report(
    model,
    processor,
    samples: list[Sample],
    fold: int,
    split: str,
    max_new_tokens: int,
    model_label: str,
    adapter_path: Optional[str],
) -> dict:
    start_time = time.time()
    predictions = []
    correct = 0
    total_by_category: Counter[str] = Counter()
    correct_by_category: Counter[str] = Counter()
    mismatches_by_category: dict[str, list[dict]] = defaultdict(list)

    was_training = model.training
    model.eval()
    try:
        for sample in samples:
            prediction = generate_prediction(model, processor, sample, max_new_tokens)
            target = normalize_output(sample.output_text)
            is_correct = prediction == target
            category = sample.category
            correct += int(is_correct)
            total_by_category[category] += 1
            correct_by_category[category] += int(is_correct)

            item = {
                "file_name": sample.file_name,
                "category": category,
                "source_no": sample.source_no,
                "target": target,
                "prediction": prediction,
                "correct": is_correct,
            }
            predictions.append(item)
            if not is_correct:
                mismatches_by_category[category].append(item)
    finally:
        if was_training:
            model.train()

    runtime = time.time() - start_time
    exact_match = correct / len(samples) if samples else 0.0
    by_category = {
        category: {
            "num_samples": total_by_category[category],
            "num_correct": correct_by_category[category],
            "exact_match": (
                correct_by_category[category] / total_by_category[category]
                if total_by_category[category]
                else 0.0
            ),
        }
        for category in sorted(total_by_category)
    }
    return {
        "fold": fold,
        "split": split,
        "model_label": model_label,
        "adapter_path": adapter_path,
        "num_samples": len(samples),
        "num_correct": correct,
        "exact_match": exact_match,
        "runtime": runtime,
        "samples_per_second": (len(samples) / runtime) if runtime > 0 else 0.0,
        "by_category": by_category,
        "mismatches_by_category": {
            category: items for category, items in sorted(mismatches_by_category.items())
        },
        "predictions": predictions,
    }


def write_generation_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def evaluate_loaded_model(
    model,
    processor,
    samples: list[Sample],
    fold: int,
    split: str,
    max_new_tokens: int,
    model_label: str,
    adapter_path: Optional[str],
    output_path: Optional[Path] = None,
) -> dict:
    report = build_generation_report(
        model=model,
        processor=processor,
        samples=samples,
        fold=fold,
        split=split,
        max_new_tokens=max_new_tokens,
        model_label=model_label,
        adapter_path=adapter_path,
    )
    if output_path is not None:
        write_generation_report(report, output_path)
    return report


def resolve_eval_output_path(config: EvalConfig, adapter_path: Optional[Path]) -> Path:
    if config.output_path:
        return Path(config.output_path).resolve()

    if config.report_name:
        report_name = config.report_name
    elif config.use_base_model:
        report_name = f"base_model_{config.split}_fold_{config.fold}.json"
    else:
        report_name = f"{config.split}_predictions_fold_{config.fold}.json"

    base_dir = adapter_path if adapter_path is not None else Path(config.output_root).resolve() / f"fold_{config.fold}"
    return base_dir / report_name


def main() -> None:
    HfArgumentParser, set_seed = _load_transformers_set_seed()
    parser = HfArgumentParser(EvalConfig)
    config = parser.parse_args_into_dataclasses()[0]

    repo_root = Path(config.repo_root).resolve()
    processed_root = Path(config.processed_root).resolve()
    output_root = Path(config.output_root).resolve()
    adapter_path = None
    if not config.use_base_model:
        adapter_path = (
            Path(config.adapter_path).resolve()
            if config.adapter_path
            else output_root / f"fold_{config.fold}"
        )

    set_seed(config.seed)
    processor = load_processor(
        config.model_name_or_path,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    model = load_model_for_inference(
        model_name_or_path=config.model_name_or_path,
        adapter_path=str(adapter_path) if adapter_path else None,
        torch_dtype=config.torch_dtype,
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
    )

    samples = load_fold_samples(config.fold, config.split, repo_root, processed_root)
    output_path = resolve_eval_output_path(config, adapter_path)
    model_label = "base_model" if config.use_base_model else "adapter_model"
    report = evaluate_loaded_model(
        model=model,
        processor=processor,
        samples=samples,
        fold=config.fold,
        split=config.split,
        max_new_tokens=config.max_new_tokens,
        model_label=model_label,
        adapter_path=str(adapter_path) if adapter_path else None,
        output_path=output_path,
    )
    print(
        json.dumps(
            {
                "fold": config.fold,
                "split": config.split,
                "model_label": model_label,
                "exact_match": report["exact_match"],
                "num_samples": report["num_samples"],
                "output_path": str(output_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
