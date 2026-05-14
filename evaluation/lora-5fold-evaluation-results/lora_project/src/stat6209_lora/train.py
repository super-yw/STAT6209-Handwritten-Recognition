from __future__ import annotations

import gc
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from transformers import TrainerCallback as TransformersTrainerCallback
except ImportError:  # pragma: no cover - runtime dependency
    class TransformersTrainerCallback:  # type: ignore[no-redef]
        pass

from .data import (
    DEFAULT_BASE_MODEL,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROCESSED_ROOT,
    REPO_ROOT,
    Sample,
    build_training_messages,
    load_fold_samples,
)
from .evaluate import evaluate_loaded_model
from .model import load_model_for_inference, load_model_for_training, load_processor


def _load_transformers_objects():
    try:
        from transformers import HfArgumentParser, Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "transformers is required for training. Install lora_project/requirements.txt first."
        ) from exc
    return HfArgumentParser, Trainer, TrainerCallback, TrainingArguments, set_seed


def _build_training_arguments(TrainingArguments, config: "TrainConfig", output_dir: Path):
    args_kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": config.overwrite_output_dir,
        "do_train": True,
        "do_eval": True,
        "use_cpu": not torch.cuda.is_available(),
        "num_train_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "lr_scheduler_type": config.lr_scheduler_type,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_checkpointing": config.gradient_checkpointing,
        "dataloader_num_workers": config.dataloader_num_workers,
        "logging_steps": config.logging_steps,
        "evaluation_strategy": "epoch",
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": config.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "remove_unused_columns": False,
        "report_to": "none",
        "bf16": config.torch_dtype.lower() in {"bfloat16", "bf16"},
        "fp16": config.torch_dtype.lower() in {"float16", "fp16"},
        "label_names": ["labels"],
    }
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    filtered_kwargs = {key: value for key, value in args_kwargs.items() if key in accepted}
    return TrainingArguments(**filtered_kwargs)


def _build_trainer(
    Trainer,
    model,
    training_args,
    train_dataset,
    eval_dataset,
    collator,
    processor,
):
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
        "processing_class": processor,
        "tokenizer": getattr(processor, "tokenizer", processor),
    }
    accepted = set(inspect.signature(Trainer.__init__).parameters)
    filtered_kwargs = {key: value for key, value in trainer_kwargs.items() if key in accepted}
    return Trainer(**filtered_kwargs)


def _epoch_label(epoch_value: Optional[float]) -> str:
    if epoch_value is None:
        return "unknown"
    rounded = round(epoch_value)
    if abs(epoch_value - rounded) < 1e-6:
        return str(int(rounded))
    return f"{epoch_value:.3f}".replace(".", "_")


def _clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class GenerationEvalCallback(TransformersTrainerCallback):
    def __init__(
        self,
        eval_samples: list[Sample],
        processor,
        output_dir: Path,
        fold: int,
        max_new_tokens: int,
    ):
        self.eval_samples = eval_samples
        self.processor = processor
        self.output_dir = output_dir
        self.fold = fold
        self.max_new_tokens = max_new_tokens
        self.history_path = output_dir / "generation_eval_history.jsonl"

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if not state.is_world_process_zero or model is None:
            return control

        epoch_label = _epoch_label(state.epoch)
        report_path = self.output_dir / f"epoch_{epoch_label}_generation_eval.json"
        report = evaluate_loaded_model(
            model=model,
            processor=self.processor,
            samples=self.eval_samples,
            fold=self.fold,
            split="eval",
            max_new_tokens=self.max_new_tokens,
            model_label="adapter_model",
            adapter_path=str(self.output_dir),
            output_path=report_path,
        )
        history_item = {
            "epoch": state.epoch,
            "exact_match": report["exact_match"],
            "num_samples": report["num_samples"],
            "report_path": str(report_path),
            "trainer_metrics": metrics or {},
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(history_item, ensure_ascii=False) + "\n")
        print(
            json.dumps(
                {
                    "epoch": state.epoch,
                    "generation_eval_exact_match": report["exact_match"],
                    "generation_eval_num_samples": report["num_samples"],
                    "generation_eval_report_path": str(report_path),
                },
                ensure_ascii=False,
            )
        )
        return control


@dataclass
class TrainConfig:
    fold: int
    repo_root: str = str(REPO_ROOT)
    processed_root: str = str(DEFAULT_PROCESSED_ROOT)
    model_name_or_path: str = str(DEFAULT_BASE_MODEL)
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    num_train_epochs: float = 5.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    max_length: int = 2048
    generation_eval_max_new_tokens: int = 16
    dataloader_num_workers: int = 0
    logging_steps: int = 10
    save_total_limit: int = 2
    seed: int = 42
    torch_dtype: str = "bfloat16"
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,v_proj"
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    run_base_model_eval: bool = True
    run_epoch_generation_eval: bool = True
    run_final_generation_eval: bool = True
    resume_from_checkpoint: Optional[str] = None
    overwrite_output_dir: bool = False


class SampleDataset(Dataset):
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]


class Qwen3VLSupervisedCollator:
    def __init__(self, processor, max_length: int):
        self.processor = processor
        self.max_length = max_length

    @staticmethod
    def _open_image(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def __call__(self, samples: list[Sample]) -> dict[str, torch.Tensor]:
        images = [self._open_image(sample.image_path) for sample in samples]
        prompt_texts = []
        full_texts = []
        for sample in samples:
            messages = build_training_messages(sample.prompt_text, sample.output_text)
            prompt_messages = messages[:-1]
            prompt_texts.append(
                self.processor.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            full_texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )

        full_batch = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        labels = full_batch["input_ids"].clone()
        labels[full_batch["attention_mask"] == 0] = -100
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, :prompt_length] = -100
        full_batch["labels"] = labels
        return full_batch


def main() -> None:
    HfArgumentParser, Trainer, TrainerCallback, TrainingArguments, set_seed = _load_transformers_objects()
    parser = HfArgumentParser(TrainConfig)
    config = parser.parse_args_into_dataclasses()[0]

    repo_root = Path(config.repo_root).resolve()
    processed_root = Path(config.processed_root).resolve()
    output_dir = Path(config.output_root).resolve() / f"fold_{config.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    processor = load_processor(
        config.model_name_or_path,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    train_samples = load_fold_samples(config.fold, "train", repo_root, processed_root)
    eval_samples = load_fold_samples(config.fold, "eval", repo_root, processed_root)

    if config.run_base_model_eval:
        base_model = load_model_for_inference(
            model_name_or_path=config.model_name_or_path,
            adapter_path=None,
            torch_dtype=config.torch_dtype,
            load_in_4bit=config.load_in_4bit,
            bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
        )
        base_report_path = output_dir / f"base_model_eval_fold_{config.fold}.json"
        base_report = evaluate_loaded_model(
            model=base_model,
            processor=processor,
            samples=eval_samples,
            fold=config.fold,
            split="eval",
            max_new_tokens=config.generation_eval_max_new_tokens,
            model_label="base_model",
            adapter_path=None,
            output_path=base_report_path,
        )
        print(
            json.dumps(
                {
                    "baseline_exact_match": base_report["exact_match"],
                    "baseline_num_samples": base_report["num_samples"],
                    "baseline_report_path": str(base_report_path),
                },
                ensure_ascii=False,
            )
        )
        del base_model
        gc.collect()
        _clear_cuda_cache()

    collator = Qwen3VLSupervisedCollator(processor, max_length=config.max_length)
    model = load_model_for_training(
        model_name_or_path=config.model_name_or_path,
        torch_dtype=config.torch_dtype,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
        gradient_checkpointing=config.gradient_checkpointing,
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
    )

    training_args = _build_training_arguments(TrainingArguments, config, output_dir)

    trainer = _build_trainer(
        Trainer=Trainer,
        model=model,
        training_args=training_args,
        train_dataset=SampleDataset(train_samples),
        eval_dataset=SampleDataset(eval_samples),
        collator=collator,
        processor=processor,
    )

    if config.run_epoch_generation_eval:
        trainer.add_callback(
            GenerationEvalCallback(
                eval_samples=eval_samples,
                processor=processor,
                output_dir=output_dir,
                fold=config.fold,
                max_new_tokens=config.generation_eval_max_new_tokens,
            )
        )

    run_metadata = {
        "fold": config.fold,
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "model_name_or_path": config.model_name_or_path,
        "output_dir": str(output_dir),
        "base_weights_are_read_only": True,
        "run_base_model_eval": config.run_base_model_eval,
        "run_epoch_generation_eval": config.run_epoch_generation_eval,
        "run_final_generation_eval": config.run_final_generation_eval,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model()
    processor.save_pretrained(output_dir)

    if config.run_final_generation_eval:
        final_report_path = output_dir / f"best_model_eval_fold_{config.fold}.json"
        final_report = evaluate_loaded_model(
            model=trainer.model,
            processor=processor,
            samples=eval_samples,
            fold=config.fold,
            split="eval",
            max_new_tokens=config.generation_eval_max_new_tokens,
            model_label="best_adapter_model",
            adapter_path=str(output_dir),
            output_path=final_report_path,
        )
        print(
            json.dumps(
                {
                    "final_exact_match": final_report["exact_match"],
                    "final_num_samples": final_report["num_samples"],
                    "final_report_path": str(final_report_path),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
