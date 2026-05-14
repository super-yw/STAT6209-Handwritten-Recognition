from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "training&validate data for lora" / "processed"
DEFAULT_BASE_MODEL = Path("/data/Work/Quark/models/Qwen3-VL-4B-Instruct")
DEFAULT_OUTPUT_ROOT = Path("/data/Work/Quark/lora_outputs/stat6209_qwen3vl4b_instruct")


@dataclass(frozen=True)
class Sample:
    file_name: str
    category: str
    source_no: str
    image_path: Path
    prompt_text: str
    output_text: str


def load_summary(processed_root: Path = DEFAULT_PROCESSED_ROOT) -> dict:
    return json.loads((processed_root / "dataset_summary.json").read_text(encoding="utf-8"))


def normalize_output(text: str) -> str:
    return text.strip()


def build_user_messages(prompt_text: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def build_training_messages(prompt_text: str, output_text: str) -> list[dict]:
    messages = build_user_messages(prompt_text)
    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": normalize_output(output_text)}],
        }
    )
    return messages


def _read_metadata_csv(metadata_path: Path, repo_root: Path, prompt_text: str) -> list[Sample]:
    samples: list[Sample] = []
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = repo_root / row["input_img"]
            samples.append(
                Sample(
                    file_name=row["file_name"],
                    category=row["category"],
                    source_no=row["source_no"],
                    image_path=image_path,
                    prompt_text=prompt_text,
                    output_text=normalize_output(row["output_text"]),
                )
            )
    return samples


def _fallback_samples_from_summary(
    repo_root: Path,
    processed_root: Path,
    fold: int,
    split: str,
) -> list[Sample]:
    summary = load_summary(processed_root)
    prompt_text = summary["text_prompt"]
    complete_metadata = processed_root / "complete_dataset_metadata.csv"
    samples = _read_metadata_csv(complete_metadata, repo_root, prompt_text)
    excluded = {
        category: set(values)
        for category, values in summary["folds"][fold - 1]["excluded_source_nos"].items()
    }
    want_eval = split == "eval"
    return [
        sample
        for sample in samples
        if (sample.source_no in excluded[sample.category]) == want_eval
    ]


def load_fold_samples(
    fold: int,
    split: str,
    repo_root: Path = REPO_ROOT,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
) -> list[Sample]:
    if split not in {"train", "eval"}:
        raise ValueError(f"Unsupported split: {split}")

    summary = load_summary(processed_root)
    prompt_text = summary["text_prompt"]
    metadata_path = processed_root / f"fold_{fold}" / f"{split}_metadata.csv"
    if metadata_path.exists():
        return _read_metadata_csv(metadata_path, repo_root, prompt_text)
    return _fallback_samples_from_summary(repo_root, processed_root, fold, split)

