#!/usr/bin/env python3

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent


TEXT_PROMPT = dedent(
    """
    The image may contain handwritten numbers and printed numbers.

    Your task:
    1. If there is any handwritten number, return the handwritten number.
    2. If there is no handwritten number, return the most prominent printed number.
    3. If there is a crossed-out mark (e.g., "x", "/", "\\") but no handwritten number, return 0.

    Output requirements:
    - Return a number only.
    - Do not include any explanation.
    - Do not return symbols or expressions (e.g., "x", "/", "\\", "(1+2)").
    """
).strip()

ROOT_DIR = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT_DIR / "cut imgs"
LORA_DIR = ROOT_DIR / "training&validate data for lora"
LABELS_CSV = LORA_DIR / "labels.csv"
KFOLD_JSON = LORA_DIR / "K_fold_no.json"
OUTPUT_DIR = LORA_DIR / "processed"
FILE_NO_RE = re.compile(r"po_(\d{3})")

CATEGORY_PATTERNS = {
    "dn": "marked_dn",
    "dsp": "marked_dsp",
    "inv": "marked_invoice",
}


@dataclass(frozen=True)
class Record:
    file_name: str
    category: str
    source_no: str
    input_img: str
    output_text: str

    def to_training_example(self) -> dict[str, str]:
        return {
            "text_prompt": TEXT_PROMPT,
            "input_img": self.input_img,
            "output_text": self.output_text,
        }


def detect_category(file_name: str) -> str:
    for category, pattern in CATEGORY_PATTERNS.items():
        if pattern in file_name:
            return category
    raise ValueError(f"Cannot detect category from file name: {file_name}")


def extract_source_no(file_name: str) -> str:
    match = FILE_NO_RE.search(file_name)
    if not match:
        raise ValueError(f"Cannot extract source number from file name: {file_name}")
    return match.group(1)


def split_into_five_groups(items: list[str]) -> list[list[str]]:
    base, extra = divmod(len(items), 5)
    groups: list[list[str]] = []
    start = 0
    for index in range(5):
        size = base + (1 if index < extra else 0)
        groups.append(items[start : start + size])
        start += size
    return groups


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_metadata_csv(path: Path, records: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file_name", "category", "source_no", "input_img", "output_text"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "file_name": record.file_name,
                    "category": record.category,
                    "source_no": record.source_no,
                    "input_img": record.input_img,
                    "output_text": record.output_text,
                }
            )


def load_records() -> tuple[list[Record], dict[str, object]]:
    image_names = {path.name for path in IMAGE_DIR.iterdir() if path.is_file()}

    with LABELS_CSV.open(newline="", encoding="utf-8-sig") as handle:
        label_rows = list(csv.DictReader(handle))

    label_names = [row["file name"] for row in label_rows]
    label_name_set = set(label_names)
    image_only = sorted(image_names - label_name_set)
    label_only = sorted(label_name_set - image_names)

    clean_records: list[Record] = []
    for row in label_rows:
        file_name = row["file name"]
        if file_name not in image_names:
            continue
        clean_records.append(
            Record(
                file_name=file_name,
                category=detect_category(file_name),
                source_no=extract_source_no(file_name),
                input_img=str(Path("cut imgs") / file_name),
                output_text=row["value"].strip(),
            )
        )

    clean_records.sort(key=lambda record: record.file_name)
    per_category = Counter(record.category for record in clean_records)

    metadata = {
        "image_count": len(image_names),
        "label_row_count": len(label_rows),
        "clean_record_count": len(clean_records),
        "image_only_count": len(image_only),
        "image_only_files": image_only,
        "label_only_count": len(label_only),
        "label_only_files": label_only,
        "complete_dataset_counts_by_category": dict(sorted(per_category.items())),
    }
    return clean_records, metadata


def build_folds(records: list[Record]) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    kfold_data = json.loads(KFOLD_JSON.read_text(encoding="utf-8"))
    missing_kfold_ids: dict[str, list[str]] = {}
    available_ids = {
        category: {record.source_no for record in records if record.category == category}
        for category in CATEGORY_PATTERNS
    }

    fold_groups = {
        category: split_into_five_groups(kfold_data[category])
        for category in CATEGORY_PATTERNS
    }
    for category, ids in kfold_data.items():
        missing_kfold_ids[category] = sorted(set(ids) - available_ids[category])

    fold_summaries: list[dict[str, object]] = []
    for fold_index in range(5):
        excluded_by_category = {
            category: set(fold_groups[category][fold_index]) for category in CATEGORY_PATTERNS
        }
        train_records = [
            record
            for record in records
            if record.source_no not in excluded_by_category[record.category]
        ]

        eval_records = [
            record
            for record in records
            if record.source_no in excluded_by_category[record.category]
        ]

        fold_dir = OUTPUT_DIR / f"fold_{fold_index + 1}"
        write_jsonl(
            fold_dir / "train.jsonl",
            [record.to_training_example() for record in train_records],
        )
        write_metadata_csv(fold_dir / "train_metadata.csv", train_records)
        write_jsonl(
            fold_dir / "eval.jsonl",
            [record.to_training_example() for record in eval_records],
        )
        write_metadata_csv(fold_dir / "eval_metadata.csv", eval_records)

        eval_counts = Counter(record.category for record in eval_records)
        kept_counts = Counter(record.category for record in train_records)
        fold_summaries.append(
            {
                "fold": fold_index + 1,
                "excluded_source_nos": {
                    category: fold_groups[category][fold_index]
                    for category in CATEGORY_PATTERNS
                },
                "eval_record_count": len(eval_records),
                "eval_record_count_by_category": {
                    category: eval_counts.get(category, 0)
                    for category in CATEGORY_PATTERNS
                },
                "train_record_count": len(train_records),
                "train_record_count_by_category": {
                    category: kept_counts.get(category, 0)
                    for category in CATEGORY_PATTERNS
                },
            }
        )

    return fold_summaries, missing_kfold_ids


def main() -> None:
    records, metadata = load_records()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        OUTPUT_DIR / "complete_dataset.jsonl",
        [record.to_training_example() for record in records],
    )
    write_metadata_csv(OUTPUT_DIR / "complete_dataset_metadata.csv", records)

    fold_summaries, missing_kfold_ids = build_folds(records)
    summary = {
        "text_prompt": TEXT_PROMPT,
        **metadata,
        "kfold_missing_source_nos_in_complete_dataset": missing_kfold_ids,
        "folds": fold_summaries,
    }
    (OUTPUT_DIR / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
