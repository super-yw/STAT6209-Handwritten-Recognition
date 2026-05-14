from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Optional

from .data import DEFAULT_OUTPUT_ROOT


def _load_transformers_parser():
    try:
        from transformers import HfArgumentParser
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "transformers is required for CLI parsing. Install lora_project/requirements.txt first."
        ) from exc
    return HfArgumentParser


@dataclass
class SummaryConfig:
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    base_report_pattern: str = "base_model_eval_fold_{fold}.json"
    adapter_report_pattern: str = "best_model_eval_fold_{fold}.json"
    output_path: Optional[str] = None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_reports(output_root: Path, pattern: str) -> dict:
    reports = []
    for fold in range(1, 6):
        path = output_root / f"fold_{fold}" / pattern.format(fold=fold)
        if not path.exists():
            continue
        report = _read_json(path)
        report["report_path"] = str(path)
        reports.append(report)

    if not reports:
        return {
            "num_folds_found": 0,
            "mean_exact_match": None,
            "folds": [],
        }

    category_values: dict[str, list[float]] = defaultdict(list)
    for report in reports:
        for category, metrics in report.get("by_category", {}).items():
            category_values[category].append(metrics["exact_match"])

    return {
        "num_folds_found": len(reports),
        "mean_exact_match": mean(report["exact_match"] for report in reports),
        "folds": [
            {
                "fold": report["fold"],
                "exact_match": report["exact_match"],
                "num_samples": report["num_samples"],
                "report_path": report["report_path"],
            }
            for report in reports
        ],
        "mean_exact_match_by_category": {
            category: mean(values) for category, values in sorted(category_values.items())
        },
    }


def main() -> None:
    HfArgumentParser = _load_transformers_parser()
    parser = HfArgumentParser(SummaryConfig)
    config = parser.parse_args_into_dataclasses()[0]

    output_root = Path(config.output_root).resolve()
    summary = {
        "output_root": str(output_root),
        "base_model": _summarize_reports(output_root, config.base_report_pattern),
        "adapter_model": _summarize_reports(output_root, config.adapter_report_pattern),
    }

    if (
        summary["base_model"]["mean_exact_match"] is not None
        and summary["adapter_model"]["mean_exact_match"] is not None
    ):
        summary["adapter_minus_base_mean_exact_match"] = (
            summary["adapter_model"]["mean_exact_match"]
            - summary["base_model"]["mean_exact_match"]
        )
    else:
        summary["adapter_minus_base_mean_exact_match"] = None

    if config.output_path:
        output_path = Path(config.output_path).resolve()
    else:
        output_path = output_root / "five_fold_summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary_path": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
