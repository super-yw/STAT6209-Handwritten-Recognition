#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-/data/Work/Quark/models/Qwen3-VL-4B-Instruct}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${REPO_ROOT}/training&validate data for lora/processed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/Work/Quark/lora_outputs/stat6209_qwen3vl4b_instruct}"
PYTHON_BIN="${PYTHON_BIN:-/home/i-zhaowei/miniconda3/envs/quark/bin/python}"

for fold in 1 2 3 4 5; do
  "${PYTHON_BIN}" -m stat6209_lora.evaluate \
    --fold "${fold}" \
    --repo_root "${REPO_ROOT}" \
    --processed_root "${PROCESSED_ROOT}" \
    --model_name_or_path "${MODEL_PATH}" \
    --output_root "${OUTPUT_ROOT}" \
    --use_base_model true \
    "$@"
done
