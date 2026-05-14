#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/data/Work/Quark/lora_outputs/stat6209_qwen3vl4b_instruct}"
PYTHON_BIN="${PYTHON_BIN:-/home/i-zhaowei/miniconda3/envs/quark/bin/python}"

"${PYTHON_BIN}" -m stat6209_lora.summarize \
  --output_root "${OUTPUT_ROOT}" \
  "$@"
