#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for fold in 1 2 3 4 5; do
  bash "${SCRIPT_DIR}/run_fold.sh" "${fold}" "$@"
done
