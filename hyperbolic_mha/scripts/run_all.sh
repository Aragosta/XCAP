#!/usr/bin/env bash
# Full pipeline: correctness gate, experiment, report.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== correctness gate =="
python -m pytest tests -q

echo "== experiment (preset: ${1:-fast}) =="
python -m hmha.experiment --preset "${1:-fast}"

echo "== report =="
python -m hmha.report

echo "Done. See results/report.md"
