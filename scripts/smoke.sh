#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${PYTHON:-python3}"
run_root="${1:-runs/cpu-smoke}"
if [ -e "$run_root" ]; then
  echo "Output already exists; choose a new smoke directory: $run_root" >&2
  exit 1
fi
"$python_bin" -m shakewm.cli fixtures --output "$run_root/raw" --seconds 4
"$python_bin" -m shakewm.cli fit-norm --manifest "$run_root/raw/manifest.json" --output "$run_root/norm.json"
"$python_bin" -m shakewm.cli cache --config configs/smoke.json --manifest "$run_root/raw/manifest.json" --output "$run_root/cache" --encoder mock --device cpu
"$python_bin" -m shakewm.cli train --config configs/smoke.json --manifest "$run_root/raw/manifest.json" --normalization "$run_root/norm.json" --cache "$run_root/cache" --output "$run_root/v0" --stop-after 1 --device cpu
"$python_bin" -m shakewm.cli train --config configs/smoke.json --manifest "$run_root/raw/manifest.json" --normalization "$run_root/norm.json" --cache "$run_root/cache" --output "$run_root/v0" --resume "$run_root/v0/latest.pt" --device cpu
for mode in v0 rgb_only v1; do
  config="configs/smoke.json"
  if [ "$mode" != v0 ]; then
    config="configs/smoke_${mode}.json"
    "$python_bin" -m shakewm.cli train --config "$config" --manifest "$run_root/raw/manifest.json" --normalization "$run_root/norm.json" --cache "$run_root/cache" --output "$run_root/$mode" --device cpu
  fi
  "$python_bin" -m shakewm.cli eval --config "$config" --manifest "$run_root/raw/manifest.json" --normalization "$run_root/norm.json" --cache "$run_root/cache" --checkpoint "$run_root/$mode/latest.pt" --output "$run_root/$mode/eval.json" --device cpu
done
"$python_bin" -m shakewm.cli train --config configs/smoke.json --manifest "$run_root/raw/manifest.json" --normalization "$run_root/norm.json" --encoder mock --output "$run_root/online" --stop-after 1 --device cpu
echo "CPU synthetic smoke completed at $run_root. This is not pretrained V-JEPA or GPU acceptance."
