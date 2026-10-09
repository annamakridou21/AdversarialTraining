#!/usr/bin/env bash
# Run from the repository root with the intended Python environment active.
set -euo pipefail
PYTHON="${PYTHON:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
DATA_DIR="${DATA_DIR:-data}"
for model in standard robust; do
    "$PYTHON" -m scripts.analyze_robustness \
        --checkpoint "$OUTPUT_DIR/models/${model}_best.pth" --data-dir "$DATA_DIR" \
        --save-results "$OUTPUT_DIR/analysis/${model}_robustness.json" "$@"
    "$PYTHON" -m scripts.visualize_attacks \
        --checkpoint "$OUTPUT_DIR/models/${model}_best.pth" --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR/visualizations" --output-name "${model}.png" "$@"
done
"$PYTHON" -m scripts.create_comparison_plots \
    --standard "$OUTPUT_DIR/analysis/standard_robustness.json" \
    --robust "$OUTPUT_DIR/analysis/robust_robustness.json" \
    --output "$OUTPUT_DIR/plots/robustness_comparison.png"
"$PYTHON" -m scripts.create_summary_report \
    --standard "$OUTPUT_DIR/analysis/standard_robustness.json" \
    --robust "$OUTPUT_DIR/analysis/robust_robustness.json" \
    --output "$OUTPUT_DIR/analysis/summary.md"
