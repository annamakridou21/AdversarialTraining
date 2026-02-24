#!/bin/bash

# Comprehensive Analysis Script for CS-573 Project
# Runs all analysis and visualization tasks

echo "=========================================="
echo "CS-573 Adversarial Training Analysis"
echo "=========================================="
echo ""

# Activate virtual environment
source venv-wsl/bin/activate

# Create output directories
mkdir -p output/visualizations output/plots output/analysis

echo "[1/6] Analyzing Standard Model Robustness..."
python analyze_robustness.py \
    --checkpoint output/models/standard_final.pth \
    --save-results output/analysis/standard_robustness.json

echo ""
echo "[2/6] Analyzing Robust Model Robustness..."
python analyze_robustness.py \
    --checkpoint output/models/robust_best.pth \
    --save-results output/analysis/robust_robustness.json

echo ""
echo "[3/6] Visualizing Adversarial Examples (Standard Model)..."
python visualize_attacks.py \
    --checkpoint output/models/standard_final.pth \
    --num-images 8 \
    --output-name adversarial_examples_standard.png

echo ""
echo "[4/6] Visualizing Adversarial Examples (Robust Model)..."
python visualize_attacks.py \
    --checkpoint output/models/robust_best.pth \
    --num-images 8 \
    --output-name adversarial_examples_robust.png

echo ""
echo "[5/6] Testing Different Attack Strengths on Standard Model..."
python visualize_attacks.py \
    --checkpoint output/models/standard_final.pth \
    --num-images 5 \
    --epsilon 0.00784 \
    --output-name adversarial_weak_attack.png

python visualize_attacks.py \
    --checkpoint output/models/standard_final.pth \
    --num-images 5 \
    --epsilon 0.03137 \
    --output-name adversarial_strong_attack.png

echo ""
echo "[6/6] Creating Summary Report..."
python -c "
import json
import os

print('\n' + '='*60)
print('ANALYSIS SUMMARY')
print('='*60)

# Load standard model analysis
with open('output/analysis/standard_robustness.json', 'r') as f:
    std_data = json.load(f)

# Load robust model analysis  
with open('output/analysis/robust_robustness.json', 'r') as f:
    rob_data = json.load(f)

print('\nSTANDARD MODEL:')
print('-' * 40)
for eps, acc in std_data['epsilon_analysis'].items():
    print(f'  ε = {float(eps):.4f} ({float(eps)*255:4.1f}/255): {acc:5.2f}%')
print(f\"  Attack Success Rate: {std_data['attack_success_analysis']['attack_success_rate']:.2f}%\")

print('\nROBUST MODEL:')
print('-' * 40)
for eps, acc in rob_data['epsilon_analysis'].items():
    print(f'  ε = {float(eps):.4f} ({float(eps)*255:4.1f}/255): {acc:5.2f}%')
print(f\"  Attack Success Rate: {rob_data['attack_success_analysis']['attack_success_rate']:.2f}%\")

print('\n' + '='*60)
print('All analysis complete!')
print('='*60)
print('\nGenerated files:')
print('  - output/analysis/standard_robustness.json')
print('  - output/analysis/robust_robustness.json')
print('  - output/visualizations/adversarial_examples_standard.png')
print('  - output/visualizations/adversarial_examples_robust.png')
print('  - output/visualizations/adversarial_weak_attack.png')
print('  - output/visualizations/adversarial_strong_attack.png')
"

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
