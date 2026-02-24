#!/usr/bin/env python3
"""
Generate a comprehensive summary report comparing standard and robust models.
Reads analysis results and creates markdown summary.
"""

import json
import os

def load_analysis(filepath):
    """Load analysis results from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filepath} not found")
        return None

def format_epsilon(eps):
    """Convert epsilon to readable format."""
    return f"{eps:.4f} ({eps*255:.1f}/255)"

def main():
    print("=" * 80)
    print("COMPREHENSIVE ROBUSTNESS ANALYSIS REPORT")
    print("=" * 80)
    print()
    
    # Load analysis results
    standard_analysis = load_analysis('output/analysis_standard.json')
    robust_analysis = load_analysis('output/analysis_robust.json')
    
    if not standard_analysis:
        print("ERROR: Could not load standard model analysis")
        return
    
    if not robust_analysis:
        print("WARNING: Robust model analysis not yet complete")
        print("Run: python analyze_robustness.py --checkpoint output/models/robust_best.pth --save-results output/analysis_robust.json")
        print()
    
    # Section 1: Clean vs Adversarial Accuracy
    print("## 1. BASELINE COMPARISON")
    print("-" * 80)
    
    std_attack = standard_analysis.get('attack_success_analysis', {})
    print(f"Standard Model (ERM Training):")
    print(f"  Clean Accuracy:      {std_attack.get('clean_accuracy', 0):.2f}%")
    print(f"  Robust Accuracy:     {std_attack.get('adv_accuracy', 0):.2f}% (ε=4/255)")
    print(f"  Attack Success Rate: {std_attack.get('attack_success_rate', 0):.2f}%")
    print()
    
    if robust_analysis:
        rob_attack = robust_analysis.get('attack_success_analysis', {})
        print(f"Robust Model (Adversarial Training):")
        print(f"  Clean Accuracy:      {rob_attack.get('clean_accuracy', 0):.2f}%")
        print(f"  Robust Accuracy:     {rob_attack.get('adv_accuracy', 0):.2f}% (ε=4/255)")
        print(f"  Attack Success Rate: {rob_attack.get('attack_success_rate', 0):.2f}%")
        print()
        
        # Calculate improvements
        clean_diff = rob_attack.get('clean_accuracy', 0) - std_attack.get('clean_accuracy', 0)
        robust_diff = rob_attack.get('adv_accuracy', 0) - std_attack.get('adv_accuracy', 0)
        attack_diff = std_attack.get('attack_success_rate', 0) - rob_attack.get('attack_success_rate', 0)
        
        print(f"Improvement:")
        print(f"  Clean Accuracy:      {clean_diff:+.2f}% ({'↑' if clean_diff > 0 else '↓'})")
        print(f"  Robust Accuracy:     {robust_diff:+.2f}% ({'↑' if robust_diff > 0 else '↓'})")
        print(f"  Attack Success Rate: {attack_diff:+.2f}% reduction ({'↑' if attack_diff > 0 else '↓'})")
    
    print()
    print("=" * 80)
    
    # Section 2: Multi-Epsilon Analysis
    print("## 2. ROBUSTNESS ACROSS DIFFERENT ATTACK STRENGTHS")
    print("-" * 80)
    
    std_eps = standard_analysis.get('epsilon_analysis', {})
    
    print("\nStandard Model:")
    print(f"{'Epsilon':>20} | {'Robust Acc':>12}")
    print("-" * 35)
    for eps, acc in sorted(std_eps.items(), key=lambda x: float(x[0])):
        eps_val = float(eps)
        print(f"{format_epsilon(eps_val):>20} | {acc:>11.2f}%")
    
    if robust_analysis:
        rob_eps = robust_analysis.get('epsilon_analysis', {})
        print("\nRobust Model:")
        print(f"{'Epsilon':>20} | {'Robust Acc':>12}")
        print("-" * 35)
        for eps, acc in sorted(rob_eps.items(), key=lambda x: float(x[0])):
            eps_val = float(eps)
            print(f"{format_epsilon(eps_val):>20} | {acc:>11.2f}%")
        
        # Comparison table
        print("\nSide-by-Side Comparison:")
        print(f"{'Epsilon':>20} | {'Standard':>12} | {'Robust':>12} | {'Improvement':>12}")
        print("-" * 60)
        
        for eps in sorted(std_eps.keys(), key=float):
            eps_val = float(eps)
            std_val = std_eps.get(eps, 0)
            rob_val = rob_eps.get(eps, 0)
            diff = rob_val - std_val
            print(f"{format_epsilon(eps_val):>20} | {std_val:>11.2f}% | {rob_val:>11.2f}% | {diff:>+11.2f}%")
    
    print()
    print("=" * 80)
    
    # Section 3: Key Findings
    print("## 3. KEY FINDINGS")
    print("-" * 80)
    print()
    
    if robust_analysis:
        rob_attack = robust_analysis.get('attack_success_analysis', {})
        
        print("✓ Adversarial Training Effectiveness:")
        robust_acc = rob_attack.get('adv_accuracy', 0)
        std_robust_acc = std_attack.get('adv_accuracy', 0)
        improvement_factor = robust_acc / std_robust_acc if std_robust_acc > 0 else 0
        print(f"  - {improvement_factor:.1f}x improvement in robust accuracy")
        print(f"  - From {std_robust_acc:.2f}% → {robust_acc:.2f}%")
        print()
        
        print("✓ Accuracy-Robustness Trade-off:")
        clean_loss = std_attack.get('clean_accuracy', 0) - rob_attack.get('clean_accuracy', 0)
        robust_gain = robust_acc - std_robust_acc
        if clean_loss > 0:
            print(f"  - Lost {clean_loss:.2f}% clean accuracy")
        else:
            print(f"  - Gained {abs(clean_loss):.2f}% clean accuracy")
        print(f"  - Gained {robust_gain:.2f}% robust accuracy")
        print(f"  - Trade-off ratio: {abs(clean_loss/robust_gain):.2f}:1")
        print()
        
        print("✓ Attack Mitigation:")
        print(f"  - Reduced attack success rate by {attack_diff:.2f}%")
        print(f"  - From {std_attack.get('attack_success_rate', 0):.2f}% → {rob_attack.get('attack_success_rate', 0):.2f}%")
    
    print()
    print("=" * 80)
    print()
    
    # Save to file
    with open('output/SUMMARY_REPORT.txt', 'w') as f:
        # Redirect print to file (simplified - just key metrics)
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE ROBUSTNESS ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("STANDARD MODEL (ERM Training):\n")
        f.write(f"  Clean Accuracy:      {std_attack.get('clean_accuracy', 0):.2f}%\n")
        f.write(f"  Robust Accuracy:     {std_attack.get('adv_accuracy', 0):.2f}% (ε=4/255)\n")
        f.write(f"  Attack Success Rate: {std_attack.get('attack_success_rate', 0):.2f}%\n\n")
        
        if robust_analysis:
            rob_attack = robust_analysis.get('attack_success_analysis', {})
            f.write("ROBUST MODEL (Adversarial Training):\n")
            f.write(f"  Clean Accuracy:      {rob_attack.get('clean_accuracy', 0):.2f}%\n")
            f.write(f"  Robust Accuracy:     {rob_attack.get('adv_accuracy', 0):.2f}% (ε=4/255)\n")
            f.write(f"  Attack Success Rate: {rob_attack.get('attack_success_rate', 0):.2f}%\n\n")
            
            f.write("IMPROVEMENTS:\n")
            f.write(f"  Clean Accuracy:      {clean_diff:+.2f}%\n")
            f.write(f"  Robust Accuracy:     {robust_diff:+.2f}%\n")
            f.write(f"  Attack Success Rate: {attack_diff:+.2f}% reduction\n")
    
    print("Report saved to: output/SUMMARY_REPORT.txt")

if __name__ == '__main__':
    main()
