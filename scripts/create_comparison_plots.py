#!/usr/bin/env python3
"""
Create comprehensive comparison visualizations for the final report.
Generates plots comparing standard vs robust models across multiple metrics.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def load_analysis(filepath):
    """Load analysis results from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filepath} not found")
        return None

def plot_epsilon_comparison(standard_data, robust_data, output_path):
    """Plot robustness comparison across different epsilon values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract epsilon data
    std_eps = standard_data.get('epsilon_analysis', {})
    rob_eps = robust_data.get('epsilon_analysis', {}) if robust_data else {}
    
    epsilons = sorted([float(e) for e in std_eps.keys()])
    epsilon_labels = [f'{e*255:.1f}/255' for e in epsilons]
    
    std_accs = [std_eps[str(e)] for e in epsilons]
    rob_accs = [rob_eps.get(str(e), 0) for e in epsilons] if robust_data else [0] * len(epsilons)
    
    # Plot 1: Robustness curves
    x = np.arange(len(epsilons))
    width = 0.35
    
    ax1.bar(x - width/2, std_accs, width, label='Standard Model', alpha=0.8, color='#3498db')
    ax1.bar(x + width/2, rob_accs, width, label='Robust Model', alpha=0.8, color='#e74c3c')
    
    ax1.set_xlabel('Attack Strength (ε)', fontsize=12)
    ax1.set_ylabel('Robust Accuracy (%)', fontsize=12)
    ax1.set_title('Robustness Across Different Attack Strengths', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(epsilon_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, max(max(std_accs), max(rob_accs)) + 10])
    
    # Plot 2: Robustness degradation curve
    ax2.plot(epsilons, std_accs, 'o-', label='Standard Model', linewidth=2, markersize=8, color='#3498db')
    if robust_data:
        ax2.plot(epsilons, rob_accs, 's-', label='Robust Model', linewidth=2, markersize=8, color='#e74c3c')
    
    ax2.set_xlabel('Perturbation Budget (ε)', fontsize=12)
    ax2.set_ylabel('Robust Accuracy (%)', fontsize=12)
    ax2.set_title('Robustness Degradation Curve', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, max(max(std_accs), max(rob_accs)) + 10])
    
    # Add epsilon=4/255 vertical line (training epsilon)
    training_eps = 4/255
    if training_eps in epsilons:
        idx = epsilons.index(training_eps)
        ax2.axvline(x=training_eps, color='gray', linestyle='--', alpha=0.5, label='Training ε')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Epsilon comparison plot saved to: {output_path}")
    plt.close()

def plot_accuracy_comparison(standard_data, robust_data, output_path):
    """Plot clean vs robust accuracy comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract attack success data
    std_attack = standard_data.get('attack_success_analysis', {})
    rob_attack = robust_data.get('attack_success_analysis', {}) if robust_data else {}
    
    models = ['Standard\nModel', 'Robust\nModel']
    clean_accs = [
        std_attack.get('clean_accuracy', 0),
        rob_attack.get('clean_accuracy', 0) if robust_data else 0
    ]
    robust_accs = [
        std_attack.get('adv_accuracy', 0),
        rob_attack.get('adv_accuracy', 0) if robust_data else 0
    ]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, clean_accs, width, label='Clean Accuracy', alpha=0.8, color='#2ecc71')
    bars2 = ax.bar(x + width/2, robust_accs, width, label='Robust Accuracy (ε=4/255)', alpha=0.8, color='#e67e22')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Clean vs Robust Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(max(clean_accs), max(robust_accs)) + 15])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy comparison plot saved to: {output_path}")
    plt.close()

def plot_attack_success_comparison(standard_data, robust_data, output_path):
    """Plot attack success rate comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    std_attack = standard_data.get('attack_success_analysis', {})
    rob_attack = robust_data.get('attack_success_analysis', {}) if robust_data else {}
    
    models = ['Standard\nModel', 'Robust\nModel']
    success_rates = [
        std_attack.get('attack_success_rate', 0),
        rob_attack.get('attack_success_rate', 0) if robust_data else 0
    ]
    
    colors = ['#e74c3c', '#27ae60']
    bars = ax.bar(models, success_rates, alpha=0.8, color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
    ax.set_title('PGD Attack Success Rate (ε=4/255)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    # Add reference line at 50%
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Attack success comparison plot saved to: {output_path}")
    plt.close()

def create_improvement_summary(standard_data, robust_data, output_path):
    """Create a visual summary of improvements."""
    if not robust_data:
        print("Robust model data not available yet")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    std_attack = standard_data.get('attack_success_analysis', {})
    rob_attack = robust_data.get('attack_success_analysis', {})
    
    # Calculate improvements
    clean_diff = rob_attack.get('clean_accuracy', 0) - std_attack.get('clean_accuracy', 0)
    robust_diff = rob_attack.get('adv_accuracy', 0) - std_attack.get('adv_accuracy', 0)
    attack_reduction = std_attack.get('attack_success_rate', 0) - rob_attack.get('attack_success_rate', 0)
    
    # Create summary text
    summary_text = f"""
    ADVERSARIAL TRAINING EFFECTIVENESS SUMMARY
    ═══════════════════════════════════════════════════════════════
    
    STANDARD MODEL (ERM Training)
    ───────────────────────────────────────────────────────────────
    Clean Accuracy:          {std_attack.get('clean_accuracy', 0):.2f}%
    Robust Accuracy:         {std_attack.get('adv_accuracy', 0):.2f}% (ε=4/255)
    Attack Success Rate:     {std_attack.get('attack_success_rate', 0):.2f}%
    
    ROBUST MODEL (Adversarial Training)
    ───────────────────────────────────────────────────────────────
    Clean Accuracy:          {rob_attack.get('clean_accuracy', 0):.2f}%
    Robust Accuracy:         {rob_attack.get('adv_accuracy', 0):.2f}% (ε=4/255)
    Attack Success Rate:     {rob_attack.get('attack_success_rate', 0):.2f}%
    
    IMPROVEMENTS
    ───────────────────────────────────────────────────────────────
    Clean Accuracy:          {clean_diff:+.2f}% {'↑' if clean_diff > 0 else '↓'}
    Robust Accuracy:         {robust_diff:+.2f}% {'↑' if robust_diff > 0 else '↓'}
    Attack Success Rate:     {attack_reduction:+.2f}% reduction {'↑' if attack_reduction > 0 else '↓'}
    
    ROBUSTNESS IMPROVEMENT FACTOR
    ───────────────────────────────────────────────────────────────
    {rob_attack.get('adv_accuracy', 0) / std_attack.get('adv_accuracy', 1):.2f}x improvement in robust accuracy
    """
    
    ax.text(0.5, 0.5, summary_text, fontsize=12, family='monospace',
            ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Improvement summary saved to: {output_path}")
    plt.close()

def main():
    print("=" * 80)
    print("GENERATING COMPREHENSIVE COMPARISON VISUALIZATIONS")
    print("=" * 80)
    print()
    
    # Load data
    standard_data = load_analysis('output/analysis_standard.json')
    robust_data = load_analysis('output/analysis_robust.json')
    
    if not standard_data:
        print("ERROR: Standard model analysis not found!")
        return
    
    # Create output directory
    import os
    os.makedirs('output/plots', exist_ok=True)
    
    # Generate all plots
    print("\n1. Creating epsilon comparison plots...")
    plot_epsilon_comparison(standard_data, robust_data, 'output/plots/epsilon_comparison.png')
    
    print("\n2. Creating accuracy comparison plot...")
    plot_accuracy_comparison(standard_data, robust_data, 'output/plots/accuracy_comparison.png')
    
    print("\n3. Creating attack success comparison plot...")
    plot_attack_success_comparison(standard_data, robust_data, 'output/plots/attack_success_comparison.png')
    
    if robust_data:
        print("\n4. Creating improvement summary...")
        create_improvement_summary(standard_data, robust_data, 'output/plots/improvement_summary.png')
    
    print()
    print("=" * 80)
    print("All visualizations generated successfully!")
    print("Output directory: output/plots/")
    print("=" * 80)

if __name__ == '__main__':
    main()
