"""
Plot attention metrics across layers 3-12
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_metrics():
    """Generate plots for attention metrics"""

    # Load results
    results_file = Path("results/experiment_02_attention_metrics/attention_metrics.json")

    if not results_file.exists():
        print(f"Error: {results_file} not found. Run experiment first.")
        return

    with open(results_file, 'r') as f:
        results = json.load(f)

    # Extract data
    layers = list(range(3, 13))

    in_box_means = [results[f"layer_{l}"]['in_box_ratio_mean'] for l in layers]
    in_box_stds = [results[f"layer_{l}"]['in_box_ratio_std'] for l in layers]

    entropy_means = [results[f"layer_{l}"]['entropy_mean'] for l in layers]
    entropy_stds = [results[f"layer_{l}"]['entropy_std'] for l in layers]

    concentration_means = [results[f"layer_{l}"]['concentration_mean'] for l in layers]
    concentration_stds = [results[f"layer_{l}"]['concentration_std'] for l in layers]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: In-Box Ratio
    axes[0].errorbar(layers, in_box_means, yerr=in_box_stds,
                     marker='o', linewidth=2, markersize=8, capsize=5)
    axes[0].axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='Viability threshold (0.3)')
    axes[0].set_xlabel('Layer', fontsize=12)
    axes[0].set_ylabel('In-Box Attention Ratio', fontsize=12)
    axes[0].set_title('In-Box Ratio (Higher = Better)\n% of attention inside GT boxes', fontsize=13, fontweight='bold')
    axes[0].set_xticks(layers)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim(0, 1.0)

    # Plot 2: Entropy
    axes[1].errorbar(layers, entropy_means, yerr=entropy_stds,
                     marker='o', linewidth=2, markersize=8, capsize=5, color='orange')
    axes[1].set_xlabel('Layer', fontsize=12)
    axes[1].set_ylabel('Attention Entropy', fontsize=12)
    axes[1].set_title('Attention Entropy (Lower = Better)\nMeasure of attention focus', fontsize=13, fontweight='bold')
    axes[1].set_xticks(layers)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Concentration
    axes[2].errorbar(layers, concentration_means, yerr=concentration_stds,
                     marker='o', linewidth=2, markersize=8, capsize=5, color='green')
    axes[2].set_xlabel('Layer', fontsize=12)
    axes[2].set_ylabel('Attention Concentration (Top-10%)', fontsize=12)
    axes[2].set_title('Attention Concentration (Higher = Better)\n% of attention in top-10% patches', fontsize=13, fontweight='bold')
    axes[2].set_xticks(layers)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1.0)

    plt.tight_layout()

    # Save plot
    output_path = Path("results/experiment_02_attention_metrics/attention_metrics_plot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")

    # Create combined plot
    fig2, ax = plt.subplots(figsize=(12, 6))

    # Normalize metrics to 0-1 for comparison
    in_box_norm = np.array(in_box_means)  # Already 0-1
    entropy_norm = 1 - (np.array(entropy_means) / np.max(entropy_means))  # Invert and normalize
    concentration_norm = np.array(concentration_means)  # Already 0-1

    ax.plot(layers, in_box_norm, marker='o', linewidth=2, markersize=8, label='In-Box Ratio')
    ax.plot(layers, entropy_norm, marker='s', linewidth=2, markersize=8, label='Focus (1 - norm. entropy)')
    ax.plot(layers, concentration_norm, marker='^', linewidth=2, markersize=8, label='Concentration')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Normalized Score (Higher = Better)', fontsize=12)
    ax.set_title('Combined Attention Quality Metrics Across Layers', fontsize=14, fontweight='bold')
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    output_path2 = Path("results/experiment_02_attention_metrics/attention_metrics_combined.png")
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✓ Combined plot saved to: {output_path2}")

    print("\nPlots generated successfully!")


if __name__ == "__main__":
    plot_metrics()
