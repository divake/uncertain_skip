#!/usr/bin/env python3
"""
Generate fair comparison plot from MOT17-04 results
All models tracking the same object on the same dataset
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Fair comparison results from MOT17-04 (all models tracking same object)
models_list = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x', 'Adaptive']

# Results from fair comparison on MOT17-04
tracking_rates = [0.9846, 0.9847, 0.9845, 0.9820, 0.9847, 0.9820]
avg_confidences = [0.6828, 0.7496, 0.7206, 0.7206, 0.7677, 0.7416]
frames_tracked = [384, 386, 382, 382, 385, 381]
total_frames = [390, 392, 388, 389, 391, 388]
model_params = [3.2, 11.2, 25.9, 43.7, 68.2, 16.7]  # Adaptive avg: 16.7M
fps_values = [68.4, 70.1, 90.5, 56.0, 75.5, 73.3]

# Create figure with better layout
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Fair Comparison: All Models on MOT17-04 (Same Object)', 
             fontsize=16, fontweight='bold', y=0.98)

# Create subplot grid with proper spacing
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25, top=0.92, bottom=0.08)
axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

# Define colors for each model
colors = ['green', 'yellow', 'orange', 'magenta', 'red', 'blue']

# 1. Tracking Success Rate
ax = axes[0]
bars = ax.bar(models_list, tracking_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Tracking Success Rate', fontsize=11)
ax.set_title('Tracking Success Rate (Fair Comparison)', fontsize=12, fontweight='bold', pad=10)
ax.set_ylim(0.95, 1.0)
ax.grid(True, alpha=0.3, axis='y')

for bar, rate in zip(bars, tracking_rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.001,
            f'{rate:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
# Highlight adaptive result
bars[-1].set_linewidth(3)
bars[-1].set_edgecolor('darkblue')

# 2. Average Confidence
ax = axes[1]
bars = ax.bar(models_list, avg_confidences, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Average Confidence', fontsize=11)
ax.set_title('Average Tracking Confidence', fontsize=12, fontweight='bold', pad=10)
ax.set_ylim(0.6, 0.8)
ax.grid(True, alpha=0.3, axis='y')

for bar, conf in zip(bars, avg_confidences):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.005,
            f'{conf:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

bars[-1].set_linewidth(3)
bars[-1].set_edgecolor('darkblue')

# 3. Frames Tracked (all on MOT17-04)
ax = axes[2]
bar_width = 0.8
x = np.arange(len(models_list))

bars = ax.bar(x, frames_tracked, width=bar_width, color=colors, alpha=0.8, 
              edgecolor='black', linewidth=1.5, label='Tracked')

# Add semi-transparent bars showing frames until loss
for i, (pos, tracked, total) in enumerate(zip(x, frames_tracked, total_frames)):
    ax.bar(pos, total - tracked, width=bar_width, 
           bottom=tracked, color='gray', alpha=0.2)

ax.set_ylabel('Frames', fontsize=11)
ax.set_title('Frames Tracked Until Object Loss', fontsize=12, fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(models_list, rotation=0)
ax.set_ylim(370, 395)
ax.grid(True, alpha=0.3, axis='y')

for i, (bar, frames, total) in enumerate(zip(bars, frames_tracked, total_frames)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2,
            f'{frames}/{total}', ha='center', va='top', fontweight='bold', 
            color='white', fontsize=9)

# Add note about same object
ax.text(2.5, 393, 'All models tracking same object on MOT17-04', 
        ha='center', fontsize=9, 
        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

bars[-1].set_linewidth(3)
bars[-1].set_edgecolor('darkblue')

# 4. Model Efficiency with annotations
ax = axes[3]

# Plot fixed models
for i in range(5):
    ax.scatter(model_params[i], tracking_rates[i], s=150, color=colors[i], 
              alpha=0.7, label=models_list[i], edgecolor='black', linewidth=1.5)

# Plot adaptive with special marker
ax.scatter(model_params[5], tracking_rates[5], s=350, color='blue', 
          marker='*', label='Adaptive', 
          edgecolor='darkblue', linewidth=3, zorder=10)

# Add annotation for adaptive
ax.annotate('Adaptive\n16.7M params\n98.2% success\n16 switches',
            xy=(model_params[5], tracking_rates[5]), 
            xytext=(model_params[5]+8, tracking_rates[5]-0.002),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3',
                          color='darkblue', linewidth=1.5),
            fontsize=9, ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))

ax.set_xlabel('Model Parameters (M)', fontsize=11)
ax.set_ylabel('Tracking Success Rate', fontsize=11)
ax.set_title('Model Efficiency: Parameters vs Performance', fontsize=12, fontweight='bold', pad=10)
ax.legend(loc='lower right', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 75)
ax.set_ylim(0.975, 0.990)

# Add efficiency zone
ax.axvspan(10, 25, alpha=0.1, color='green', label='Efficiency Zone')
ax.text(17.5, 0.9765, 'Optimal\nEfficiency', ha='center', fontsize=8, 
        style='italic', color='darkgreen')

# Save the plot
output_path = Path("results/adaptive/fair_comparison_mot17_04.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
print(f"✓ Fair comparison plot saved to {output_path}")

# Create summary DataFrame
summary_df = pd.DataFrame({
    'Model': models_list,
    'Success Rate': tracking_rates,
    'Confidence': avg_confidences,
    'Frames': [f'{t}/{tot}' for t, tot in zip(frames_tracked, total_frames)],
    'Parameters (M)': model_params,
    'FPS': fps_values
})

print("\nFair Comparison Summary (All models on MOT17-04, same object):")
print(summary_df.to_string(index=False))

# Calculate adaptive efficiency
adaptive_params_saved = (68.2 - 16.7) / 68.2 * 100
print(f"\nAdaptive model efficiency:")
print(f"  - Parameter reduction: {adaptive_params_saved:.1f}% vs YOLOv8x")
print(f"  - Similar tracking success: 98.2% (Adaptive) vs 98.5% (YOLOv8x)")
print(f"  - Dynamic adaptation: 16 model switches")

plt.show()