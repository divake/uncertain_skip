#!/usr/bin/env python3
"""
Fix the comparison plot layout issue - remove empty space at top
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Load the MOT17-04 tracking results
with open('results/adaptive/tracking_results.json', 'r') as f:
    adaptive_data = json.load(f)

# Extract MOT17-04 adaptive performance
adaptive_frames = adaptive_data['summary']['tracked_frames']  # 373
adaptive_total = adaptive_data['summary']['total_frames']  # 399
adaptive_rate = adaptive_data['summary']['tracking_rate']  # 0.935
adaptive_confidence = adaptive_data['summary']['avg_confidence']  # 0.695
adaptive_params = adaptive_data['summary']['avg_model_params']  # 25.9

# Data for comparison
models_list = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x', 'Adaptive']

# From previous fixed model tests on MOT17-02
tracking_rates = [0.857, 0.968, 0.929, 0.857, 0.854, adaptive_rate]
avg_confidences = [0.468, 0.764, 0.654, 0.538, 0.572, adaptive_confidence]
frames_tracked = [36, 333, 474, 36, 35, adaptive_frames]
model_params = [3.2, 11.2, 25.9, 43.7, 68.2, adaptive_params]

# Create figure with better layout
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Fixed vs Adaptive Model Comparison\n(Adaptive: MOT17-04, 398 frames tracked)', 
             fontsize=16, fontweight='bold', y=0.98)  # Adjusted y position

# Create subplot grid with proper spacing
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25, top=0.92, bottom=0.08)
axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

# Define colors for each model
colors = ['green', 'yellow', 'orange', 'magenta', 'red', 'blue']

# 1. Tracking Success Rate
ax = axes[0]
bars = ax.bar(models_list, tracking_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Tracking Success Rate', fontsize=11)
ax.set_title('Tracking Success Rate Comparison', fontsize=12, fontweight='bold', pad=10)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis='y')

for bar, rate in zip(bars, tracking_rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
            f'{rate:.2%}' if bar.get_x() < 5 else f'{rate:.1%}',
            ha='center', va='bottom', fontweight='bold', fontsize=10)
    
# Highlight adaptive result
bars[-1].set_linewidth(3)
bars[-1].set_edgecolor('darkblue')

# 2. Average Confidence
ax = axes[1]
bars = ax.bar(models_list, avg_confidences, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Average Confidence', fontsize=11)
ax.set_title('Average Tracking Confidence', fontsize=12, fontweight='bold', pad=10)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, axis='y')

for bar, conf in zip(bars, avg_confidences):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
            f'{conf:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

bars[-1].set_linewidth(3)
bars[-1].set_edgecolor('darkblue')

# 3. Frames Tracked with context
ax = axes[2]
bar_width = 0.8
x = np.arange(len(models_list))

# Fixed models tested on 600 frames of MOT17-02, Adaptive on 1050 frames of MOT17-04
max_frames = [600, 600, 600, 600, 600, 1050]
bars = ax.bar(x, frames_tracked, width=bar_width, color=colors, alpha=0.8, 
              edgecolor='black', linewidth=1.5, label='Tracked')

# Add semi-transparent bars showing total frames available
for i, (pos, total) in enumerate(zip(x, max_frames)):
    if i < 5:  # Fixed models
        ax.bar(pos, total - frames_tracked[i], width=bar_width, 
               bottom=frames_tracked[i], color='gray', alpha=0.2)
    else:  # Adaptive
        ax.bar(pos, 399 - frames_tracked[i], width=bar_width,
               bottom=frames_tracked[i], color='gray', alpha=0.2)
        ax.text(pos, 420, 'MOT17-04', 
                ha='center', fontsize=9, style='italic')

ax.set_ylabel('Frames', fontsize=11)
ax.set_title('Total Frames Tracked', fontsize=12, fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(models_list, rotation=0)
ax.set_ylim(0, 500)
ax.grid(True, alpha=0.3, axis='y')

for i, (bar, frames) in enumerate(zip(bars, frames_tracked)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
            f'{frames}', ha='center', va='center', fontweight='bold', 
            color='white' if frames > 100 else 'black', fontsize=10)

# Add dataset labels
ax.text(2.5, 470, 'Fixed: MOT17-02', ha='center', fontsize=9, 
        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
ax.text(5, 200, 'Adaptive:\nMOT17-04\n(398 frames)', 
        ha='center', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

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
ax.annotate('Adaptive\nMOT17-04\n93.5% success\n10 switches',
            xy=(model_params[5], tracking_rates[5]), 
            xytext=(model_params[5]-12, tracking_rates[5]-0.06),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                          color='darkblue', linewidth=1.5),
            fontsize=9, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))

ax.set_xlabel('Model Parameters (M)', fontsize=11)
ax.set_ylabel('Tracking Success Rate', fontsize=11)
ax.set_title('Model Efficiency: Parameters vs Performance', fontsize=12, fontweight='bold', pad=10)
ax.legend(loc='lower right', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 75)
ax.set_ylim(0.65, 1.0)

# Save the fixed plot
output_path = Path("results/adaptive/fixed_vs_adaptive_comparison.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
print(f"✓ Fixed comparison plot saved to {output_path}")

# Show plot info
print(f"Figure size: {fig.get_size_inches()}")
print(f"DPI: 150")
print(f"Output size will be approximately {fig.get_size_inches()[0]*150:.0f} x {fig.get_size_inches()[1]*150:.0f} pixels")

plt.show()