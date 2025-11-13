"""
CVPR Publication-Quality Plot: Clean and Beautiful
- Top 65%: Smooth uncertainty curves (epistemic & aleatoric)
- Bottom 35%: Model selection timeline with colored boxes
- Square format, Times Roman, thick lines, bold labels
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import uniform_filter1d
from src.utils.mot_utils import load_mot_gt

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['xtick.major.width'] = 2.5
plt.rcParams['ytick.major.width'] = 2.5
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['lines.linewidth'] = 4

# Load data
with open('data/mot17_04_uncertainty.json', 'r') as f:
    uncertainty_data = json.load(f)

ground_truth = load_mot_gt("data/MOT17/train/MOT17-04-FRCNN/gt/gt.txt")
ground_truth = np.array(ground_truth)
gt_track = ground_truth[ground_truth[:, 1] == 1]

def calculate_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    box1 = [x1, y1, x1 + w1, y1 + h1]
    box2 = [x2, y2, x2 + w2, y2 + h2]
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

# Match detections to GT
frame_data = []
for frame_idx in range(1050):
    frame_num = frame_idx + 1
    gt_frame = gt_track[gt_track[:, 0] == frame_num]
    if len(gt_frame) == 0:
        continue
    gt_bbox = gt_frame[0, 2:6]
    frame_key = str(frame_num)
    if frame_key not in uncertainty_data['frames']:
        continue
    frame_detections = uncertainty_data['frames'][frame_key]
    best_iou = 0.0
    best_detection = None
    for det in frame_detections:
        iou = calculate_iou(det['bbox'], gt_bbox)
        if iou > best_iou:
            best_iou = iou
            best_detection = det
    if best_detection is None or best_iou < 0.3:
        continue
    frame_data.append({
        'frame': frame_num,
        'confidence': best_detection['confidence'],
        'aleatoric': best_detection['aleatoric'],
        'epistemic': best_detection['epistemic'],
        'iou': best_iou
    })

frame_data = np.array([(d['frame'], d['confidence'], d['aleatoric'], d['epistemic'], d['iou'])
                        for d in frame_data])

frames = frame_data[:, 0]
aleatoric = frame_data[:, 2]
epistemic = frame_data[:, 3]

# Apply temporal smoothing
WINDOW_SIZE = 50
epistemic_smooth = uniform_filter1d(epistemic, size=WINDOW_SIZE, mode='nearest')
aleatoric_smooth = uniform_filter1d(aleatoric, size=WINDOW_SIZE, mode='nearest')

# Define stable epistemic regimes
ep_33 = np.percentile(epistemic_smooth, 33)
ep_67 = np.percentile(epistemic_smooth, 67)

# Assign models with hysteresis
MIN_REGION_LENGTH = 30

def assign_model_cvpr(ep_smooth, al_smooth, ep_33, ep_67):
    assignments = []
    current_model = 2  # Start with medium
    frames_in_current = 0

    for i in range(len(ep_smooth)):
        ep = ep_smooth[i]
        al = al_smooth[i]

        # Determine desired model
        if ep < ep_33:  # Low epistemic
            if al < np.percentile(al_smooth, 50):
                desired = 0  # nano
            else:
                desired = 1  # small
        elif ep < ep_67:  # Medium epistemic
            if al < np.percentile(al_smooth, 33):
                desired = 3  # large
            else:
                desired = 2  # medium
        else:  # High epistemic
            if al < np.percentile(al_smooth, 50):
                desired = 4  # xlarge
            else:
                desired = 3  # large

        # Apply hysteresis
        if desired != current_model:
            if frames_in_current >= MIN_REGION_LENGTH:
                current_model = desired
                frames_in_current = 0

        assignments.append(current_model)
        frames_in_current += 1

    return np.array(assignments)

assignments = assign_model_cvpr(epistemic_smooth, aleatoric_smooth, ep_33, ep_67)

# Identify regions for model transitions
regions = []
current_model = assignments[0]
region_start_idx = 0

for i in range(1, len(assignments)):
    if assignments[i] != current_model:
        regions.append({
            'start_frame': frames[region_start_idx],
            'end_frame': frames[i-1],
            'start_idx': region_start_idx,
            'end_idx': i-1,
            'model': current_model
        })
        current_model = assignments[i]
        region_start_idx = i

# Last region
regions.append({
    'start_frame': frames[region_start_idx],
    'end_frame': frames[-1],
    'start_idx': region_start_idx,
    'end_idx': len(assignments)-1,
    'model': current_model
})

# Model names and colors (same as original plot)
model_names = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x']
model_colors = ['green', 'yellow', 'orange', 'red', 'darkred']

# Create figure with custom height ratios
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(2, 1, height_ratios=[0.65, 0.35], hspace=0.05)

# TOP PLOT: Uncertainty curves (65% of height)
ax1 = fig.add_subplot(gs[0])

# Plot smooth uncertainty lines
ax1.plot(frames, epistemic_smooth, color='#D62728', linewidth=4,
         label='Epistemic', zorder=10)
ax1.plot(frames, aleatoric_smooth, color='#1F77B4', linewidth=4,
         label='Aleatoric', zorder=10)

# Styling
ax1.set_ylabel('Uncertainty', fontsize=26, fontweight='bold')
ax1.tick_params(axis='both', which='major', labelsize=20, width=2.5, length=6)
for label in ax1.get_yticklabels():
    label.set_fontweight('bold')
ax1.grid(True, alpha=0.3, linewidth=1.5, zorder=1)
ax1.set_xlim([frames[0], frames[-1]])
ax1.set_ylim([0, max(epistemic_smooth.max(), aleatoric_smooth.max()) * 1.1])

# Remove x-axis tick labels from top plot (will show only on bottom plot)
ax1.tick_params(labelbottom=False)

# Legend for uncertainty
legend1 = ax1.legend(loc='upper right', frameon=True, fancybox=True,
                     shadow=True, fontsize=16, title='Uncertainty Type',
                     title_fontsize=18)
legend1.get_title().set_fontweight('bold')

# BOTTOM PLOT: Model selection timeline (35% of height)
ax2 = fig.add_subplot(gs[1], sharex=ax1)

# Draw colored rectangles for each model region
for r in regions:
    start_frame = r['start_frame']
    end_frame = r['end_frame']
    width = end_frame - start_frame
    model_idx = int(r['model'])

    rect = Rectangle((start_frame, 0), width, 1,
                     facecolor=model_colors[model_idx],
                     edgecolor='black', linewidth=2, zorder=5)
    ax2.add_patch(rect)

# Styling
ax2.set_xlabel('Frame', fontsize=26, fontweight='bold')
ax2.set_ylabel('Model', fontsize=26, fontweight='bold')
ax2.set_xlim([frames[0], frames[-1]])
ax2.set_ylim([0, 1])
ax2.set_yticks([])  # No y-ticks needed
ax2.tick_params(axis='x', which='major', labelsize=20, width=2.5, length=6)
for label in ax2.get_xticklabels():
    label.set_fontweight('bold')
ax2.grid(True, alpha=0.3, linewidth=1.5, axis='x', zorder=1)

# Create legend for models - position inside plot area
from matplotlib.patches import Patch
model_handles = [
    Patch(facecolor=model_colors[i], edgecolor='black', linewidth=1.5, label=model_names[i])
    for i in range(5)
]
legend2 = ax2.legend(handles=model_handles, loc='lower left', bbox_to_anchor=(0.0, 0.05),
                     frameon=True, fancybox=True, shadow=True,
                     fontsize=14, title='YOLO Model', title_fontsize=15, ncol=5)
legend2.get_title().set_fontweight('bold')

# Tight layout
plt.tight_layout()

# Save as high-quality image
output_path = 'results/rl_evaluation/cvpr_publication_quality.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ Saved publication-quality plot to {output_path}")
print(f"  - Square format with dual panels")
print(f"  - Top panel (65%): Uncertainty curves")
print(f"  - Bottom panel (35%): Model timeline with colored boxes")
print(f"  - Resolution: 300 DPI")
print(f"  - Font: Times Roman (serif)")
print(f"  - Line thickness: 4pt")

plt.close()
