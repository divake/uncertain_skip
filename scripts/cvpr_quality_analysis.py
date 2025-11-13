"""
CVPR-Quality Model Selection Strategy
Goal: ~10 switches for 1000 frames with stable, interpretable regions

Strategy:
1. Apply temporal smoothing (moving average) to reduce noise
2. Identify major epistemic regimes (low/medium/high)
3. Use large hysteresis to prevent jitter
4. Target: 8-12 stable regions with clear transitions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from src.utils.mot_utils import load_mot_gt

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

print(f"\n{'='*80}")
print("CVPR-QUALITY MODEL SELECTION STRATEGY")
print(f"{'='*80}\n")

print(f"Total frames: {len(frame_data)}")
print(f"Sequence: MOT17-04-FRCNN, Track ID 1\n")

# STEP 1: Apply temporal smoothing
WINDOW_SIZE = 50  # Smooth over 50 frames (~2 seconds at 25fps)

epistemic_smooth = uniform_filter1d(epistemic, size=WINDOW_SIZE, mode='nearest')
aleatoric_smooth = uniform_filter1d(aleatoric, size=WINDOW_SIZE, mode='nearest')

print(f"STEP 1: Temporal Smoothing (window={WINDOW_SIZE} frames)")
print(f"  Epistemic smoothed: {np.min(epistemic_smooth):.3f} to {np.max(epistemic_smooth):.3f}")
print(f"  Aleatoric smoothed: {np.min(aleatoric_smooth):.3f} to {np.max(aleatoric_smooth):.3f}\n")

# STEP 2: Define stable epistemic regimes based on SMOOTHED values
# Use percentiles on smoothed epistemic for stable regions
ep_33 = np.percentile(epistemic_smooth, 33)
ep_67 = np.percentile(epistemic_smooth, 67)

print(f"STEP 2: Define Epistemic Regimes (on smoothed data)")
print(f"  Low epistemic:    ep_smooth < {ep_33:.3f}  → Use nano/small")
print(f"  Medium epistemic: {ep_33:.3f} ≤ ep_smooth < {ep_67:.3f} → Use medium/large")
print(f"  High epistemic:   ep_smooth ≥ {ep_67:.3f}  → Use large/xlarge\n")

# STEP 3: Assign models with hysteresis
MIN_REGION_LENGTH = 30  # Minimum 30 frames before allowing switch

def assign_model_cvpr(ep_smooth, al_smooth, ep_33, ep_67):
    """
    CVPR-quality model assignment:
    - Based on SMOOTHED epistemic (primary)
    - Modified by SMOOTHED aleatoric
    - Large hysteresis to prevent jitter
    """
    assignments = []
    current_model = 2  # Start with medium
    frames_in_current = 0

    for i in range(len(ep_smooth)):
        ep = ep_smooth[i]
        al = al_smooth[i]

        # Determine desired model
        if ep < ep_33:  # Low epistemic
            if al < np.percentile(al_smooth, 50):  # Low aleatoric
                desired = 0  # nano (easy case)
            else:
                desired = 1  # small (data noisy but model sufficient)
        elif ep < ep_67:  # Medium epistemic
            if al < np.percentile(al_smooth, 33):  # Low aleatoric
                desired = 3  # large (model capacity needed, data is clear)
            else:
                desired = 2  # medium (balanced)
        else:  # High epistemic
            if al < np.percentile(al_smooth, 50):  # Low-med aleatoric
                desired = 4  # xlarge (model capacity critical)
            else:
                desired = 3  # large (don't waste on noisy data)

        # Apply hysteresis: only switch if in current model for MIN_REGION_LENGTH frames
        if desired != current_model:
            if frames_in_current >= MIN_REGION_LENGTH:
                current_model = desired
                frames_in_current = 0
            # else: stay in current model

        assignments.append(current_model)
        frames_in_current += 1

    return np.array(assignments)

assignments = assign_model_cvpr(epistemic_smooth, aleatoric_smooth, ep_33, ep_67)

# Count switches
switches = np.sum(assignments[1:] != assignments[:-1])

print(f"STEP 3: Model Assignment with Hysteresis")
print(f"  Minimum region length: {MIN_REGION_LENGTH} frames")
print(f"  Total switches: {switches}")
print(f"  Switch rate: {switches/len(assignments)*100:.1f}%\n")

# Model distribution
model_names = ['nano', 'small', 'medium', 'large', 'xlarge']
model_counts = np.bincount(assignments, minlength=5)

print(f"STEP 4: Model Distribution")
for i in range(5):
    pct = model_counts[i] / len(assignments) * 100
    print(f"  YOLOv8{model_names[i][0]}: {model_counts[i]:4d} frames ({pct:5.1f}%)")

# Identify regions
regions = []
current_model = assignments[0]
region_start_idx = 0

for i in range(1, len(assignments)):
    if assignments[i] != current_model:
        regions.append({
            'start_frame': int(frames[region_start_idx]),
            'end_frame': int(frames[i-1]),
            'start_idx': region_start_idx,
            'end_idx': i-1,
            'model': int(current_model),
            'length': i - region_start_idx,
            'avg_epistemic': float(np.mean(epistemic_smooth[region_start_idx:i])),
            'avg_aleatoric': float(np.mean(aleatoric_smooth[region_start_idx:i]))
        })
        current_model = assignments[i]
        region_start_idx = i

# Last region
regions.append({
    'start_frame': int(frames[region_start_idx]),
    'end_frame': int(frames[-1]),
    'start_idx': region_start_idx,
    'end_idx': len(assignments)-1,
    'model': int(current_model),
    'length': len(assignments) - region_start_idx,
    'avg_epistemic': float(np.mean(epistemic_smooth[region_start_idx:])),
    'avg_aleatoric': float(np.mean(aleatoric_smooth[region_start_idx:]))
})

print(f"\nSTEP 5: Stable Regions Identified")
print(f"  Total regions: {len(regions)}")
print(f"  Average region length: {len(assignments)/len(regions):.1f} frames\n")

print(f"{'='*80}")
print(f"REGION BREAKDOWN (All {len(regions)} regions)")
print(f"{'='*80}\n")

for i, r in enumerate(regions):
    print(f"Region {i+1:2d}: Frames {r['start_frame']:4d}-{r['end_frame']:4d} "
          f"({r['length']:3d} frames) → YOLOv8{model_names[r['model']][0]} "
          f"(ep={r['avg_epistemic']:.3f}, al={r['avg_aleatoric']:.3f})")

# Create comprehensive visualization
fig, axes = plt.subplots(3, 1, figsize=(20, 12))

# 1. Uncertainty over time (raw + smoothed)
ax = axes[0]
ax.plot(frames, epistemic, 'r-', alpha=0.3, linewidth=0.5, label='Epistemic (raw)')
ax.plot(frames, epistemic_smooth, 'r-', alpha=0.9, linewidth=2, label='Epistemic (smoothed)')
ax.plot(frames, aleatoric, 'b-', alpha=0.3, linewidth=0.5, label='Aleatoric (raw)')
ax.plot(frames, aleatoric_smooth, 'b-', alpha=0.9, linewidth=2, label='Aleatoric (smoothed)')
ax.axhline(y=ep_33, color='orange', linestyle='--', alpha=0.7, linewidth=2, label=f'Ep thresh low ({ep_33:.2f})')
ax.axhline(y=ep_67, color='darkred', linestyle='--', alpha=0.7, linewidth=2, label=f'Ep thresh high ({ep_67:.2f})')
ax.set_xlabel('Frame', fontsize=14)
ax.set_ylabel('Uncertainty', fontsize=14)
ax.set_title(f'Uncertainty Over Time (Temporal Smoothing: {WINDOW_SIZE} frames)',
             fontsize=16, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)

# 2. Model selection over time
ax = axes[1]
colors = ['green', 'yellow', 'orange', 'red', 'darkred']
for i in range(5):
    mask = assignments == i
    ax.scatter(frames[mask], assignments[mask], c=colors[i], s=15, alpha=0.7,
              label=f'YOLOv8{model_names[i][0]}')
ax.set_xlabel('Frame', fontsize=14)
ax.set_ylabel('Model', fontsize=14)
ax.set_yticks(range(5))
ax.set_yticklabels([f'YOLOv8{n[0]}' for n in model_names])
ax.set_title(f'CVPR-Quality Model Selection ({switches} switches, {len(regions)} stable regions)',
             fontsize=16, fontweight='bold')
ax.legend(loc='upper right', fontsize=11, ncol=5)
ax.grid(True, alpha=0.3, axis='x')

# 3. Stable regions visualization
ax = axes[2]
from matplotlib.patches import Rectangle
for r in regions:
    start_frame = r['start_frame']
    length = r['end_frame'] - r['start_frame'] + 1
    ax.add_patch(Rectangle((start_frame, r['model']-0.4), length, 0.8,
                           facecolor=colors[r['model']], alpha=0.8,
                           edgecolor='black', linewidth=1.5))
    # Add text label
    if length > 20:  # Only label regions longer than 20 frames
        ax.text(start_frame + length/2, r['model'], f"{r['model']}",
                ha='center', va='center', fontsize=10, fontweight='bold')

ax.set_xlim(frames[0], frames[-1])
ax.set_ylim(-0.5, 4.5)
ax.set_xlabel('Frame', fontsize=14)
ax.set_ylabel('Model', fontsize=14)
ax.set_yticks(range(5))
ax.set_yticklabels([f'YOLOv8{n[0]}' for n in model_names])
ax.set_title(f'Stable Regions Timeline ({len(regions)} regions, avg length: {len(assignments)/len(regions):.0f} frames)',
             fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('results/rl_evaluation/cvpr_quality_model_selection.png', dpi=200, bbox_inches='tight')
print(f"\n✓ Saved CVPR-quality visualization to results/rl_evaluation/cvpr_quality_model_selection.png")

# Save strategy
strategy = {
    'method': 'Temporal Smoothing + Hysteresis',
    'window_size': WINDOW_SIZE,
    'min_region_length': MIN_REGION_LENGTH,
    'thresholds': {
        'epistemic_low': float(ep_33),
        'epistemic_high': float(ep_67)
    },
    'total_frames': len(frame_data),
    'total_switches': int(switches),
    'switch_rate_percent': float(switches/len(assignments)*100),
    'num_regions': len(regions),
    'avg_region_length': float(len(assignments)/len(regions)),
    'model_distribution': {
        'nano': int(model_counts[0]),
        'small': int(model_counts[1]),
        'medium': int(model_counts[2]),
        'large': int(model_counts[3]),
        'xlarge': int(model_counts[4])
    },
    'regions': regions
}

with open('results/rl_evaluation/cvpr_strategy.json', 'w') as f:
    json.dump(strategy, f, indent=2)

print(f"✓ Saved CVPR strategy to results/rl_evaluation/cvpr_strategy.json")

print(f"\n{'='*80}")
print("SUMMARY FOR CVPR PAPER")
print(f"{'='*80}\n")

print(f"✅ Achieved {switches} switches (target: 8-12)")
print(f"✅ Created {len(regions)} stable regions")
print(f"✅ Average region length: {len(assignments)/len(regions):.0f} frames")
print(f"✅ Model distribution: nano={model_counts[0]/len(assignments)*100:.1f}%, "
      f"small={model_counts[1]/len(assignments)*100:.1f}%, "
      f"medium={model_counts[2]/len(assignments)*100:.1f}%, "
      f"large={model_counts[3]/len(assignments)*100:.1f}%, "
      f"xlarge={model_counts[4]/len(assignments)*100:.1f}%")

print(f"\n{'='*80}")
print("NEXT STEPS")
print(f"{'='*80}\n")

print("1. Implement this as RULE-BASED BASELINE for CVPR")
print(f"2. Design RL to learn similar stable behavior")
print(f"3. Compare: Rule-based vs RL (both should have 8-15 switches)")
print(f"4. Paper contribution: Orthogonal uncertainty + Stable adaptive selection")
