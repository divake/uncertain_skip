"""
Evaluate Trained RL Policy vs Rule-Based Model Selection

This script compares:
1. Rule-based adaptive model selection (threshold-based)
2. Trained RL policy with real orthogonal uncertainty

Both use the same real uncertainty data from MOT17-04.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

from src.core.rl_model_selector import RLModelSelector
from src.utils.mot_utils import load_mot_gt


def calculate_iou(bbox1, bbox2):
    """Calculate IoU between two bboxes [x, y, w, h]"""
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

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def rule_based_selection(confidence, aleatoric, epistemic, current_model,
                         frames_since_switch, cooldown_counter):
    """
    Rule-based model selection (from Phase 1)

    Thresholds:
    - High confidence (>0.85): Use small models
    - Medium confidence (0.50-0.85): Use medium models
    - Low confidence (<0.35): Use large models
    """
    model_names = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']

    # Cooldown check
    if cooldown_counter > 0:
        return current_model, cooldown_counter - 1, frames_since_switch + 1

    # Determine target model based on thresholds
    if confidence > 0.85 and aleatoric < 0.3:
        target_model = 'yolov8n'  # Very easy
    elif confidence > 0.75 and aleatoric < 0.4:
        target_model = 'yolov8s'  # Easy
    elif confidence > 0.60:
        target_model = 'yolov8m'  # Medium
    elif confidence > 0.40 or epistemic < 0.4:
        target_model = 'yolov8l'  # Hard
    else:
        target_model = 'yolov8x'  # Very hard

    # Hysteresis: need 3 consecutive frames before switch
    if target_model != current_model:
        if frames_since_switch < 3:
            return current_model, 0, frames_since_switch + 1
        else:
            # Switch model and set cooldown
            return target_model, 10, 0

    return current_model, 0, frames_since_switch + 1


def evaluate_tracking(uncertainty_data, ground_truth_path, track_id=1,
                     max_frames=1050, method='rule-based', rl_selector=None):
    """
    Evaluate tracking with specified method

    Args:
        uncertainty_data: Precomputed uncertainty JSON
        ground_truth_path: Path to GT file
        track_id: Track to follow
        max_frames: Max frames to process
        method: 'rule-based' or 'rl'
        rl_selector: RLModelSelector instance (if method='rl')
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING: {method.upper()}")
    print(f"{'='*80}")

    # Load ground truth
    ground_truth = load_mot_gt(ground_truth_path)
    ground_truth = np.array(ground_truth)
    gt_track = ground_truth[ground_truth[:, 1] == track_id]

    print(f"Ground truth track {track_id}: {len(gt_track)} frames")

    # Model parameters
    model_params = {
        'yolov8n': 3.2,
        'yolov8s': 11.2,
        'yolov8m': 25.9,
        'yolov8l': 43.7,
        'yolov8x': 68.2
    }

    # Tracking state
    current_model = 'yolov8m'  # Start with medium
    confidence_history = []
    aleatoric_history = []
    epistemic_history = []
    frames_since_switch = 0
    cooldown_counter = 0
    prev_bbox = None

    # Results
    results = {
        'frames_tracked': 0,
        'frames_lost': 0,
        'total_cost': 0.0,
        'model_switches': 0,
        'model_usage': defaultdict(int),
        'avg_confidence': [],
        'avg_iou': [],
        'frame_results': []
    }

    # Process each frame
    for frame_idx in tqdm(range(max_frames), desc=f"{method} tracking"):
        frame_num = frame_idx + 1

        # Get ground truth
        gt_frame = gt_track[gt_track[:, 0] == frame_num]
        if len(gt_frame) == 0:
            continue

        gt_bbox = gt_frame[0, 2:6]

        # Get uncertainty
        frame_key = str(frame_num)
        if frame_key not in uncertainty_data['frames']:
            continue

        frame_detections = uncertainty_data['frames'][frame_key]

        # Find best matching detection
        best_iou = 0.0
        best_detection = None

        for det in frame_detections:
            det_bbox = det['bbox']
            iou = calculate_iou(det_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_detection = det

        if best_detection is None or best_iou < 0.3:
            # Lost track
            results['frames_lost'] += 1
            confidence_history = []
            aleatoric_history = []
            epistemic_history = []
            prev_bbox = None
            continue

        # Extract features
        confidence = best_detection['confidence']
        aleatoric = best_detection['aleatoric']
        epistemic = best_detection['epistemic']

        confidence_history.append(confidence)
        aleatoric_history.append(aleatoric)
        epistemic_history.append(epistemic)

        # Model selection
        prev_model = current_model

        if method == 'rule-based':
            current_model, cooldown_counter, frames_since_switch = rule_based_selection(
                confidence, aleatoric, epistemic, current_model,
                frames_since_switch, cooldown_counter
            )
        else:  # RL
            if len(confidence_history) >= 3:
                # Build state
                object_size = (gt_bbox[2] * gt_bbox[3]) / (1080 * 1920)
                center_x = gt_bbox[0] + gt_bbox[2] / 2
                center_y = gt_bbox[1] + gt_bbox[3] / 2
                edge_dist = min(center_x, center_y, 1920 - center_x, 1080 - center_y) / 1920

                state = rl_selector.extract_state(
                    confidence=confidence,
                    confidence_history=confidence_history,
                    current_model=current_model,
                    iou=best_iou,
                    bbox=gt_bbox,
                    frame_shape=(1080, 1920),
                    aleatoric=aleatoric,
                    aleatoric_history=aleatoric_history,
                    epistemic=epistemic,
                    epistemic_history=epistemic_history
                )

                current_model = rl_selector.select_model(state)

        # Track model switch
        if current_model != prev_model:
            results['model_switches'] += 1

        # Update results
        results['frames_tracked'] += 1
        results['total_cost'] += model_params[current_model]
        results['model_usage'][current_model] += 1
        results['avg_confidence'].append(confidence)
        results['avg_iou'].append(best_iou)

        results['frame_results'].append({
            'frame': frame_num,
            'model': current_model,
            'confidence': confidence,
            'aleatoric': aleatoric,
            'epistemic': epistemic,
            'iou': best_iou,
            'cost': model_params[current_model]
        })

        prev_bbox = gt_bbox

    # Compute summary statistics
    total_frames = results['frames_tracked'] + results['frames_lost']
    results['success_rate'] = results['frames_tracked'] / total_frames if total_frames > 0 else 0
    results['avg_cost'] = results['total_cost'] / results['frames_tracked'] if results['frames_tracked'] > 0 else 0
    results['avg_confidence'] = np.mean(results['avg_confidence']) if results['avg_confidence'] else 0
    results['avg_iou'] = np.mean(results['avg_iou']) if results['avg_iou'] else 0

    # Computational savings vs yolov8x
    savings = (model_params['yolov8x'] - results['avg_cost']) / model_params['yolov8x'] * 100
    results['savings_vs_x'] = savings

    # Computational savings vs yolov8l
    savings_vs_l = (model_params['yolov8l'] - results['avg_cost']) / model_params['yolov8l'] * 100
    results['savings_vs_l'] = savings_vs_l

    return results


def plot_comparison(rule_results, rl_results, output_path):
    """Generate comparison visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('RL vs Rule-Based Adaptive Tracking Comparison', fontsize=16, fontweight='bold')

    # 1. Model usage distribution
    ax = axes[0, 0]
    models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
    rule_usage = [rule_results['model_usage'][m] for m in models]
    rl_usage = [rl_results['model_usage'][m] for m in models]

    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width/2, rule_usage, width, label='Rule-based', alpha=0.8)
    ax.bar(x + width/2, rl_usage, width, label='RL', alpha=0.8)
    ax.set_xlabel('Model')
    ax.set_ylabel('Frames Used')
    ax.set_title('Model Usage Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('yolov8', '') for m in models])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Performance metrics
    ax = axes[0, 1]
    metrics = ['Success Rate', 'Avg Confidence', 'Avg IoU']
    rule_vals = [rule_results['success_rate'], rule_results['avg_confidence'], rule_results['avg_iou']]
    rl_vals = [rl_results['success_rate'], rl_results['avg_confidence'], rl_results['avg_iou']]

    x = np.arange(len(metrics))
    ax.bar(x - width/2, rule_vals, width, label='Rule-based', alpha=0.8)
    ax.bar(x + width/2, rl_vals, width, label='RL', alpha=0.8)
    ax.set_ylabel('Value')
    ax.set_title('Tracking Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    # 3. Computational efficiency
    ax = axes[0, 2]
    metrics = ['Avg Cost\n(M params)', 'Savings vs\nYOLOv8x (%)', 'Model\nSwitches']
    rule_vals = [rule_results['avg_cost'], rule_results['savings_vs_x'], rule_results['model_switches']]
    rl_vals = [rl_results['avg_cost'], rl_results['savings_vs_x'], rl_results['model_switches']]

    # Normalize for visualization
    normalized_rule = [rule_vals[0]/68.2, rule_vals[1]/100, rule_vals[2]/100]
    normalized_rl = [rl_vals[0]/68.2, rl_vals[1]/100, rl_vals[2]/100]

    x = np.arange(len(metrics))
    ax.bar(x - width/2, normalized_rule, width, label='Rule-based', alpha=0.8)
    ax.bar(x + width/2, normalized_rl, width, label='RL', alpha=0.8)
    ax.set_ylabel('Normalized Value')
    ax.set_title('Computational Efficiency')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Model selection over time (first 200 frames)
    ax = axes[1, 0]
    rule_frames = [r['frame'] for r in rule_results['frame_results'][:200]]
    rule_models = [['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'].index(r['model'])
                   for r in rule_results['frame_results'][:200]]
    rl_frames = [r['frame'] for r in rl_results['frame_results'][:200]]
    rl_models = [['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'].index(r['model'])
                 for r in rl_results['frame_results'][:200]]

    ax.plot(rule_frames, rule_models, 'o-', label='Rule-based', alpha=0.6, markersize=2)
    ax.plot(rl_frames, rl_models, 's-', label='RL', alpha=0.6, markersize=2)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Model Size')
    ax.set_title('Model Selection Over Time (First 200 Frames)')
    ax.set_yticks(range(5))
    ax.set_yticklabels(['n', 's', 'm', 'l', 'x'])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Uncertainty vs Model Selection
    ax = axes[1, 1]
    rule_epistemic = [r['epistemic'] for r in rule_results['frame_results'][::5]]
    rule_model_idx = [['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'].index(r['model'])
                      for r in rule_results['frame_results'][::5]]
    rl_epistemic = [r['epistemic'] for r in rl_results['frame_results'][::5]]
    rl_model_idx = [['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'].index(r['model'])
                    for r in rl_results['frame_results'][::5]]

    ax.scatter(rule_epistemic, rule_model_idx, alpha=0.4, s=20, label='Rule-based')
    ax.scatter(rl_epistemic, rl_model_idx, alpha=0.4, s=20, label='RL')
    ax.set_xlabel('Epistemic Uncertainty')
    ax.set_ylabel('Model Size')
    ax.set_title('Epistemic Uncertainty vs Model Selection')
    ax.set_yticks(range(5))
    ax.set_yticklabels(['n', 's', 'm', 'l', 'x'])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Summary table
    ax = axes[1, 2]
    ax.axis('off')

    summary_data = [
        ['Metric', 'Rule-Based', 'RL', 'Δ'],
        ['Success Rate', f"{rule_results['success_rate']:.1%}", f"{rl_results['success_rate']:.1%}",
         f"{(rl_results['success_rate']-rule_results['success_rate'])*100:+.1f}%"],
        ['Avg Confidence', f"{rule_results['avg_confidence']:.3f}", f"{rl_results['avg_confidence']:.3f}",
         f"{rl_results['avg_confidence']-rule_results['avg_confidence']:+.3f}"],
        ['Avg IoU', f"{rule_results['avg_iou']:.3f}", f"{rl_results['avg_iou']:.3f}",
         f"{rl_results['avg_iou']-rule_results['avg_iou']:+.3f}"],
        ['Avg Cost (M)', f"{rule_results['avg_cost']:.1f}", f"{rl_results['avg_cost']:.1f}",
         f"{rl_results['avg_cost']-rule_results['avg_cost']:+.1f}"],
        ['Savings vs X', f"{rule_results['savings_vs_x']:.1f}%", f"{rl_results['savings_vs_x']:.1f}%",
         f"{rl_results['savings_vs_x']-rule_results['savings_vs_x']:+.1f}%"],
        ['Switches', f"{rule_results['model_switches']}", f"{rl_results['model_switches']}",
         f"{rl_results['model_switches']-rule_results['model_switches']:+d}"],
    ]

    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.25, 0.25, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Performance Summary', fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to {output_path}")


def main():
    """Main evaluation"""
    # Paths
    uncertainty_json = "data/mot17_04_uncertainty.json"
    ground_truth_path = "data/MOT17/train/MOT17-04-FRCNN/gt/gt.txt"
    rl_model_path = "results/rl_training/final_model.pt"
    output_dir = Path("results/rl_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load uncertainty data
    print("\n" + "="*80)
    print("LOADING UNCERTAINTY DATA")
    print("="*80)
    with open(uncertainty_json, 'r') as f:
        uncertainty_data = json.load(f)

    print(f"Loaded uncertainty data:")
    print(f"  Sequence: {uncertainty_data['sequence']}")
    print(f"  Total detections: {uncertainty_data['n_detections']}")
    print(f"  Orthogonality: r = {uncertainty_data['statistics']['orthogonality']:.4f}")

    # Initialize RL selector
    print("\n" + "="*80)
    print("LOADING TRAINED RL MODEL")
    print("="*80)
    rl_selector = RLModelSelector(
        pretrained_path=rl_model_path,
        training_mode=False
    )

    # Evaluate rule-based
    rule_results = evaluate_tracking(
        uncertainty_data,
        ground_truth_path,
        track_id=1,
        max_frames=1050,
        method='rule-based',
        rl_selector=None
    )

    # Evaluate RL
    rl_results = evaluate_tracking(
        uncertainty_data,
        ground_truth_path,
        track_id=1,
        max_frames=1050,
        method='rl',
        rl_selector=rl_selector
    )

    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"\n{'Metric':<25} {'Rule-Based':<15} {'RL':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Success Rate':<25} {rule_results['success_rate']:<15.1%} {rl_results['success_rate']:<15.1%} "
          f"{(rl_results['success_rate']-rule_results['success_rate'])*100:+.1f}%")
    print(f"{'Avg Confidence':<25} {rule_results['avg_confidence']:<15.3f} {rl_results['avg_confidence']:<15.3f} "
          f"{rl_results['avg_confidence']-rule_results['avg_confidence']:+.3f}")
    print(f"{'Avg IoU':<25} {rule_results['avg_iou']:<15.3f} {rl_results['avg_iou']:<15.3f} "
          f"{rl_results['avg_iou']-rule_results['avg_iou']:+.3f}")
    print(f"{'Avg Cost (M params)':<25} {rule_results['avg_cost']:<15.1f} {rl_results['avg_cost']:<15.1f} "
          f"{rl_results['avg_cost']-rule_results['avg_cost']:+.1f}")
    print(f"{'Savings vs YOLOv8x':<25} {rule_results['savings_vs_x']:<15.1f}% {rl_results['savings_vs_x']:<15.1f}% "
          f"{rl_results['savings_vs_x']-rule_results['savings_vs_x']:+.1f}%")
    print(f"{'Model Switches':<25} {rule_results['model_switches']:<15} {rl_results['model_switches']:<15} "
          f"{rl_results['model_switches']-rule_results['model_switches']:+d}")

    # Save results
    results_json = {
        'rule_based': {k: v for k, v in rule_results.items() if k != 'frame_results'},
        'rl': {k: v for k, v in rl_results.items() if k != 'frame_results'}
    }

    # Convert defaultdict to dict for JSON serialization
    results_json['rule_based']['model_usage'] = dict(results_json['rule_based']['model_usage'])
    results_json['rl']['model_usage'] = dict(results_json['rl']['model_usage'])

    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n✓ Saved results to {output_dir / 'comparison_results.json'}")

    # Generate visualization
    plot_comparison(rule_results, rl_results, output_dir / 'rl_vs_rulebased_comparison.png')

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
