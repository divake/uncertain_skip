"""
Ablation Study: Uncertainty Decomposition for Model Selection
Compares 4 strategies on MOT17-04 Track 1:
1. Ours (Orthogonal): Epistemic-primary + Aleatoric modifier
2. Total Uncertainty: Sum of aleatoric + epistemic
3. Epistemic-Only: Only epistemic for decisions
4. Aleatoric-Only: Only aleatoric for decisions

Goal: Prove orthogonal decomposition is necessary
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from src.utils.mot_utils import load_mot_gt

# Configuration
WINDOW_SIZE = 50
MIN_REGION_LENGTH = 30
OUTPUT_DIR = Path("results/ablation_study")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ['nano', 'small', 'medium', 'large', 'xlarge']
MODEL_PARAMS = {'nano': 3.2, 'small': 11.2, 'medium': 25.9, 'large': 43.7, 'xlarge': 68.2}


def load_data():
    """Load uncertainty data and ground truth"""
    print("="*80)
    print("LOADING DATA")
    print("="*80)

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

    print(f"Loaded {len(frame_data)} frames for Track 1")
    print(f"Orthogonality r = {uncertainty_data['statistics']['orthogonality']:.4f}")
    print()

    return frames, aleatoric, epistemic


def apply_smoothing(aleatoric, epistemic):
    """Apply temporal smoothing"""
    aleatoric_smooth = uniform_filter1d(aleatoric, size=WINDOW_SIZE, mode='nearest')
    epistemic_smooth = uniform_filter1d(epistemic, size=WINDOW_SIZE, mode='nearest')
    return aleatoric_smooth, epistemic_smooth


def strategy_ours(frames, aleatoric_smooth, epistemic_smooth):
    """
    STRATEGY 1: Ours (Orthogonal)
    Epistemic-primary + Aleatoric modifier
    """
    print("\n" + "="*80)
    print("STRATEGY 1: OURS (ORTHOGONAL)")
    print("="*80)

    ep_33 = np.percentile(epistemic_smooth, 33)
    ep_67 = np.percentile(epistemic_smooth, 67)
    al_50 = np.percentile(aleatoric_smooth, 50)
    al_33 = np.percentile(aleatoric_smooth, 33)

    assignments = []
    current_model = 2
    frames_in_current = 0

    for i in range(len(epistemic_smooth)):
        ep = epistemic_smooth[i]
        al = aleatoric_smooth[i]

        # PRIMARY: Epistemic (model capacity)
        if ep < ep_33:  # Low epistemic
            if al < al_50:
                desired = 0  # nano (easy)
            else:
                desired = 1  # small (noisy but sufficient capacity)
        elif ep < ep_67:  # Medium epistemic
            if al < al_33:
                desired = 3  # large (capacity needed, data clear)
            else:
                desired = 2  # medium (balanced)
        else:  # High epistemic
            if al < al_50:
                desired = 4  # xlarge (capacity critical)
            else:
                desired = 3  # large (don't waste on noisy data)

        # Hysteresis
        if desired != current_model:
            if frames_in_current >= MIN_REGION_LENGTH:
                current_model = desired
                frames_in_current = 0

        assignments.append(current_model)
        frames_in_current += 1

    return analyze_strategy("Ours (Orthogonal)", frames, np.array(assignments),
                           epistemic_smooth, aleatoric_smooth)


def strategy_total_uncertainty(frames, aleatoric_smooth, epistemic_smooth):
    """
    STRATEGY 2: Total Uncertainty
    Sum of aleatoric + epistemic (normalized)
    """
    print("\n" + "="*80)
    print("STRATEGY 2: TOTAL UNCERTAINTY (Aleatoric + Epistemic)")
    print("="*80)

    # Normalize to [0, 1]
    al_norm = (aleatoric_smooth - aleatoric_smooth.min()) / (aleatoric_smooth.max() - aleatoric_smooth.min() + 1e-8)
    ep_norm = (epistemic_smooth - epistemic_smooth.min()) / (epistemic_smooth.max() - epistemic_smooth.min() + 1e-8)

    total_unc = al_norm + ep_norm
    total_smooth = uniform_filter1d(total_unc, size=WINDOW_SIZE, mode='nearest')

    t_33 = np.percentile(total_smooth, 33)
    t_67 = np.percentile(total_smooth, 67)

    print(f"Total uncertainty percentiles: 33rd={t_33:.3f}, 67th={t_67:.3f}")

    assignments = []
    current_model = 2
    frames_in_current = 0

    for i in range(len(total_smooth)):
        t = total_smooth[i]

        # Simple thresholding on total
        if t < t_33:
            desired = 0  # nano/small
        elif t < t_67:
            desired = 2  # medium
        else:
            desired = 4  # large/xlarge

        # Hysteresis
        if desired != current_model:
            if frames_in_current >= MIN_REGION_LENGTH:
                current_model = desired
                frames_in_current = 0

        assignments.append(current_model)
        frames_in_current += 1

    return analyze_strategy("Total Uncertainty", frames, np.array(assignments),
                           epistemic_smooth, aleatoric_smooth)


def strategy_epistemic_only(frames, aleatoric_smooth, epistemic_smooth):
    """
    STRATEGY 3: Epistemic-Only
    Only epistemic uncertainty for decisions
    """
    print("\n" + "="*80)
    print("STRATEGY 3: EPISTEMIC-ONLY")
    print("="*80)

    ep_33 = np.percentile(epistemic_smooth, 33)
    ep_67 = np.percentile(epistemic_smooth, 67)

    print(f"Epistemic percentiles: 33rd={ep_33:.3f}, 67th={ep_67:.3f}")

    assignments = []
    current_model = 2
    frames_in_current = 0

    for i in range(len(epistemic_smooth)):
        ep = epistemic_smooth[i]

        # Only epistemic
        if ep < ep_33:
            desired = 0  # nano
        elif ep < ep_67:
            desired = 2  # medium
        else:
            desired = 4  # xlarge

        # Hysteresis
        if desired != current_model:
            if frames_in_current >= MIN_REGION_LENGTH:
                current_model = desired
                frames_in_current = 0

        assignments.append(current_model)
        frames_in_current += 1

    return analyze_strategy("Epistemic-Only", frames, np.array(assignments),
                           epistemic_smooth, aleatoric_smooth)


def strategy_aleatoric_only(frames, aleatoric_smooth, epistemic_smooth):
    """
    STRATEGY 4: Aleatoric-Only
    Only aleatoric uncertainty for decisions
    """
    print("\n" + "="*80)
    print("STRATEGY 4: ALEATORIC-ONLY")
    print("="*80)

    al_33 = np.percentile(aleatoric_smooth, 33)
    al_67 = np.percentile(aleatoric_smooth, 67)

    print(f"Aleatoric percentiles: 33rd={al_33:.3f}, 67th={al_67:.3f}")

    assignments = []
    current_model = 2
    frames_in_current = 0

    for i in range(len(aleatoric_smooth)):
        al = aleatoric_smooth[i]

        # Only aleatoric
        if al < al_33:
            desired = 0  # nano
        elif al < al_67:
            desired = 2  # medium
        else:
            desired = 4  # xlarge

        # Hysteresis
        if desired != current_model:
            if frames_in_current >= MIN_REGION_LENGTH:
                current_model = desired
                frames_in_current = 0

        assignments.append(current_model)
        frames_in_current += 1

    return analyze_strategy("Aleatoric-Only", frames, np.array(assignments),
                           epistemic_smooth, aleatoric_smooth)


def analyze_strategy(name, frames, assignments, epistemic_smooth, aleatoric_smooth):
    """Analyze a strategy and return results"""
    switches = np.sum(assignments[1:] != assignments[:-1])
    model_counts = np.bincount(assignments, minlength=5)

    total_frames = len(assignments)
    model_distribution = {MODEL_NAMES[i]: int(model_counts[i]) for i in range(5)}
    model_pct = {MODEL_NAMES[i]: model_counts[i] / total_frames * 100 for i in range(5)}

    # Calculate average parameters
    avg_params = sum(MODEL_PARAMS[MODEL_NAMES[i]] * model_counts[i] for i in range(5)) / total_frames
    savings = (MODEL_PARAMS['xlarge'] - avg_params) / MODEL_PARAMS['xlarge'] * 100

    print(f"\nResults for {name}:")
    print(f"  Total frames: {total_frames}")
    print(f"  Switches: {switches} ({switches/total_frames*100:.2f}%)")
    print(f"  Model distribution:")
    for i, model in enumerate(MODEL_NAMES):
        print(f"    {model:8s}: {model_counts[i]:4d} ({model_pct[model]:5.1f}%)")
    print(f"  Avg params: {avg_params:.1f}M")
    print(f"  Savings vs XLarge: {savings:.1f}%")

    # Identify critical failure cases
    identify_failure_cases(name, assignments, epistemic_smooth, aleatoric_smooth, frames)

    return {
        'strategy': name,
        'total_frames': total_frames,
        'total_switches': int(switches),
        'switch_rate_percent': float(switches / total_frames * 100),
        'model_distribution': model_distribution,
        'model_pct': model_pct,
        'avg_params': float(avg_params),
        'savings_percent': float(savings),
        'assignments': assignments.tolist()
    }


def identify_failure_cases(name, assignments, epistemic, aleatoric, frames):
    """Identify specific failure cases for each strategy"""
    print(f"\n  Critical Cases Analysis:")

    # Case 1: High epistemic + Low aleatoric (need capacity, clear data)
    high_ep_low_al = (epistemic > np.percentile(epistemic, 80)) & (aleatoric < np.percentile(aleatoric, 20))
    if np.any(high_ep_low_al):
        models_used = assignments[high_ep_low_al]
        avg_model = np.mean(models_used)
        print(f"    High Ep + Low Al ({np.sum(high_ep_low_al)} frames): Avg model = {avg_model:.1f} (should be 3-4)")
        if avg_model < 3.0:
            print(f"      ⚠️  FAILURE: Using too small models when capacity needed!")

    # Case 2: Low epistemic + High aleatoric (sufficient capacity, noisy data)
    low_ep_high_al = (epistemic < np.percentile(epistemic, 20)) & (aleatoric > np.percentile(aleatoric, 80))
    if np.any(low_ep_high_al):
        models_used = assignments[low_ep_high_al]
        avg_model = np.mean(models_used)
        print(f"    Low Ep + High Al ({np.sum(low_ep_high_al)} frames): Avg model = {avg_model:.1f} (should be 0-1)")
        if avg_model > 2.0:
            print(f"      ⚠️  WASTE: Using large models on noisy data!")

    # Case 3: Both high (hard case)
    both_high = (epistemic > np.percentile(epistemic, 80)) & (aleatoric > np.percentile(aleatoric, 80))
    if np.any(both_high):
        models_used = assignments[both_high]
        avg_model = np.mean(models_used)
        print(f"    High Ep + High Al ({np.sum(both_high)} frames): Avg model = {avg_model:.1f} (should be 2-3)")


def create_comparison_table(results):
    """Create final comparison table"""
    print("\n" + "="*80)
    print("FINAL COMPARISON TABLE")
    print("="*80)

    # Header
    print(f"\n{'Strategy':<25} {'Frames':>8} {'N%':>6} {'S%':>6} {'M%':>6} {'L%':>6} {'X%':>6} {'Switches':>9} {'Params':>8} {'Savings':>8}")
    print("-" * 110)

    # Sort by savings (best first)
    sorted_results = sorted(results, key=lambda x: x['savings_percent'], reverse=True)

    for r in sorted_results:
        pct = r['model_pct']
        print(f"{r['strategy']:<25} {r['total_frames']:>8d} "
              f"{pct['nano']:>6.1f} {pct['small']:>6.1f} {pct['medium']:>6.1f} "
              f"{pct['large']:>6.1f} {pct['xlarge']:>6.1f} "
              f"{r['total_switches']:>9d} {r['avg_params']:>7.1f}M {r['savings_percent']:>7.1f}%")

    # Calculate differences
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    ours = next(r for r in results if 'Ours' in r['strategy'])

    for r in results:
        if r['strategy'] == ours['strategy']:
            continue

        savings_diff = ours['savings_percent'] - r['savings_percent']
        switches_diff = r['total_switches'] - ours['total_switches']

        print(f"\n{r['strategy']}:")
        print(f"  Savings difference: {savings_diff:+.1f}% (ours is better)" if savings_diff > 0
              else f"  Savings difference: {savings_diff:+.1f}% (ours is worse)")
        print(f"  Switch difference: {switches_diff:+d} (more switches)" if switches_diff > 0
              else f"  Switch difference: {switches_diff:+d} (fewer switches)")


def create_visualization(results):
    """Create comprehensive visualization"""
    print("\n[VIZ] Creating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Model distribution comparison
    ax = axes[0, 0]
    width = 0.2
    x = np.arange(5)

    for i, r in enumerate(results):
        pct_values = [r['model_pct'][model] for model in MODEL_NAMES]
        ax.bar(x + i*width, pct_values, width, label=r['strategy'], alpha=0.8)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Usage (%)', fontsize=12)
    ax.set_title('Model Distribution Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels([m[0].upper() for m in MODEL_NAMES])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Switches comparison
    ax = axes[0, 1]
    strategies = [r['strategy'] for r in results]
    switches = [r['total_switches'] for r in results]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))

    bars = ax.bar(strategies, switches, color=colors, alpha=0.8)
    ax.set_ylabel('Number of Switches', fontsize=12)
    ax.set_title('Total Switches Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

    for bar, val in zip(bars, switches):
        ax.text(bar.get_x() + bar.get_width()/2, val, str(val),
                ha='center', va='bottom', fontsize=10)

    # 3. Computational savings
    ax = axes[1, 0]
    savings = [r['savings_percent'] for r in results]
    bars = ax.bar(strategies, savings, color=colors, alpha=0.8)
    ax.set_ylabel('Savings (%)', fontsize=12)
    ax.set_title('Computational Savings vs. Fixed XLarge', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

    for bar, val in zip(bars, savings):
        ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=10)

    # 4. Ranking table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')

    sorted_results = sorted(results, key=lambda x: x['savings_percent'], reverse=True)

    table_data = []
    table_data.append(['Rank', 'Strategy', 'Savings', 'Switches', 'Avg Params'])
    for i, r in enumerate(sorted_results, 1):
        table_data.append([
            f"{i}",
            r['strategy'],
            f"{r['savings_percent']:.1f}%",
            f"{r['total_switches']}",
            f"{r['avg_params']:.1f}M"
        ])

    table = ax.table(cellText=table_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color code by rank
    for i in range(1, len(table_data)):
        if i == 1:  # Best
            table[(i, 0)].set_facecolor('#90EE90')
        elif i == len(table_data) - 1:  # Worst
            table[(i, 0)].set_facecolor('#FFB6C1')

    ax.set_title('Performance Ranking', fontsize=14, fontweight='bold')

    plt.tight_layout()
    viz_path = OUTPUT_DIR / "ablation_comparison.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"[VIZ] Saved to {viz_path}")


def save_results(results):
    """Save all results to JSON"""
    output = {
        'summary': {
            'total_strategies': len(results),
            'sequence': 'MOT17-04-FRCNN',
            'track_id': 1,
            'smoothing_window': WINDOW_SIZE,
            'min_region_length': MIN_REGION_LENGTH
        },
        'strategies': results
    }

    output_file = OUTPUT_DIR / "ablation_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[SAVE] Results saved to {output_file}")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("ABLATION STUDY: UNCERTAINTY DECOMPOSITION FOR MODEL SELECTION")
    print("="*80)
    print(f"Sequence: MOT17-04-FRCNN, Track ID: 1")
    print(f"Smoothing window: {WINDOW_SIZE} frames")
    print(f"Minimum region length: {MIN_REGION_LENGTH} frames")
    print()

    # Load data
    frames, aleatoric, epistemic = load_data()

    # Apply smoothing (common for all strategies)
    aleatoric_smooth, epistemic_smooth = apply_smoothing(aleatoric, epistemic)

    # Run all 4 strategies
    results = []

    results.append(strategy_ours(frames, aleatoric_smooth, epistemic_smooth))
    results.append(strategy_total_uncertainty(frames, aleatoric_smooth, epistemic_smooth))
    results.append(strategy_epistemic_only(frames, aleatoric_smooth, epistemic_smooth))
    results.append(strategy_aleatoric_only(frames, aleatoric_smooth, epistemic_smooth))

    # Create comparison table
    create_comparison_table(results)

    # Create visualization
    create_visualization(results)

    # Save results
    save_results(results)

    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
