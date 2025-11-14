"""
Evaluate trained RL policy on MOT17-04 and compare with rule-based method
"""

import json
import numpy as np
import torch
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

# Import from training script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_rl_mot17_04 import (
    DQN, TrackingEnvironment, MODEL_NAMES, MODEL_PARAMS,
    UNCERTAINTY_FILE, GT_FILE
)

# Configuration
OUTPUT_DIR = Path("/ssd_4TB/divake/uncertain_skip/results/rl_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = "/ssd_4TB/divake/uncertain_skip/results/rl_training/dqn_final.pt"
RULE_BASED_RESULTS = "/ssd_4TB/divake/uncertain_skip/results/rl_evaluation/cvpr_strategy.json"


def evaluate_rl_policy(env, model_path):
    """Evaluate trained RL policy"""
    print("\n" + "="*70)
    print(" EVALUATING RL POLICY")
    print("="*70)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = DQN().to(device)

    checkpoint = torch.load(model_path)
    q_network.load_state_dict(checkpoint['q_network'])
    q_network.eval()

    print(f"[EVAL] Loaded model from {model_path}")
    print(f"[EVAL] Trained for {checkpoint['episode']} episodes")

    # Run evaluation
    state, _ = env.reset()
    episode_reward = 0
    frame_results = []
    model_counts = {name: 0 for name in MODEL_NAMES}
    switches = 0
    prev_model = None

    while True:
        # Select action (greedy)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = q_network(state_tensor)
            action = q_values.argmax().item()

        model_name = MODEL_NAMES[action]
        model_counts[model_name] += 1

        if prev_model is not None and prev_model != action:
            switches += 1

        # Take step
        next_state, reward, done, info = env.step(action)

        if next_state is None:
            break

        episode_reward += reward
        frame_results.append({
            'frame': info['frame'],
            'model': model_name,
            'model_idx': action,
            'epistemic': info['epistemic'],
            'aleatoric': info['aleatoric'],
            'iou': info['iou'],
            'reward': reward
        })

        state = next_state
        prev_model = action

        if done:
            break

    # Compute statistics
    total_frames = len(frame_results)
    model_distribution = {name: (count / total_frames * 100) for name, count in model_counts.items()}

    # Compute average parameters
    avg_params = sum(MODEL_PARAMS[name] * count for name, count in model_counts.items()) / total_frames
    savings = (MODEL_PARAMS['yolov8x'] - avg_params) / MODEL_PARAMS['yolov8x'] * 100

    results = {
        'method': 'RL (Double DQN)',
        'total_frames': total_frames,
        'total_switches': switches,
        'switch_rate_percent': (switches / total_frames * 100) if total_frames > 0 else 0,
        'model_distribution': model_distribution,
        'model_counts': model_counts,
        'avg_params': avg_params,
        'savings_percent': savings,
        'total_reward': episode_reward,
        'avg_reward_per_frame': episode_reward / total_frames if total_frames > 0 else 0,
        'frame_results': frame_results
    }

    print(f"\n[EVAL] RL Policy Results:")
    print(f"  Total frames: {total_frames}")
    print(f"  Switches: {switches} ({results['switch_rate_percent']:.2f}%)")
    print(f"  Model distribution:")
    for name in MODEL_NAMES:
        print(f"    {name:10s}: {model_distribution[name]:5.1f}%")
    print(f"  Avg params: {avg_params:.1f}M")
    print(f"  Savings: {savings:.1f}%")
    print(f"  Total reward: {episode_reward:.2f}")

    return results


def load_rule_based_results(path):
    """Load rule-based method results"""
    with open(path, 'r') as f:
        data = json.load(f)

    # Convert to same format
    total_frames = data['total_frames']
    model_dist = data['model_distribution']

    model_distribution = {
        'yolov8n': model_dist['nano'] / total_frames * 100,
        'yolov8s': model_dist['small'] / total_frames * 100,
        'yolov8m': model_dist['medium'] / total_frames * 100,
        'yolov8l': model_dist['large'] / total_frames * 100,
        'yolov8x': model_dist['xlarge'] / total_frames * 100
    }

    avg_params = sum(MODEL_PARAMS[name] * model_dist[key] for name, key in [
        ('yolov8n', 'nano'), ('yolov8s', 'small'), ('yolov8m', 'medium'),
        ('yolov8l', 'large'), ('yolov8x', 'xlarge')
    ]) / total_frames

    savings = (MODEL_PARAMS['yolov8x'] - avg_params) / MODEL_PARAMS['yolov8x'] * 100

    return {
        'method': 'Rule-Based (CVPR)',
        'total_frames': total_frames,
        'total_switches': data['total_switches'],
        'switch_rate_percent': data['switch_rate_percent'],
        'model_distribution': model_distribution,
        'avg_params': avg_params,
        'savings_percent': savings
    }


def create_comparison_table(rl_results, rule_based_results):
    """Create detailed comparison table"""
    print("\n" + "="*70)
    print(" COMPARISON: RL vs. RULE-BASED")
    print("="*70)

    # Table header
    print(f"\n{'Metric':<25} {'Rule-Based':>15} {'RL Policy':>15} {'Difference':>15}")
    print("-" * 70)

    # Frames
    print(f"{'Total Frames':<25} {rule_based_results['total_frames']:>15d} "
          f"{rl_results['total_frames']:>15d} "
          f"{rl_results['total_frames'] - rule_based_results['total_frames']:>15d}")

    # Switches
    print(f"{'Switches':<25} {rule_based_results['total_switches']:>15d} "
          f"{rl_results['total_switches']:>15d} "
          f"{rl_results['total_switches'] - rule_based_results['total_switches']:>15d}")

    # Switch rate
    print(f"{'Switch Rate (%)':<25} {rule_based_results['switch_rate_percent']:>15.2f} "
          f"{rl_results['switch_rate_percent']:>15.2f} "
          f"{rl_results['switch_rate_percent'] - rule_based_results['switch_rate_percent']:>15.2f}")

    # Model distribution
    print(f"\n{'Model Distribution (%)'}:")
    for name in MODEL_NAMES:
        rb_val = rule_based_results['model_distribution'][name]
        rl_val = rl_results['model_distribution'][name]
        diff = rl_val - rb_val
        print(f"  {name:<23} {rb_val:>15.1f} {rl_val:>15.1f} {diff:>15.1f}")

    # Average parameters
    print(f"\n{'Avg Parameters (M)':<25} {rule_based_results['avg_params']:>15.1f} "
          f"{rl_results['avg_params']:>15.1f} "
          f"{rl_results['avg_params'] - rule_based_results['avg_params']:>15.1f}")

    # Computational savings
    print(f"{'Savings vs XLarge (%)':<25} {rule_based_results['savings_percent']:>15.1f} "
          f"{rl_results['savings_percent']:>15.1f} "
          f"{rl_results['savings_percent'] - rule_based_results['savings_percent']:>15.1f}")

    # RL-specific metrics
    if 'total_reward' in rl_results:
        print(f"\n{'RL Total Reward':<25} {'-':>15} {rl_results['total_reward']:>15.2f} {'-':>15}")
        print(f"{'RL Avg Reward/Frame':<25} {'-':>15} {rl_results['avg_reward_per_frame']:>15.4f} {'-':>15}")


def create_visualization(rl_results, rule_based_results, output_dir):
    """Create comparison visualizations"""
    print("\n[VIZ] Creating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Model distribution comparison
    ax = axes[0, 0]
    x = np.arange(len(MODEL_NAMES))
    width = 0.35

    rb_vals = [rule_based_results['model_distribution'][name] for name in MODEL_NAMES]
    rl_vals = [rl_results['model_distribution'][name] for name in MODEL_NAMES]

    ax.bar(x - width/2, rb_vals, width, label='Rule-Based', alpha=0.8)
    ax.bar(x + width/2, rl_vals, width, label='RL Policy', alpha=0.8)
    ax.set_xlabel('Model')
    ax.set_ylabel('Usage (%)')
    ax.set_title('Model Distribution Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace('yolov8', '') for name in MODEL_NAMES])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Switches comparison
    ax = axes[0, 1]
    methods = ['Rule-Based', 'RL Policy']
    switches = [rule_based_results['total_switches'], rl_results['total_switches']]
    colors = ['#1f77b4', '#ff7f0e']
    ax.bar(methods, switches, color=colors, alpha=0.8)
    ax.set_ylabel('Number of Switches')
    ax.set_title('Total Switches Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for i, v in enumerate(switches):
        ax.text(i, v, str(v), ha='center', va='bottom')

    # 3. Computational savings
    ax = axes[1, 0]
    savings = [rule_based_results['savings_percent'], rl_results['savings_percent']]
    ax.bar(methods, savings, color=colors, alpha=0.8)
    ax.set_ylabel('Savings (%)')
    ax.set_title('Computational Savings vs. Fixed XLarge')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])

    # Add values on bars
    for i, v in enumerate(savings):
        ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

    # 4. Model selection timeline (RL only, first 200 frames)
    if 'frame_results' in rl_results:
        ax = axes[1, 1]
        frames = [r['frame'] for r in rl_results['frame_results'][:200]]
        models = [r['model_idx'] for r in rl_results['frame_results'][:200]]

        ax.plot(frames, models, marker='o', markersize=3, alpha=0.6)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Model Index')
        ax.set_title('RL Model Selection Timeline (First 200 Frames)')
        ax.set_yticks(range(5))
        ax.set_yticklabels([name.replace('yolov8', '') for name in MODEL_NAMES])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    viz_path = output_dir / "rl_vs_rulebased_comparison.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"[VIZ] Saved visualization to {viz_path}")


def save_results(rl_results, rule_based_results, output_dir):
    """Save comparison results"""
    comparison = {
        'rl_policy': rl_results,
        'rule_based': rule_based_results,
        'summary': {
            'frames_difference': rl_results['total_frames'] - rule_based_results['total_frames'],
            'switches_difference': rl_results['total_switches'] - rule_based_results['total_switches'],
            'savings_difference': rl_results['savings_percent'] - rule_based_results['savings_percent'],
            'params_difference': rl_results['avg_params'] - rule_based_results['avg_params']
        }
    }

    output_file = output_dir / "rl_vs_rulebased_results.json"
    with open(output_file, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(o):
            if isinstance(o, np.integer):
                return int(o)
            elif isinstance(o, np.floating):
                return float(o)
            elif isinstance(o, np.ndarray):
                return o.tolist()
            return o

        json.dump(comparison, f, indent=2, default=convert)

    print(f"\n[SAVE] Results saved to {output_file}")


def main():
    """Main evaluation"""
    # Create environment
    env = TrackingEnvironment(UNCERTAINTY_FILE, GT_FILE, track_id=1)

    # Evaluate RL policy
    rl_results = evaluate_rl_policy(env, MODEL_PATH)

    # Load rule-based results
    rule_based_results = load_rule_based_results(RULE_BASED_RESULTS)

    # Create comparison
    create_comparison_table(rl_results, rule_based_results)

    # Create visualizations
    create_visualization(rl_results, rule_based_results, OUTPUT_DIR)

    # Save results
    save_results(rl_results, rule_based_results, OUTPUT_DIR)

    print("\n" + "="*70)
    print(" EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
