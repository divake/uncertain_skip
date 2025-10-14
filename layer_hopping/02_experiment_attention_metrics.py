"""
Experiment 2: Quantitative Attention Analysis

Analyzes attention quality across ALL layers (3-12) using objective metrics:
1. In-box Attention Ratio: % of attention inside ground truth bounding boxes
2. Attention Entropy: Measure of attention focus (lower = more focused)
3. Attention Concentration: % of attention in top-10% of patches

Goal: Identify which layers have sufficient attention quality for early exit
"""

import torch
import numpy as np
from transformers import YolosForObjectDetection
from tqdm import tqdm
from pathlib import Path
import json

from src.architecture_constants import MODEL_NAME, VAL_SEQUENCES, DATASET_ROOT
from src.data_loader import MOT17DetectionDataset


def get_quality_samples(min_people=8, min_avg_size=0.02, min_person_size=0.005, num_samples=20):
    """Get high-quality samples with clearly visible people"""

    dataset = MOT17DetectionDataset(
        root_path=DATASET_ROOT,
        sequences=VAL_SEQUENCES[:2]
    )

    print(f"Filtering {len(dataset)} samples for high-quality frames...")
    print(f"  Criteria: ≥{min_people} people, avg size ≥{min_avg_size*100:.1f}%, min size ≥{min_person_size*100:.1f}%")

    quality_samples = []

    for idx in range(len(dataset)):
        _, target = dataset[idx]
        boxes = target['boxes']
        num_people = boxes.shape[0]

        if num_people < min_people:
            continue

        areas = (boxes[:, 2] * boxes[:, 3]).numpy()
        avg_area = areas.mean()
        min_area = areas.min()

        if avg_area >= min_avg_size and min_area >= min_person_size:
            quality_samples.append({
                'idx': idx,
                'num_people': num_people,
                'avg_size': avg_area,
                'sequence': target['sequence'],
                'frame_id': target['frame_id']
            })

    print(f"Found {len(quality_samples)} high-quality frames")

    if len(quality_samples) == 0:
        print("\n⚠ WARNING: No frames found. Try relaxing filters.")
        return None

    quality_samples.sort(key=lambda x: (x['num_people'], x['avg_size']), reverse=True)
    selected_indices = [s['idx'] for s in quality_samples[:min(num_samples, len(quality_samples))]]

    subset = torch.utils.data.Subset(dataset, selected_indices)
    return subset, quality_samples[:min(num_samples, len(quality_samples))]


def extract_image_patch_attention(attention):
    """
    Extract attention for image patches only (exclude CLS and detection tokens)

    Args:
        attention: [num_heads, seq_len, seq_len] where seq_len = 1125

    Returns:
        [num_heads, 1024, 1024] attention between image patches
    """
    # Positions: 0=CLS, 1:1025=image patches, 1025:1125=detection tokens
    image_attention = attention[:, 1:1025, 1:1025]
    return image_attention


def compute_in_box_ratio(attention_map, boxes):
    """
    Compute percentage of attention that falls inside ground truth boxes

    Args:
        attention_map: [32, 32] attention heatmap (averaged over heads)
        boxes: [N, 4] ground truth boxes in normalized [cx, cy, w, h] format

    Returns:
        in_box_ratio: float between 0-1
    """
    h, w = attention_map.shape  # Should be 32x32

    # Create mask for ground truth boxes
    box_mask = np.zeros((h, w), dtype=bool)

    for box in boxes:
        cx, cy, box_w, box_h = box

        # Convert to patch coordinates (32x32 grid)
        x1 = int((cx - box_w/2) * w)
        y1 = int((cy - box_h/2) * h)
        x2 = int((cx + box_w/2) * w)
        y2 = int((cy + box_h/2) * h)

        # Clip to valid range
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        box_mask[y1:y2, x1:x2] = True

    # Calculate ratio
    attention_in_boxes = attention_map[box_mask].sum()
    total_attention = attention_map.sum()

    ratio = attention_in_boxes / total_attention if total_attention > 0 else 0
    return ratio


def compute_attention_entropy(attention_map):
    """
    Compute entropy of attention distribution (lower = more focused)

    Args:
        attention_map: [32, 32] attention heatmap

    Returns:
        entropy: float (in nats)
    """
    # Flatten and normalize to probability distribution
    attention_flat = attention_map.flatten()
    attention_flat = attention_flat / attention_flat.sum()

    # Compute entropy: -sum(p * log(p))
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    entropy = -np.sum(attention_flat * np.log(attention_flat + epsilon))

    return entropy


def compute_attention_concentration(attention_map, top_k_percent=10):
    """
    Compute what percentage of total attention is in top-k% of patches

    Args:
        attention_map: [32, 32] attention heatmap
        top_k_percent: percentage of patches to consider (default 10%)

    Returns:
        concentration: float between 0-1
    """
    attention_flat = attention_map.flatten()
    attention_sorted = np.sort(attention_flat)[::-1]  # Sort descending

    k = int(len(attention_flat) * top_k_percent / 100)
    top_k_sum = attention_sorted[:k].sum()
    total_sum = attention_sorted.sum()

    concentration = top_k_sum / total_sum if total_sum > 0 else 0
    return concentration


def analyze_layer_attention(model, dataset, layer_idx, device):
    """
    Analyze attention quality for a specific layer across all samples

    Returns:
        metrics: dict with mean and std for each metric
    """
    in_box_ratios = []
    entropies = []
    concentrations = []

    for idx in range(len(dataset)):
        image_tensor, target = dataset[idx]
        boxes = target['boxes'].numpy()

        # Forward pass
        pixel_values = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(pixel_values, output_attentions=True)

        # Get attention for this layer (convert to 0-indexed)
        attention = outputs.attentions[layer_idx - 1][0]  # [num_heads, seq_len, seq_len]

        # Extract image patch attention
        image_attention = extract_image_patch_attention(attention)  # [num_heads, 1024, 1024]

        # Average over heads and spatial dimensions (rows)
        attention_per_patch = image_attention.mean(dim=0).mean(dim=0)  # [1024]

        # Reshape to 32x32 grid
        attention_map = attention_per_patch.reshape(32, 32).cpu().numpy()

        # Compute metrics
        in_box_ratio = compute_in_box_ratio(attention_map, boxes)
        entropy = compute_attention_entropy(attention_map)
        concentration = compute_attention_concentration(attention_map, top_k_percent=10)

        in_box_ratios.append(in_box_ratio)
        entropies.append(entropy)
        concentrations.append(concentration)

    # Aggregate statistics
    metrics = {
        'in_box_ratio_mean': np.mean(in_box_ratios),
        'in_box_ratio_std': np.std(in_box_ratios),
        'entropy_mean': np.mean(entropies),
        'entropy_std': np.std(entropies),
        'concentration_mean': np.mean(concentrations),
        'concentration_std': np.std(concentrations),
        'in_box_ratios': in_box_ratios,
        'entropies': entropies,
        'concentrations': concentrations
    }

    return metrics


def run_experiment():
    """Main experiment function"""

    print("=" * 80)
    print(" " * 10 + "EXPERIMENT 2: QUANTITATIVE ATTENTION ANALYSIS")
    print("=" * 80)
    print("\nAnalyzing attention quality across layers 3-12")
    print("\nMetrics:")
    print("  1. In-box Ratio: % of attention inside ground truth boxes")
    print("  2. Entropy: Measure of attention focus (lower = better)")
    print("  3. Concentration: % of attention in top-10% patches (higher = better)")
    print("=" * 80)

    # Setup
    print("\n[1/4] Loading model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"      Device: {device}")

    model = YolosForObjectDetection.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    print(f"      ✓ Model loaded: {MODEL_NAME}")

    # Get dataset
    print("\n[2/4] Loading quality samples...")
    dataset, sample_info = get_quality_samples(
        min_people=8,
        min_avg_size=0.02,
        min_person_size=0.005,
        num_samples=20  # Use 20 samples for robust statistics
    )

    if dataset is None:
        print("\n✗ No suitable samples found.")
        return

    print(f"      ✓ Loaded {len(dataset)} samples")

    # Create output directory
    output_dir = Path("results/experiment_02_attention_metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze all layers
    print("\n[3/4] Analyzing attention across layers 3-12...")
    print(f"      Processing {len(dataset)} samples per layer")
    print()

    all_results = {}
    layers = list(range(3, 13))  # Layers 3-12

    for layer_idx in tqdm(layers, desc="      Progress"):
        metrics = analyze_layer_attention(model, dataset, layer_idx, device)
        all_results[f"layer_{layer_idx}"] = metrics

        tqdm.write(f"      Layer {layer_idx:2d}: "
                  f"In-box={metrics['in_box_ratio_mean']:.3f}, "
                  f"Entropy={metrics['entropy_mean']:.3f}, "
                  f"Concentration={metrics['concentration_mean']:.3f}")

    # Save results
    print("\n[4/4] Saving results...")

    # Save raw data
    results_file = output_dir / "attention_metrics.json"
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for layer, metrics in all_results.items():
            json_results[layer] = {
                'in_box_ratio_mean': float(metrics['in_box_ratio_mean']),
                'in_box_ratio_std': float(metrics['in_box_ratio_std']),
                'entropy_mean': float(metrics['entropy_mean']),
                'entropy_std': float(metrics['entropy_std']),
                'concentration_mean': float(metrics['concentration_mean']),
                'concentration_std': float(metrics['concentration_std']),
            }
        json.dump(json_results, f, indent=2)

    print(f"      ✓ Saved metrics to: {results_file}")

    # Generate summary table
    print("\n" + "=" * 80)
    print(" " * 25 + "RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Layer':<8} {'In-Box Ratio':<20} {'Entropy':<20} {'Concentration':<20}")
    print(f"{'':8} {'(higher=better)':<20} {'(lower=better)':<20} {'(higher=better)':<20}")
    print("-" * 80)

    for layer_idx in layers:
        metrics = all_results[f"layer_{layer_idx}"]
        print(f"{layer_idx:<8} "
              f"{metrics['in_box_ratio_mean']:.3f} ± {metrics['in_box_ratio_std']:.3f}      "
              f"{metrics['entropy_mean']:.3f} ± {metrics['entropy_std']:.3f}      "
              f"{metrics['concentration_mean']:.3f} ± {metrics['concentration_std']:.3f}")

    # Analysis
    print("\n" + "=" * 80)
    print(" " * 30 + "ANALYSIS")
    print("=" * 80)

    # Find best layers for each metric
    best_in_box = max(layers, key=lambda l: all_results[f"layer_{l}"]['in_box_ratio_mean'])
    best_entropy = min(layers, key=lambda l: all_results[f"layer_{l}"]['entropy_mean'])
    best_concentration = max(layers, key=lambda l: all_results[f"layer_{l}"]['concentration_mean'])

    print(f"\nBest In-Box Ratio: Layer {best_in_box} "
          f"({all_results[f'layer_{best_in_box}']['in_box_ratio_mean']:.3f})")
    print(f"Best Entropy: Layer {best_entropy} "
          f"({all_results[f'layer_{best_entropy}']['entropy_mean']:.3f})")
    print(f"Best Concentration: Layer {best_concentration} "
          f"({all_results[f'layer_{best_concentration}']['concentration_mean']:.3f})")

    # Identify viable exit layers (heuristic thresholds)
    print("\n" + "-" * 80)
    print("Viable Exit Layers (In-Box Ratio > 0.3):")
    viable_layers = [l for l in layers if all_results[f"layer_{l}"]['in_box_ratio_mean'] > 0.3]
    if viable_layers:
        print(f"  Layers: {viable_layers}")
        print(f"  → These layers focus >30% attention on people")
    else:
        print("  None found with current threshold")
        print("  → May need to use later layers for exits")

    print("\n" + "=" * 80)
    print("\nExperiment 2 complete. Results saved to:", output_dir)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_experiment()
