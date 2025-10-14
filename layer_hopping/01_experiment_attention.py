"""
Experiment 1: Attention Visualization with Quality Sample Selection

Filters:
1. ≥8 people in frame
2. Average person size ≥2% of image (clearly visible)
3. No people smaller than 0.5% of image (filters out distant/tiny people)

Goal: Visualize attention maps at layers 3, 6, 9, 12 with high-quality samples
"""

import torch
from transformers import YolosForObjectDetection
from tqdm import tqdm
from pathlib import Path
import numpy as np

from src.architecture_constants import MODEL_NAME, EXIT_LAYERS, VAL_SEQUENCES, DATASET_ROOT
from src.data_loader import MOT17DetectionDataset
from visualization.attention_viz import (
    visualize_attention_single_layer,
    visualize_multi_layer_attention,
    denormalize_image
)


def get_quality_samples(min_people=8, min_avg_size=0.02, min_person_size=0.005, num_samples=10):
    """
    Get high-quality samples with clearly visible people

    Args:
        min_people: Minimum number of people in frame
        min_avg_size: Minimum average person size (fraction of image area)
        min_person_size: Minimum size for any single person (filters tiny/distant people)
        num_samples: Number of samples to return
    """

    # Load full dataset
    dataset = MOT17DetectionDataset(
        root_path=DATASET_ROOT,
        sequences=VAL_SEQUENCES[:2]
    )

    print(f"Filtering {len(dataset)} samples for high-quality frames...")
    print(f"  Criteria:")
    print(f"    - ≥{min_people} people")
    print(f"    - Average person size ≥{min_avg_size*100:.1f}% of image")
    print(f"    - All people ≥{min_person_size*100:.1f}% of image (no tiny people)")

    # Filter samples
    quality_samples = []

    for idx in range(len(dataset)):
        _, target = dataset[idx]
        boxes = target['boxes']
        num_people = boxes.shape[0]

        if num_people < min_people:
            continue

        # Calculate person sizes (area = width × height)
        areas = (boxes[:, 2] * boxes[:, 3]).numpy()
        avg_area = areas.mean()
        min_area = areas.min()

        # Quality check
        if avg_area >= min_avg_size and min_area >= min_person_size:
            quality_samples.append({
                'idx': idx,
                'num_people': num_people,
                'avg_size': avg_area,
                'min_size': min_area,
                'sequence': target['sequence'],
                'frame_id': target['frame_id']
            })

    print(f"\nFound {len(quality_samples)} high-quality frames")

    if len(quality_samples) == 0:
        print("\n⚠ WARNING: No frames found with current criteria!")
        print("   Try relaxing the filters (e.g., min_people=5, min_avg_size=0.01)")
        return None

    # Sort by number of people, then by average size
    quality_samples.sort(key=lambda x: (x['num_people'], x['avg_size']), reverse=True)

    # Take top num_samples
    selected = quality_samples[:min(num_samples, len(quality_samples))]
    selected_indices = [s['idx'] for s in selected]

    # Print selected samples
    print("\nSelected samples:")
    print(f"{'#':<4} {'Sequence':<20} {'Frame':<6} {'People':<8} {'Avg Size':<10} {'Min Size':<10}")
    print("-" * 70)
    for i, s in enumerate(selected):
        print(f"{i:<4} {s['sequence']:<20} {s['frame_id']:<6} {s['num_people']:<8} "
              f"{s['avg_size']*100:>6.2f}%    {s['min_size']*100:>6.2f}%")

    # Create subset
    subset = torch.utils.data.Subset(dataset, selected_indices)
    return subset


def extract_attentions(model, pixel_values, layer_indices):
    """
    Extract attention maps from specified layers

    Args:
        model: YOLOS model
        pixel_values: Input images [batch, 3, H, W]
        layer_indices: List of layer indices to extract (e.g., [3, 6, 9, 12])

    Returns:
        List of attention tensors, one per layer [num_heads, seq_len, seq_len]
    """
    with torch.no_grad():
        outputs = model(pixel_values, output_attentions=True)

    attentions = outputs.attentions  # Tuple of tensors, one per layer

    # Extract specified layers (convert 1-indexed to 0-indexed)
    selected_attentions = [attentions[idx - 1][0] for idx in layer_indices]  # Remove batch dim

    return selected_attentions


def run_experiment():
    """Main experiment function"""

    print("=" * 80)
    print(" " * 12 + "EXPERIMENT 1: ATTENTION VISUALIZATION")
    print("=" * 80)
    print("\nQuality filters: ≥8 people, avg size ≥2%, all people ≥0.5%")
    print("\nGoal: Determine if intermediate transformer layers can detect people\n")
    print("Critical Question: Does layer 3 attention focus on people or is it random?")
    print("=" * 80)

    # Setup
    print("\n[1/5] Loading model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"      Device: {device}")

    model = YolosForObjectDetection.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    print(f"      ✓ Model loaded: {MODEL_NAME}")

    # Get quality dataset
    print("\n[2/5] Loading quality sample data...")
    dataset = get_quality_samples(
        min_people=8,
        min_avg_size=0.02,      # 2% of image
        min_person_size=0.005,  # 0.5% of image (filters tiny people)
        num_samples=10
    )

    if dataset is None:
        print("\n✗ No suitable samples found. Please check your dataset or relax criteria.")
        return

    # Create output directory
    print("\n[3/5] Setting up output directory...")
    output_dir = Path("results/experiment_01_attention")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"      ✓ Output: {output_dir}")

    # Process samples
    print(f"\n[4/5] Processing {len(dataset)} samples...")
    print("      Extracting attention from layers:", EXIT_LAYERS)
    print()

    for idx in tqdm(range(len(dataset)), desc="      Progress"):
        # Get sample
        image_tensor, target = dataset[idx]

        # Forward pass
        pixel_values = image_tensor.unsqueeze(0).to(device)
        attentions = extract_attentions(model, pixel_values, EXIT_LAYERS)

        # Denormalize image
        image_np = denormalize_image(image_tensor)

        # Get metadata
        sequence = target['sequence']
        frame_id = target['frame_id']
        num_boxes = target['boxes'].shape[0]

        # Calculate avg person size for naming
        avg_size = (target['boxes'][:, 2] * target['boxes'][:, 3]).mean().item()

        # Create sample directory
        sample_dir = output_dir / f"sample_{idx:03d}_{sequence}_frame{frame_id:04d}_{num_boxes}ppl_{avg_size*100:.1f}pct"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save original image
        from PIL import Image
        Image.fromarray(image_np).save(sample_dir / "original.jpg")

        # Visualize individual layers
        for i, (attn, layer_idx) in enumerate(zip(attentions, EXIT_LAYERS)):
            # Visualize first head
            save_path = sample_dir / f"layer_{layer_idx:02d}_head_0.png"
            visualize_attention_single_layer(
                attention=attn,
                image=image_np,
                layer_idx=layer_idx,
                head_idx=0,
                save_path=save_path
            )

        # Visualize all layers side by side
        comparison_path = sample_dir / "comparison_all_layers.png"
        visualize_multi_layer_attention(
            attentions=attentions,
            image=image_np,
            layer_indices=EXIT_LAYERS,
            save_path=comparison_path
        )

        # Log
        tqdm.write(f"      ✓ Sample {idx}: {sequence} frame {frame_id} ({num_boxes} people, avg size {avg_size*100:.1f}%)")

    # Summary
    print("\n[5/5] Experiment completed!")
    print(f"      Results saved to: {output_dir}")
    print(f"      Total samples: {len(dataset)}")
    print(f"      Layers analyzed: {EXIT_LAYERS}")
    print(f"      All samples have clearly visible people (avg size ≥2%)")

    # Analysis instructions
    print("\n" + "=" * 80)
    print(" " * 25 + "NEXT STEPS - MANUAL ANALYSIS")
    print("=" * 80)
    print("\nExamine the visualizations in:", output_dir)
    print("\nKey Question: Does Layer 3 attention focus on people or is it random?")
    print("\n✓ IF FOCUSED → Multi-exit YOLOS is feasible, continue experiments")
    print("✗ IF RANDOM → Layer 3 cannot detect, pivot strategy")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    run_experiment()
