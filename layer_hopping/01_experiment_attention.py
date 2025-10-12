"""
Experiment 1: Attention Visualization

Goal: Visualize attention maps at layers 3, 6, 9, 12 to understand
      what each layer focuses on and whether intermediate layers
      contain enough information for object detection.

Success Criteria:
- Layer 3 attention focuses on person regions (not random) → early exit feasible
- Layer 6 attention focuses on object boundaries → medium exit feasible
- Layer 12 attention focuses on full objects → baseline

If layer 3 attention is scattered/random → early exit NOT feasible
"""

import torch
from transformers import YolosForObjectDetection
from tqdm import tqdm
from pathlib import Path

from src.architecture_constants import MODEL_NAME, EXIT_LAYERS, VAL_SEQUENCES
from src.data_loader import get_sample_dataset
from visualization.attention_viz import (
    visualize_attention_single_layer,
    visualize_multi_layer_attention,
    denormalize_image
)


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
    print(" " * 20 + "EXPERIMENT 1: ATTENTION VISUALIZATION")
    print("=" * 80)
    print("\nGoal: Determine if intermediate transformer layers contain enough")
    print("      semantic information to support object detection.\n")
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

    # Get dataset
    print("\n[2/5] Loading sample data...")
    num_samples = 10
    sample_sequences = VAL_SEQUENCES[:2]  # First 2 validation sequences

    dataset = get_sample_dataset(sequences=sample_sequences, num_samples=num_samples)
    print(f"      ✓ Loaded {len(dataset)} samples from {sample_sequences}")

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

        # Create sample directory
        sample_dir = output_dir / f"sample_{idx:03d}_{sequence}_frame{frame_id:04d}"
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
        tqdm.write(f"      ✓ Sample {idx}: {sequence} frame {frame_id} ({num_boxes} people)")

    # Summary
    print("\n[5/5] Experiment completed!")
    print(f"      Results saved to: {output_dir}")
    print(f"      Total samples: {len(dataset)}")
    print(f"      Layers analyzed: {EXIT_LAYERS}")

    # Analysis instructions
    print("\n" + "=" * 80)
    print(" " * 25 + "NEXT STEPS - MANUAL ANALYSIS")
    print("=" * 80)
    print("\nPlease examine the visualizations and answer:")
    print("\n1. LAYER 3 (Early Exit) - Most Critical:")
    print("   - Does attention focus on person regions?")
    print("   - Or is it scattered randomly across the image?")
    print("   - ✓ If FOCUSED → early exit is FEASIBLE")
    print("   - ✗ If RANDOM → early exit is NOT FEASIBLE")

    print("\n2. LAYER 6 (Medium Exit):")
    print("   - Does attention focus on object boundaries?")
    print("   - Is it more refined than Layer 3?")

    print("\n3. LAYER 9 (Late Exit):")
    print("   - Does attention cover full object regions?")
    print("   - Is it more refined than Layer 6?")

    print("\n4. LAYER 12 (Full Model - Baseline):")
    print("   - Should show clear, focused attention on people")

    print("\n5. HIERARCHY CHECK:")
    print("   - Is there clear progression: Layer 3 < 6 < 9 < 12?")
    print("   - Does attention become more focused/refined at deeper layers?")

    print("\n" + "=" * 80)
    print(" " * 30 + "DECISION POINT")
    print("=" * 80)
    print("\nBased on Layer 3 attention patterns:")
    print("\n✓ IF FOCUSED ON PEOPLE:")
    print("  → Multi-exit YOLOS is architecturally feasible")
    print("  → Proceed to Experiment 2 (Feature Analysis)")
    print("  → Proceed to Experiment 3 (Linear Probe)")

    print("\n✗ IF RANDOM/SCATTERED:")
    print("  → Layer 3 cannot support detection")
    print("  → Consider different exit points (e.g., layers 8, 10, 12)")
    print("  → Or pivot to different architecture entirely")

    print("\n" + "=" * 80)
    print("\nExperiment 1 complete. Review visualizations to make decision.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_experiment()
