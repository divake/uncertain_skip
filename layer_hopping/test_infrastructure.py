"""
Infrastructure Test - Verify All Components Work

Run this to verify that all infrastructure components are correctly set up:
- Architecture constants
- Data loader
- Attention visualization

This test should be run once after setup and will never need to be run again
unless we suspect something is broken.
"""

import sys
from pathlib import Path

def test_architecture_constants():
    """Test that architecture constants are correct"""
    print("\n" + "="*80)
    print("TEST 1: Architecture Constants")
    print("="*80)

    from src.architecture_constants import verify_architecture
    result = verify_architecture()

    assert result, "Architecture verification failed"
    print("✓ PASSED: Architecture constants verified\n")


def test_data_loader():
    """Test that data loader works correctly"""
    print("="*80)
    print("TEST 2: Data Loader")
    print("="*80)

    from src.data_loader import verify_data_loader
    result = verify_data_loader()

    assert result, "Data loader verification failed"
    print("✓ PASSED: Data loader verified\n")


def test_attention_visualization():
    """Test that attention visualization works correctly"""
    print("="*80)
    print("TEST 3: Attention Visualization")
    print("="*80)

    from visualization.attention_viz import verify_attention_visualization
    result = verify_attention_visualization()

    assert result, "Attention visualization verification failed"
    print("✓ PASSED: Attention visualization verified\n")


def test_full_pipeline():
    """Test that the full pipeline works end-to-end"""
    print("="*80)
    print("TEST 4: Full Pipeline (Data → Model → Attention)")
    print("="*80)

    import torch
    from transformers import YolosForObjectDetection
    from src.architecture_constants import MODEL_NAME, TOTAL_SEQ_LENGTH
    from src.data_loader import get_sample_dataset
    from visualization.attention_viz import extract_image_patch_attention, denormalize_image

    # Load model
    print("Loading model...")
    model = YolosForObjectDetection.from_pretrained(MODEL_NAME)
    model.eval()
    print(f"✓ Model loaded: {MODEL_NAME}")

    # Get sample data
    print("\nLoading sample data...")
    dataset = get_sample_dataset(num_samples=1)
    image_tensor, target = dataset[0]
    print(f"✓ Sample loaded: {target['sequence']} frame {target['frame_id']}")
    print(f"  Image shape: {image_tensor.shape}")
    print(f"  Boxes: {target['boxes'].shape[0]}")

    # Forward pass
    print("\nForward pass...")
    pixel_values = image_tensor.unsqueeze(0)
    with torch.no_grad():
        outputs = model(pixel_values, output_attentions=True, output_hidden_states=True)

    print(f"✓ Forward pass successful")
    print(f"  Sequence length: {outputs.hidden_states[0].shape[1]} (expected: {TOTAL_SEQ_LENGTH})")
    print(f"  Num attention layers: {len(outputs.attentions)}")

    # Extract attention
    print("\nExtracting attention...")
    layer_3_attention = outputs.attentions[2]  # Layer 3 (0-indexed)
    attention_single = layer_3_attention[0]  # Remove batch dimension

    image_patch_attn = extract_image_patch_attention(attention_single)
    print(f"✓ Extracted image patch attention: {image_patch_attn.shape}")

    # Denormalize image
    print("\nDenormalizing image...")
    image_np = denormalize_image(image_tensor)
    print(f"✓ Denormalized image: {image_np.shape}, dtype: {image_np.dtype}")

    print("\n✓ PASSED: Full pipeline works end-to-end\n")
    return True


def run_all_tests():
    """Run all infrastructure tests"""
    print("\n" + "="*80)
    print(" " * 20 + "INFRASTRUCTURE TEST SUITE")
    print("="*80)
    print("\nThis verifies that all infrastructure components work correctly.")
    print("These components will NEVER change in future experiments.\n")

    try:
        test_architecture_constants()
        test_data_loader()
        test_attention_visualization()
        test_full_pipeline()

        print("="*80)
        print(" " * 25 + "ALL TESTS PASSED ✓")
        print("="*80)
        print("\nInfrastructure is VERIFIED and READY for experiments.")
        print("You can now run experiments with confidence that the foundation is solid.")
        print("\n" + "="*80 + "\n")

        return True

    except Exception as e:
        print("\n" + "="*80)
        print(" " * 25 + "TEST FAILED ✗")
        print("="*80)
        print(f"\nError: {str(e)}")
        print("\nPlease fix the issue before running experiments.\n")
        import traceback
        traceback.print_exc()

        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
