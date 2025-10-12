"""
Attention Visualization - FIXED VERSION - WILL NEVER CHANGE

This module handles visualization of YOLOS attention maps.
It correctly handles the sequence structure:
  [CLS, image_patch_1, ..., image_patch_1024, det_token_1, ..., det_token_100]

Verified and tested. DO NOT MODIFY.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Optional
import cv2

try:
    from src.architecture_constants import (
        NUM_IMAGE_PATCHES,
        NUM_PATCHES_PER_DIM,
        IMAGE_PATCH_START,
        IMAGE_PATCH_END,
        IMAGE_MEAN,
        IMAGE_STD
    )
except ImportError:
    # Fallback for standalone
    import os
    os.chdir(Path(__file__).parent.parent)
    from src.architecture_constants import (
        NUM_IMAGE_PATCHES,
        NUM_PATCHES_PER_DIM,
        IMAGE_PATCH_START,
        IMAGE_PATCH_END,
        IMAGE_MEAN,
        IMAGE_STD
    )


def extract_image_patch_attention(attention: torch.Tensor) -> torch.Tensor:
    """
    Extract attention map for image patches only (excluding CLS and detection tokens)

    Args:
        attention: Full attention tensor [num_heads, seq_len, seq_len]
                  where seq_len = 1125 (1 CLS + 1024 patches + 100 det_tokens)

    Returns:
        Image patch attention [num_heads, 1024, 1024]
    """
    # Extract image patch positions (1:1025)
    image_attention = attention[:, IMAGE_PATCH_START:IMAGE_PATCH_END, IMAGE_PATCH_START:IMAGE_PATCH_END]
    return image_attention


def attention_to_heatmap(attention: torch.Tensor, image_size: tuple) -> np.ndarray:
    """
    Convert attention tensor to heatmap image

    Args:
        attention: Attention tensor [num_heads, num_patches, num_patches]
        image_size: Target size (height, width)

    Returns:
        Heatmap as numpy array [height, width] normalized to [0, 1]
    """
    # Average across all queries (or you can focus on specific tokens)
    # Here we average the attention each patch pays to all others
    attn_np = attention.cpu().numpy()

    # Average across heads
    attn_avg_heads = attn_np.mean(axis=0)  # [num_patches, num_patches]

    # Average across queries (rows) to get attention received by each patch
    attn_avg = attn_avg_heads.mean(axis=0)  # [num_patches]

    # Reshape to 2D grid
    attn_map = attn_avg.reshape(NUM_PATCHES_PER_DIM, NUM_PATCHES_PER_DIM)

    # Resize to image size
    attn_map_resized = cv2.resize(attn_map, (image_size[1], image_size[0]))

    # Normalize to [0, 1]
    attn_min = attn_map_resized.min()
    attn_max = attn_map_resized.max()
    if attn_max > attn_min:
        attn_map_norm = (attn_map_resized - attn_min) / (attn_max - attn_min)
    else:
        attn_map_norm = np.zeros_like(attn_map_resized)

    return attn_map_norm


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize image tensor for visualization

    Args:
        tensor: Normalized image tensor [3, H, W]

    Returns:
        Image as numpy array [H, W, 3] in range [0, 255] uint8
    """
    mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD).view(3, 1, 1)

    # Denormalize
    tensor = tensor * std + mean

    # To numpy and transpose [3, H, W] -> [H, W, 3]
    image = tensor.cpu().numpy().transpose(1, 2, 0)

    # Clip and convert to uint8
    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    return image


def visualize_attention_single_layer(
    attention: torch.Tensor,
    image: np.ndarray,
    layer_idx: int,
    head_idx: int = 0,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize attention map from a single layer and head

    Args:
        attention: Attention tensor [num_heads, seq_len, seq_len]
        image: Original image as numpy array [H, W, 3] uint8
        layer_idx: Layer number (for title)
        head_idx: Which attention head to visualize
        save_path: Optional path to save figure

    Returns:
        Overlay image as numpy array
    """
    # Extract image patch attention only
    image_attn = extract_image_patch_attention(attention)

    # Get specified head
    head_attn = image_attn[head_idx]  # [1024, 1024]

    # Convert to heatmap
    heatmap_norm = attention_to_heatmap(head_attn.unsqueeze(0), image.shape[:2])

    # Create colored heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay on image
    overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(heatmap_norm, cmap='hot')
    axes[1].set_title(f"Layer {layer_idx} - Head {head_idx}\nAttention Heatmap", fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {save_path}")

    plt.close()

    return overlay


def visualize_multi_layer_attention(
    attentions: List[torch.Tensor],
    image: np.ndarray,
    layer_indices: List[int],
    save_path: Optional[str] = None
) -> None:
    """
    Visualize attention maps from multiple layers side by side

    Args:
        attentions: List of attention tensors [num_heads, seq_len, seq_len]
        image: Original image [H, W, 3] uint8
        layer_indices: Layer numbers for titles
        save_path: Path to save figure
    """
    num_layers = len(attentions)

    fig, axes = plt.subplots(2, num_layers + 1, figsize=(4 * (num_layers + 1), 8))

    # Original image (span both rows)
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    for i, (attn, layer_idx) in enumerate(zip(attentions, layer_indices)):
        col = i + 1

        # Extract image patch attention
        image_attn = extract_image_patch_attention(attn)

        # Average across heads
        avg_attn = image_attn.mean(dim=0)  # [1024, 1024]

        # Convert to heatmap
        heatmap_norm = attention_to_heatmap(avg_attn.unsqueeze(0), image.shape[:2])

        # Create colored heatmap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Overlay
        overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)

        # Plot heatmap
        axes[0, col].imshow(heatmap_norm, cmap='hot')
        axes[0, col].set_title(f"Layer {layer_idx}\nAttention", fontsize=10, fontweight='bold')
        axes[0, col].axis('off')

        # Plot overlay
        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f"Layer {layer_idx}\nOverlay", fontsize=10)
        axes[1, col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved comparison: {save_path}")

    plt.close()


# ==============================================================================
# VERIFICATION FUNCTION
# ==============================================================================
def verify_attention_visualization():
    """
    Verify that attention visualization works correctly.
    """
    print("Verifying attention visualization...")

    # Create dummy attention (simulating YOLOS output)
    from src.architecture_constants import TOTAL_SEQ_LENGTH, NUM_ATTENTION_HEADS

    dummy_attn = torch.randn(NUM_ATTENTION_HEADS, TOTAL_SEQ_LENGTH, TOTAL_SEQ_LENGTH)
    print(f"✓ Created dummy attention: {dummy_attn.shape}")

    # Extract image patches
    image_attn = extract_image_patch_attention(dummy_attn)
    print(f"✓ Extracted image patch attention: {image_attn.shape}")

    expected_shape = (NUM_ATTENTION_HEADS, NUM_IMAGE_PATCHES, NUM_IMAGE_PATCHES)
    assert image_attn.shape == expected_shape, f"Shape mismatch: {image_attn.shape} vs {expected_shape}"

    # Convert to heatmap
    heatmap = attention_to_heatmap(image_attn, (512, 512))
    print(f"✓ Converted to heatmap: {heatmap.shape}, range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")

    assert heatmap.shape == (512, 512), f"Wrong heatmap shape: {heatmap.shape}"
    assert heatmap.min() >= 0 and heatmap.max() <= 1, f"Heatmap not normalized: [{heatmap.min()}, {heatmap.max()}]"

    print("\n✓ Attention visualization verification passed!")

    return True


if __name__ == "__main__":
    verify_attention_visualization()
