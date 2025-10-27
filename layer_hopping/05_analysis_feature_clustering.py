"""
Test 3: Feature Clustering Analysis

Extracts Layer 8 features for patches inside vs outside person bboxes.
Determines if features are semantically meaningful (cluster separately) or just texture.
"""

import sys
import os
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.data_loader import MOT17DetectionDataset
from src.architecture_constants import VAL_SEQUENCES
from src.utils import get_device

# Import multi-exit YOLOS
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'multi_exit_yolos'))
from models.multi_exit_yolos import build_multi_exit_yolos


def extract_features_for_patches(model, dataloader, device, layer=8, num_samples=500):
    """
    Extract Layer 8 features for patches inside and outside person bboxes.

    Args:
        model: Multi-exit YOLOS model
        dataloader: Validation dataloader
        device: Device to run on
        layer: Which layer to extract features from
        num_samples: How many patch samples to collect per category

    Returns:
        dict: Features for in-box and out-of-box patches
    """
    model.eval()

    in_box_features = []
    out_of_box_features = []

    print(f"\n{'='*80}")
    print(f"Extracting Layer {layer} Features for Patches")
    print(f"{'='*80}\n")

    total_samples_needed = num_samples * 2

    for images, targets in tqdm(dataloader, desc=f"Extracting features"):
        if len(in_box_features) >= num_samples and len(out_of_box_features) >= num_samples:
            break

        images = images.to(device)
        batch_size = images.shape[0]

        # Forward pass through backbone to get intermediate features
        with torch.no_grad():
            # Get encoder outputs at specified layer
            pixel_values = images

            # Process through YOLOS backbone
            outputs = model.backbone(pixel_values, output_hidden_states=True)

            # Get hidden states at the specified layer
            # hidden_states: tuple of (batch_size, num_patches + 1, hidden_dim)
            # Index mapping: layer 0 = embeddings, layer 1-12 = encoder layers
            hidden_states = outputs.hidden_states[layer]  # Shape: [B, 101, 384] for YOLOS-small

            # Remove CLS token (first token)
            patch_features = hidden_states[:, 1:, :]  # [B, 100, 384]

        # For each image in batch
        for i in range(batch_size):
            if len(in_box_features) >= num_samples and len(out_of_box_features) >= num_samples:
                break

            # Get ground truth boxes for this image
            gt_boxes = targets[i]['boxes']  # [N, 4] in [cx, cy, w, h] format

            if len(gt_boxes) == 0:
                continue

            # Patch features for this image: [100, 384]
            features = patch_features[i].cpu().numpy()

            # YOLOS uses 10x10 grid of patches (100 patches total) for 512x512 image
            # Each patch is 51.2x51.2 pixels
            grid_size = 10
            patch_size = 512 / grid_size  # 51.2 pixels

            # Determine which patches are inside/outside bboxes
            for patch_idx in range(100):
                # Calculate patch center in normalized coordinates
                row = patch_idx // grid_size
                col = patch_idx % grid_size

                patch_cx = (col + 0.5) / grid_size  # Normalized [0, 1]
                patch_cy = (row + 0.5) / grid_size  # Normalized [0, 1]

                # Check if patch center is inside any ground truth box
                is_inside = False
                for box in gt_boxes:
                    box_cx, box_cy, box_w, box_h = box.tolist()

                    # Check if patch center is inside this box
                    if (abs(patch_cx - box_cx) < box_w / 2 and
                        abs(patch_cy - box_cy) < box_h / 2):
                        is_inside = True
                        break

                # Collect features based on location
                if is_inside and len(in_box_features) < num_samples:
                    in_box_features.append(features[patch_idx])
                elif not is_inside and len(out_of_box_features) < num_samples:
                    out_of_box_features.append(features[patch_idx])

    in_box_features = np.array(in_box_features)
    out_of_box_features = np.array(out_of_box_features)

    print(f"\nCollected {len(in_box_features)} in-box patch features")
    print(f"Collected {len(out_of_box_features)} out-of-box patch features")

    return {
        'in_box_features': in_box_features,
        'out_of_box_features': out_of_box_features,
    }


def analyze_feature_clustering(features_dict, layer):
    """
    Analyze if in-box and out-of-box features cluster separately.

    Args:
        features_dict: Dictionary with in_box_features and out_of_box_features
        layer: Layer number

    Returns:
        dict: Clustering analysis results
    """
    in_box = features_dict['in_box_features']
    out_of_box = features_dict['out_of_box_features']

    print(f"\n{'='*80}")
    print(f"Layer {layer} - Feature Clustering Analysis")
    print(f"{'='*80}\n")

    # Combine features and create labels
    all_features = np.vstack([in_box, out_of_box])
    labels = np.array([0] * len(in_box) + [1] * len(out_of_box))  # 0=in-box, 1=out-of-box

    print(f"Total samples: {len(all_features)}")
    print(f"  In-box: {len(in_box)}")
    print(f"  Out-of-box: {len(out_of_box)}")
    print(f"Feature dimension: {all_features.shape[1]}")
    print()

    # 1. PCA for dimensionality reduction
    print("Running PCA...")
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(all_features)

    explained_variance = pca.explained_variance_ratio_[:10].sum()
    print(f"Variance explained by top 10 components: {explained_variance:.4f}")
    print()

    # 2. K-means clustering (k=2)
    print("Running K-Means clustering (k=2)...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_predictions = kmeans.fit_predict(features_pca)

    # Calculate clustering accuracy
    # Try both label assignments (cluster 0 = in-box OR cluster 1 = in-box)
    accuracy_1 = np.mean(cluster_predictions == labels)
    accuracy_2 = np.mean(cluster_predictions != labels)
    clustering_accuracy = max(accuracy_1, accuracy_2)

    print(f"K-Means clustering accuracy: {clustering_accuracy:.4f}")
    print()

    # 3. Silhouette score (measures cluster separation quality)
    print("Computing silhouette score...")
    silhouette = silhouette_score(features_pca, labels)
    print(f"Silhouette score: {silhouette:.4f}")
    print(f"  Range: [-1, 1], where >0.5 = good separation")
    print()

    # 4. Mean distance within vs between groups
    print("Computing feature distances...")

    in_box_pca = features_pca[labels == 0]
    out_of_box_pca = features_pca[labels == 1]

    # Within-group distances
    in_box_center = in_box_pca.mean(axis=0)
    out_of_box_center = out_of_box_pca.mean(axis=0)

    within_in_box = np.linalg.norm(in_box_pca - in_box_center, axis=1).mean()
    within_out_of_box = np.linalg.norm(out_of_box_pca - out_of_box_center, axis=1).mean()
    within_group_dist = (within_in_box + within_out_of_box) / 2

    # Between-group distance
    between_group_dist = np.linalg.norm(in_box_center - out_of_box_center)

    separation_ratio = between_group_dist / within_group_dist

    print(f"Within-group distance: {within_group_dist:.4f}")
    print(f"Between-group distance: {between_group_dist:.4f}")
    print(f"Separation ratio: {separation_ratio:.4f}")
    print(f"  Ratio > 1.5 indicates good separation")
    print()

    # 5. t-SNE for visualization (2D projection)
    print("Running t-SNE for visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_tsne = tsne.fit_transform(features_pca)

    # Calculate overlap in t-SNE space
    in_box_tsne = features_tsne[labels == 0]
    out_of_box_tsne = features_tsne[labels == 1]

    in_box_tsne_center = in_box_tsne.mean(axis=0)
    out_of_box_tsne_center = out_of_box_tsne.mean(axis=0)

    tsne_separation = np.linalg.norm(in_box_tsne_center - out_of_box_tsne_center)
    tsne_spread = (np.linalg.norm(in_box_tsne - in_box_tsne_center, axis=1).mean() +
                   np.linalg.norm(out_of_box_tsne - out_of_box_tsne_center, axis=1).mean()) / 2

    tsne_ratio = tsne_separation / tsne_spread

    print(f"t-SNE separation: {tsne_separation:.4f}")
    print(f"t-SNE spread: {tsne_spread:.4f}")
    print(f"t-SNE ratio: {tsne_ratio:.4f}")
    print()

    return {
        'layer': layer,
        'num_samples': len(all_features),
        'feature_dim': all_features.shape[1],
        'pca_variance_explained': float(explained_variance),
        'kmeans_accuracy': float(clustering_accuracy),
        'silhouette_score': float(silhouette),
        'within_group_distance': float(within_group_dist),
        'between_group_distance': float(between_group_dist),
        'separation_ratio': float(separation_ratio),
        'tsne_separation': float(tsne_separation),
        'tsne_spread': float(tsne_spread),
        'tsne_ratio': float(tsne_ratio),
        # Save t-SNE coordinates for plotting
        'tsne_coords_in_box': in_box_tsne.tolist(),
        'tsne_coords_out_of_box': out_of_box_tsne.tolist(),
    }


def interpret_clustering_results(results):
    """Interpret what the clustering results tell us"""
    print(f"\n{'='*80}")
    print("Interpretation: What Do These Features Tell Us?")
    print(f"{'='*80}\n")

    silhouette = results['silhouette_score']
    separation_ratio = results['separation_ratio']
    kmeans_acc = results['kmeans_accuracy']

    print(f"Key Metrics:")
    print(f"  Silhouette score: {silhouette:.4f} (>0.5 = good separation)")
    print(f"  Separation ratio: {separation_ratio:.4f} (>1.5 = good separation)")
    print(f"  K-means accuracy: {kmeans_acc:.4f} (>0.7 = clusters match labels)")
    print()

    # Diagnose feature quality
    print(f"Layer {results['layer']} Feature Quality:")

    if silhouette > 0.5:
        print("  ✅ High silhouette score - features have clear cluster structure")
    elif silhouette > 0.2:
        print("  ⚠️  Moderate silhouette score - some cluster structure present")
    else:
        print("  ❌ Low silhouette score - features don't cluster well")

    if separation_ratio > 1.5:
        print("  ✅ High separation ratio - in-box and out-of-box features are distinct")
    elif separation_ratio > 1.0:
        print("  ⚠️  Moderate separation - some distinction between groups")
    else:
        print("  ❌ Low separation - in-box and out-of-box features are mixed")

    if kmeans_acc > 0.7:
        print("  ✅ High K-means accuracy - features cluster by object presence")
    elif kmeans_acc > 0.6:
        print("  ⚠️  Moderate accuracy - some clustering by object presence")
    else:
        print("  ❌ Low accuracy - clustering doesn't match object presence")

    print()

    # Overall verdict
    print("Overall Verdict:")

    if silhouette > 0.5 and separation_ratio > 1.5:
        verdict = "✅ GOOD - Layer 8 features are semantically meaningful (separate object from background)"
        interpretation = "Features contain object-level semantic information"
    elif silhouette > 0.2 and separation_ratio > 1.0:
        verdict = "⚠️  MODERATE - Layer 8 features have some semantic structure but weak"
        interpretation = "Features contain some object information but also lots of texture/noise"
    else:
        verdict = "❌ POOR - Layer 8 features are mostly texture (no semantic object information)"
        interpretation = "Features are dominated by low-level texture, no high-level object understanding"

    print(f"  {verdict}")
    print(f"  {interpretation}")
    print()

    return {
        'verdict': verdict,
        'interpretation': interpretation,
        'silhouette_score': silhouette,
        'separation_ratio': separation_ratio,
        'kmeans_accuracy': kmeans_acc,
    }


def main():
    """Main analysis function"""

    # Setup
    device = get_device()

    # Load config
    config_path = 'multi_exit_yolos/configs/multi_exit_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_root = config['dataset']['root_path']
    image_size = config['dataset']['image_size']

    print(f"\n{'='*80}")
    print("Test 3: Feature Clustering Analysis")
    print(f"{'='*80}\n")
    print(f"Dataset: MOT17 Validation")
    print(f"Root path: {dataset_root}")
    print(f"Image size: {image_size}")
    print()

    # Load validation dataset
    print("Loading validation dataset...")
    val_dataset = MOT17DetectionDataset(
        root_path=dataset_root,
        sequences=VAL_SEQUENCES,
        image_size=image_size
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda batch: (
            torch.stack([b[0] for b in batch]),
            [b[1] for b in batch]
        )
    )

    print(f"Val dataset: {len(val_dataset)} images")
    print()

    # Load model (we only need the backbone, not the trained detection heads)
    print("Loading multi-exit YOLOS model...")
    model = build_multi_exit_yolos(config)

    # Load checkpoint (even though detection heads failed, backbone is still pretrained)
    checkpoint_path = 'multi_exit_yolos/results/multi_exit_training/20251020_165209/best_model.pt'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Checkpoint loaded")
    else:
        print(f"⚠ Warning: Checkpoint not found, using pretrained backbone only")

    model.to(device)
    model.eval()
    print()

    # Extract features for Layer 8
    features_dict = extract_features_for_patches(
        model, val_loader, device, layer=8, num_samples=500
    )

    # Analyze clustering
    results = analyze_feature_clustering(features_dict, layer=8)

    # Interpret results
    interpretation = interpret_clustering_results(results)

    # Combine results
    final_results = {
        'clustering_analysis': results,
        'interpretation': interpretation,
    }

    # Save results
    output_dir = Path('results/experiment_06_feature_clustering')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'feature_clustering_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")

    print("Feature clustering analysis complete")

    return final_results


if __name__ == "__main__":
    results = main()
