"""
Re-run Analysis with Official Pretrained YOLOS Model

Tests Layer 8 vs Layer 12 features and detection performance using the
official pretrained YOLOS-small model from Hugging Face.

Key differences from previous analysis:
- Uses PRETRAINED detection head at Layer 12 (not randomly initialized)
- Uses PRETRAINED backbone features at all layers
- Compares Layer 8 (early, no detection head) vs Layer 12 (pretrained detection head)
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
from transformers import YolosForObjectDetection

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.data_loader import MOT17DetectionDataset
from src.architecture_constants import VAL_SEQUENCES
from src.utils import get_device


def compute_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    Boxes in [cx, cy, w, h] format (normalized).

    Args:
        boxes1: [N, 4] tensor
        boxes2: [M, 4] tensor

    Returns:
        iou: [N, M] tensor
    """
    # Convert to [x1, y1, x2, y2]
    boxes1_x1y1x2y2 = torch.zeros_like(boxes1)
    boxes1_x1y1x2y2[:, 0] = boxes1[:, 0] - boxes1[:, 2] / 2
    boxes1_x1y1x2y2[:, 1] = boxes1[:, 1] - boxes1[:, 3] / 2
    boxes1_x1y1x2y2[:, 2] = boxes1[:, 0] + boxes1[:, 2] / 2
    boxes1_x1y1x2y2[:, 3] = boxes1[:, 1] + boxes1[:, 3] / 2

    boxes2_x1y1x2y2 = torch.zeros_like(boxes2)
    boxes2_x1y1x2y2[:, 0] = boxes2[:, 0] - boxes2[:, 2] / 2
    boxes2_x1y1x2y2[:, 1] = boxes2[:, 1] - boxes2[:, 3] / 2
    boxes2_x1y1x2y2[:, 2] = boxes2[:, 0] + boxes2[:, 2] / 2
    boxes2_x1y1x2y2[:, 3] = boxes2[:, 1] + boxes2[:, 3] / 2

    # Compute intersection
    x1 = torch.max(boxes1_x1y1x2y2[:, None, 0], boxes2_x1y1x2y2[None, :, 0])
    y1 = torch.max(boxes1_x1y1x2y2[:, None, 1], boxes2_x1y1x2y2[None, :, 1])
    x2 = torch.min(boxes1_x1y1x2y2[:, None, 2], boxes2_x1y1x2y2[None, :, 2])
    y2 = torch.min(boxes1_x1y1x2y2[:, None, 3], boxes2_x1y1x2y2[None, :, 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Compute union
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    union = area1[:, None] + area2[None, :] - intersection

    iou = intersection / (union + 1e-6)
    return iou


def evaluate_layer12_detection(model, dataloader, device, conf_threshold=0.5):
    """
    Evaluate Layer 12 detection performance using PRETRAINED detection head.

    Args:
        model: Pretrained YolosForObjectDetection
        dataloader: Validation dataloader
        device: Device to run on
        conf_threshold: Confidence threshold for filtering predictions

    Returns:
        dict: Localization metrics
    """
    model.eval()

    all_center_errors = []
    all_width_errors = []
    all_height_errors = []
    all_size_ratios = []
    all_aspect_ratio_errors = []
    all_best_ious = []

    num_images = 0
    total_predictions = 0
    total_gt = 0

    print(f"\n{'='*80}")
    print(f"Evaluating Layer 12 Detection (Pretrained Head)")
    print(f"{'='*80}\n")

    for images, targets in tqdm(dataloader, desc="Layer 12"):
        images = images.to(device)
        batch_size = images.shape[0]

        with torch.no_grad():
            # Get Layer 12 predictions from pretrained model
            outputs = model(images)

            # Extract predictions
            logits = outputs.logits  # [B, 100, 91] - 91 COCO classes
            pred_boxes = outputs.pred_boxes  # [B, 100, 4] in [cx, cy, w, h] format

        # For each image in batch
        for i in range(batch_size):
            num_images += 1

            # Get ground truth
            gt_boxes = targets[i]['boxes'].to(device)  # [N, 4]
            total_gt += len(gt_boxes)

            if len(gt_boxes) == 0:
                continue

            # Get predictions for this image
            image_logits = logits[i]  # [100, 91]
            image_boxes = pred_boxes[i]  # [100, 4]

            # YOLOS trained on COCO - find person class index
            # COCO classes: 0-90, where person is typically class 0
            # But YOLOS outputs are logits over 91 classes
            # We need to use softmax across all classes, then take max
            class_probs = torch.softmax(image_logits, dim=-1)  # [100, 91]

            # Get max confidence score across all classes for each detection token
            max_scores, max_classes = class_probs.max(dim=-1)  # [100]

            # Use max scores regardless of class (we'll filter later if needed)
            person_scores = max_scores

            # Filter by confidence
            keep = person_scores >= conf_threshold
            filtered_boxes = image_boxes[keep]
            filtered_scores = person_scores[keep]

            total_predictions += len(filtered_boxes)

            if len(filtered_boxes) == 0:
                continue

            # Compute IoU between predictions and GT
            iou_matrix = compute_iou(filtered_boxes, gt_boxes)  # [N_pred, N_gt]

            # For each prediction, find best matching GT box
            best_ious, best_gt_indices = iou_matrix.max(dim=1)

            # Only consider predictions that have some overlap (IoU > 0.1)
            valid_matches = best_ious > 0.1

            if valid_matches.sum() == 0:
                continue

            # Get matched predictions and GT boxes
            pred_matched = filtered_boxes[valid_matches]
            gt_matched = gt_boxes[best_gt_indices[valid_matches]]
            matched_ious = best_ious[valid_matches]

            # Compute localization errors
            # Center error (L2 distance between centers)
            pred_centers = pred_matched[:, :2]
            gt_centers = gt_matched[:, :2]
            center_errors = torch.norm(pred_centers - gt_centers, dim=1)

            # Width/height errors (absolute difference)
            width_errors = torch.abs(pred_matched[:, 2] - gt_matched[:, 2])
            height_errors = torch.abs(pred_matched[:, 3] - gt_matched[:, 3])

            # Size ratio (predicted area / GT area)
            pred_areas = pred_matched[:, 2] * pred_matched[:, 3]
            gt_areas = gt_matched[:, 2] * gt_matched[:, 3]
            size_ratios = pred_areas / (gt_areas + 1e-6)

            # Aspect ratio error
            pred_aspect = pred_matched[:, 2] / (pred_matched[:, 3] + 1e-6)
            gt_aspect = gt_matched[:, 2] / (gt_matched[:, 3] + 1e-6)
            aspect_ratio_errors = torch.abs(pred_aspect - gt_aspect)

            # Store metrics
            all_center_errors.extend(center_errors.cpu().tolist())
            all_width_errors.extend(width_errors.cpu().tolist())
            all_height_errors.extend(height_errors.cpu().tolist())
            all_size_ratios.extend(size_ratios.cpu().tolist())
            all_aspect_ratio_errors.extend(aspect_ratio_errors.cpu().tolist())
            all_best_ious.extend(matched_ious.cpu().tolist())

    # Compute statistics
    results = {
        'layer': 12,
        'num_images': num_images,
        'total_predictions': total_predictions,
        'total_gt': total_gt,
        'predictions_per_image': total_predictions / num_images if num_images > 0 else 0,
        'center_error_mean': np.mean(all_center_errors) if all_center_errors else 0,
        'center_error_std': np.std(all_center_errors) if all_center_errors else 0,
        'width_error_mean': np.mean(all_width_errors) if all_width_errors else 0,
        'width_error_std': np.std(all_width_errors) if all_width_errors else 0,
        'height_error_mean': np.mean(all_height_errors) if all_height_errors else 0,
        'height_error_std': np.std(all_height_errors) if all_height_errors else 0,
        'size_ratio_mean': np.mean(all_size_ratios) if all_size_ratios else 0,
        'size_ratio_std': np.std(all_size_ratios) if all_size_ratios else 0,
        'aspect_ratio_error_mean': np.mean(all_aspect_ratio_errors) if all_aspect_ratio_errors else 0,
        'aspect_ratio_error_std': np.std(all_aspect_ratio_errors) if all_aspect_ratio_errors else 0,
        'best_iou_mean': np.mean(all_best_ious) if all_best_ious else 0,
        'best_iou_std': np.std(all_best_ious) if all_best_ious else 0,
    }

    return results


def extract_layer8_features(model, dataloader, device, num_samples=500):
    """
    Extract Layer 8 features for in-box vs out-of-box patches.

    Args:
        model: Pretrained YolosForObjectDetection
        dataloader: Validation dataloader
        device: Device to run on
        num_samples: Number of samples per category

    Returns:
        dict: Features for in-box and out-of-box patches
    """
    model.eval()

    in_box_features = []
    out_of_box_features = []

    print(f"\n{'='*80}")
    print(f"Extracting Layer 8 Features from Pretrained Backbone")
    print(f"{'='*80}\n")

    for images, targets in tqdm(dataloader, desc="Extracting features"):
        if len(in_box_features) >= num_samples and len(out_of_box_features) >= num_samples:
            break

        images = images.to(device)
        batch_size = images.shape[0]

        with torch.no_grad():
            # Forward pass to get Layer 8 hidden states
            outputs = model.vit(images, output_hidden_states=True)

            # Get Layer 8 hidden states (index 8: 0=embeddings, 1-12=encoder layers)
            hidden_states = outputs.hidden_states[8]  # [B, 101, 384]

            # Remove CLS token
            patch_features = hidden_states[:, 1:, :]  # [B, 100, 384]

        # For each image in batch
        for i in range(batch_size):
            if len(in_box_features) >= num_samples and len(out_of_box_features) >= num_samples:
                break

            # Get ground truth boxes
            gt_boxes = targets[i]['boxes']  # [N, 4] in [cx, cy, w, h]

            if len(gt_boxes) == 0:
                continue

            # Patch features for this image
            features = patch_features[i].cpu().numpy()  # [100, 384]

            # YOLOS uses 10x10 grid of patches
            grid_size = 10

            # Check each patch
            for patch_idx in range(100):
                row = patch_idx // grid_size
                col = patch_idx % grid_size

                patch_cx = (col + 0.5) / grid_size
                patch_cy = (row + 0.5) / grid_size

                # Check if patch is inside any GT box
                is_inside = False
                for box in gt_boxes:
                    box_cx, box_cy, box_w, box_h = box.tolist()

                    if (abs(patch_cx - box_cx) < box_w / 2 and
                        abs(patch_cy - box_cy) < box_h / 2):
                        is_inside = True
                        break

                # Collect features
                if is_inside and len(in_box_features) < num_samples:
                    in_box_features.append(features[patch_idx])
                elif not is_inside and len(out_of_box_features) < num_samples:
                    out_of_box_features.append(features[patch_idx])

    in_box_features = np.array(in_box_features)
    out_of_box_features = np.array(out_of_box_features)

    print(f"\nCollected {len(in_box_features)} in-box features")
    print(f"Collected {len(out_of_box_features)} out-of-box features")

    return {
        'in_box_features': in_box_features,
        'out_of_box_features': out_of_box_features,
    }


def analyze_feature_clustering(features_dict, layer):
    """Analyze feature clustering quality"""

    in_box = features_dict['in_box_features']
    out_of_box = features_dict['out_of_box_features']

    print(f"\n{'='*80}")
    print(f"Layer {layer} - Feature Clustering Analysis")
    print(f"{'='*80}\n")

    # Combine features
    all_features = np.vstack([in_box, out_of_box])
    labels = np.array([0] * len(in_box) + [1] * len(out_of_box))

    print(f"Total samples: {len(all_features)}")
    print(f"Feature dimension: {all_features.shape[1]}")
    print()

    # PCA
    print("Running PCA...")
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(all_features)
    explained_variance = pca.explained_variance_ratio_[:10].sum()
    print(f"Variance explained by top 10 components: {explained_variance:.4f}")
    print()

    # K-means
    print("Running K-Means clustering...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_predictions = kmeans.fit_predict(features_pca)

    accuracy_1 = np.mean(cluster_predictions == labels)
    accuracy_2 = np.mean(cluster_predictions != labels)
    clustering_accuracy = max(accuracy_1, accuracy_2)
    print(f"K-Means accuracy: {clustering_accuracy:.4f}")
    print()

    # Silhouette score
    print("Computing silhouette score...")
    silhouette = silhouette_score(features_pca, labels)
    print(f"Silhouette score: {silhouette:.4f}")
    print()

    # Separation ratio
    print("Computing separation ratio...")
    in_box_pca = features_pca[labels == 0]
    out_of_box_pca = features_pca[labels == 1]

    in_box_center = in_box_pca.mean(axis=0)
    out_of_box_center = out_of_box_pca.mean(axis=0)

    within_in_box = np.linalg.norm(in_box_pca - in_box_center, axis=1).mean()
    within_out_of_box = np.linalg.norm(out_of_box_pca - out_of_box_center, axis=1).mean()
    within_group_dist = (within_in_box + within_out_of_box) / 2

    between_group_dist = np.linalg.norm(in_box_center - out_of_box_center)
    separation_ratio = between_group_dist / within_group_dist

    print(f"Within-group distance: {within_group_dist:.4f}")
    print(f"Between-group distance: {between_group_dist:.4f}")
    print(f"Separation ratio: {separation_ratio:.4f}")
    print()

    return {
        'layer': layer,
        'pca_variance_explained': float(explained_variance),
        'kmeans_accuracy': float(clustering_accuracy),
        'silhouette_score': float(silhouette),
        'within_group_distance': float(within_group_dist),
        'between_group_distance': float(between_group_dist),
        'separation_ratio': float(separation_ratio),
    }


def interpret_results(layer12_localization, layer8_clustering):
    """Interpret and compare results"""

    print(f"\n{'='*80}")
    print("INTERPRETATION: Layer 8 vs Layer 12 Comparison")
    print(f"{'='*80}\n")

    print("Layer 12 Detection Performance (Pretrained Head):")
    print(f"  Mean IoU: {layer12_localization['best_iou_mean']:.4f}")
    print(f"  Center error: {layer12_localization['center_error_mean']:.4f}")
    print(f"  Size ratio: {layer12_localization['size_ratio_mean']:.4f}")
    print(f"  Predictions per image: {layer12_localization['predictions_per_image']:.2f}")
    print()

    if layer12_localization['best_iou_mean'] > 0.5:
        print("  ✅ EXCELLENT - Pretrained Layer 12 works very well!")
    elif layer12_localization['best_iou_mean'] > 0.3:
        print("  ✅ GOOD - Pretrained Layer 12 achieves decent detection")
    else:
        print("  ⚠️  MODERATE - Pretrained Layer 12 has room for improvement")
    print()

    print("Layer 8 Feature Quality (Pretrained Backbone):")
    print(f"  Silhouette score: {layer8_clustering['silhouette_score']:.4f} (>0.5 = good)")
    print(f"  Separation ratio: {layer8_clustering['separation_ratio']:.4f} (>1.5 = good)")
    print(f"  K-means accuracy: {layer8_clustering['kmeans_accuracy']:.4f} (>0.7 = good)")
    print()

    if layer8_clustering['silhouette_score'] > 0.5 and layer8_clustering['separation_ratio'] > 1.5:
        verdict = "✅ GOOD - Layer 8 features ARE semantically meaningful"
        interpretation = "Layer 8 can likely support detection with proper training"
    elif layer8_clustering['silhouette_score'] > 0.2 and layer8_clustering['separation_ratio'] > 1.0:
        verdict = "⚠️  MODERATE - Layer 8 features have some semantic structure"
        interpretation = "Layer 8 might support detection but will be challenging"
    else:
        verdict = "❌ POOR - Layer 8 features are texture-dominated"
        interpretation = "Layer 8 likely cannot support good detection"

    print(f"  {verdict}")
    print(f"  {interpretation}")
    print()

    return {
        'layer12_verdict': 'good' if layer12_localization['best_iou_mean'] > 0.3 else 'moderate',
        'layer8_verdict': verdict,
        'layer8_interpretation': interpretation,
    }


def main():
    """Main analysis with pretrained model"""

    device = get_device()

    # Load config for dataset paths
    config_path = 'configs/experiment_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_root = config['dataset']['root_path']
    image_size = config['image']['size']

    print(f"\n{'='*80}")
    print("Analysis with Official Pretrained YOLOS-small")
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

    # Load pretrained model
    print("Loading pretrained YOLOS-small from HuggingFace...")
    model_path = "models/yolos-small-pretrained"
    model = YolosForObjectDetection.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print(f"✓ Model loaded from {model_path}")
    print()

    # Test 1: Layer 12 detection performance (pretrained head)
    # Use lower confidence threshold to see any predictions at all
    layer12_results = evaluate_layer12_detection(model, val_loader, device, conf_threshold=0.1)

    # Test 2: Layer 8 feature clustering (pretrained backbone)
    layer8_features = extract_layer8_features(model, val_loader, device, num_samples=500)
    layer8_clustering = analyze_feature_clustering(layer8_features, layer=8)

    # Interpret results
    interpretation = interpret_results(layer12_results, layer8_clustering)

    # Save results
    output_dir = Path('results/experiment_07_pretrained_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    final_results = {
        'layer12_localization': layer12_results,
        'layer8_clustering': layer8_clustering,
        'interpretation': interpretation,
    }

    output_file = output_dir / 'pretrained_model_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")

    print("Analysis complete!")

    return final_results


if __name__ == "__main__":
    results = main()
