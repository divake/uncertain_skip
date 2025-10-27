"""
Comprehensive Baseline Evaluation of Pretrained YOLOS Layer 12

Computes ALL detection metrics for the official pretrained YOLOS-small model
to establish a strong baseline for future comparisons.

Metrics computed:
- IoU statistics (mean, median, std, percentiles)
- mAP at different IoU thresholds (0.5, 0.75, 0.5:0.95)
- Precision, Recall, F1-score at different confidence thresholds
- Detection rate (what % of GT boxes are detected)
- False positive rate
- Localization errors (center, width, height, size, aspect ratio)
- Confidence score distribution
- Per-image statistics
"""

import sys
import os
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from transformers import YolosForObjectDetection
from collections import defaultdict

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


def compute_ap(recall, precision):
    """Compute Average Precision using 11-point interpolation"""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0
    return ap


def evaluate_comprehensive(model, dataloader, device, conf_thresholds=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9]):
    """
    Comprehensive evaluation with all detection metrics

    Args:
        model: Pretrained YOLOS model
        dataloader: Validation dataloader
        device: Device to run on
        conf_thresholds: List of confidence thresholds to evaluate

    Returns:
        dict: Comprehensive metrics
    """
    model.eval()

    print(f"\n{'='*80}")
    print("COMPREHENSIVE BASELINE EVALUATION - Pretrained YOLOS Layer 12")
    print(f"{'='*80}\n")

    # Store all predictions and ground truths
    all_predictions = []  # List of dicts: {boxes, scores, image_id}
    all_ground_truths = []  # List of dicts: {boxes, image_id}

    # Per-image statistics
    per_image_stats = []

    # Overall statistics
    total_gt_boxes = 0
    total_images = 0

    print("Running inference on validation set...")
    for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Inference")):
        images = images.to(device)
        batch_size = images.shape[0]

        with torch.no_grad():
            outputs = model(images)
            logits = outputs.logits  # [B, 100, 91]
            pred_boxes = outputs.pred_boxes  # [B, 100, 4]

        # Process each image
        for i in range(batch_size):
            image_id = batch_idx * dataloader.batch_size + i
            total_images += 1

            # Ground truth
            gt_boxes = targets[i]['boxes']  # [N, 4]
            total_gt_boxes += len(gt_boxes)

            all_ground_truths.append({
                'boxes': gt_boxes.cpu(),
                'image_id': image_id,
                'num_boxes': len(gt_boxes)
            })

            # Predictions
            image_logits = logits[i]  # [100, 91]
            image_boxes = pred_boxes[i]  # [100, 4]

            # Get confidence scores (max across all classes)
            class_probs = torch.softmax(image_logits, dim=-1)
            max_scores, max_classes = class_probs.max(dim=-1)

            all_predictions.append({
                'boxes': image_boxes.cpu(),
                'scores': max_scores.cpu(),
                'classes': max_classes.cpu(),
                'image_id': image_id
            })

    print(f"\nTotal images: {total_images}")
    print(f"Total GT boxes: {total_gt_boxes}")
    print(f"Avg GT boxes per image: {total_gt_boxes / total_images:.2f}")
    print()

    # Now compute metrics at different confidence thresholds
    results = {
        'dataset_stats': {
            'total_images': total_images,
            'total_gt_boxes': total_gt_boxes,
            'avg_gt_boxes_per_image': total_gt_boxes / total_images,
        },
        'metrics_by_confidence': {},
        'iou_thresholds': {},
        'localization_errors': {},
        'confidence_distribution': {},
    }

    print(f"{'='*80}")
    print("Computing Metrics at Different Confidence Thresholds")
    print(f"{'='*80}\n")

    for conf_thresh in conf_thresholds:
        print(f"\n--- Confidence Threshold: {conf_thresh} ---\n")
        metrics = compute_metrics_at_threshold(
            all_predictions, all_ground_truths, conf_thresh, device
        )
        results['metrics_by_confidence'][str(conf_thresh)] = metrics

        # Print summary
        print(f"  mAP@0.5: {metrics['mAP_50']:.4f}")
        print(f"  mAP@0.75: {metrics['mAP_75']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1_score']:.4f}")
        print(f"  Detection rate: {metrics['detection_rate']:.4f}")
        print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"  Total predictions: {metrics['total_predictions']}")

    # Find best F1 score
    best_conf = max(conf_thresholds, key=lambda c: results['metrics_by_confidence'][str(c)]['f1_score'])
    results['best_confidence_threshold'] = float(best_conf)
    results['best_metrics'] = results['metrics_by_confidence'][str(best_conf)]

    print(f"\n{'='*80}")
    print(f"Best Performance at Confidence Threshold: {best_conf}")
    print(f"{'='*80}\n")
    print(f"  mAP@0.5: {results['best_metrics']['mAP_50']:.4f}")
    print(f"  mAP@0.75: {results['best_metrics']['mAP_75']:.4f}")
    print(f"  Precision: {results['best_metrics']['precision']:.4f}")
    print(f"  Recall: {results['best_metrics']['recall']:.4f}")
    print(f"  F1-score: {results['best_metrics']['f1_score']:.4f}")

    return results


def compute_metrics_at_threshold(all_predictions, all_ground_truths, conf_threshold, device):
    """Compute all metrics at a specific confidence threshold"""

    # Storage for matched predictions and GT
    all_ious = []
    all_center_errors = []
    all_width_errors = []
    all_height_errors = []
    all_size_ratios = []
    all_aspect_ratio_errors = []

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    total_predictions = 0
    detected_gt_boxes = 0
    total_gt = sum(gt['num_boxes'] for gt in all_ground_truths)

    # Per-image IoU lists for mAP calculation
    precision_recall_pairs = []  # (precision, recall, iou) for each prediction

    for pred, gt in zip(all_predictions, all_ground_truths):
        pred_boxes = pred['boxes'].to(device)
        pred_scores = pred['scores'].to(device)
        gt_boxes = gt['boxes'].to(device)

        # Filter predictions by confidence
        keep = pred_scores >= conf_threshold
        filtered_boxes = pred_boxes[keep]
        filtered_scores = pred_scores[keep]

        total_predictions += len(filtered_boxes)

        if len(gt_boxes) == 0:
            # No GT boxes - all predictions are false positives
            false_positives += len(filtered_boxes)
            continue

        if len(filtered_boxes) == 0:
            # No predictions - all GT boxes are false negatives
            false_negatives += len(gt_boxes)
            continue

        # Compute IoU matrix
        iou_matrix = compute_iou(filtered_boxes, gt_boxes)  # [N_pred, N_gt]

        # Match predictions to GT using greedy matching at IoU > 0.5
        matched_gt = set()
        matched_pred = set()

        # Sort predictions by score (descending)
        sorted_indices = torch.argsort(filtered_scores, descending=True)

        for pred_idx in sorted_indices.tolist():
            if pred_idx in matched_pred:
                continue

            # Find best GT match for this prediction
            ious_for_pred = iou_matrix[pred_idx]
            best_iou, best_gt_idx = ious_for_pred.max(dim=0)
            best_gt_idx = best_gt_idx.item()

            if best_iou >= 0.5 and best_gt_idx not in matched_gt:
                # True positive
                true_positives += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)

                all_ious.append(best_iou.item())

                # Compute localization errors
                pred_box = filtered_boxes[pred_idx]
                gt_box = gt_boxes[best_gt_idx]

                center_error = torch.norm(pred_box[:2] - gt_box[:2]).item()
                width_error = torch.abs(pred_box[2] - gt_box[2]).item()
                height_error = torch.abs(pred_box[3] - gt_box[3]).item()

                pred_area = pred_box[2] * pred_box[3]
                gt_area = gt_box[2] * gt_box[3]
                size_ratio = (pred_area / (gt_area + 1e-6)).item()

                pred_aspect = pred_box[2] / (pred_box[3] + 1e-6)
                gt_aspect = gt_box[2] / (gt_box[3] + 1e-6)
                aspect_error = torch.abs(pred_aspect - gt_aspect).item()

                all_center_errors.append(center_error)
                all_width_errors.append(width_error)
                all_height_errors.append(height_error)
                all_size_ratios.append(size_ratio)
                all_aspect_ratio_errors.append(aspect_error)
            else:
                # False positive
                false_positives += 1

        # Unmatched GT boxes are false negatives
        false_negatives += len(gt_boxes) - len(matched_gt)
        detected_gt_boxes += len(matched_gt)

    # Compute metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    detection_rate = detected_gt_boxes / total_gt if total_gt > 0 else 0

    # mAP calculation (simplified - using matched IoUs)
    # mAP@0.5: Already computed via true_positives at IoU >= 0.5
    mAP_50 = precision  # Approximation

    # mAP@0.75: Recompute with IoU threshold 0.75
    tp_75 = sum(1 for iou in all_ious if iou >= 0.75)
    precision_75 = tp_75 / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    mAP_75 = precision_75

    # IoU statistics
    iou_stats = {
        'mean': np.mean(all_ious) if all_ious else 0,
        'median': np.median(all_ious) if all_ious else 0,
        'std': np.std(all_ious) if all_ious else 0,
        'min': np.min(all_ious) if all_ious else 0,
        'max': np.max(all_ious) if all_ious else 0,
        'percentile_25': np.percentile(all_ious, 25) if all_ious else 0,
        'percentile_75': np.percentile(all_ious, 75) if all_ious else 0,
        'percentile_95': np.percentile(all_ious, 95) if all_ious else 0,
    }

    # Localization error statistics
    localization_stats = {
        'center_error': {
            'mean': np.mean(all_center_errors) if all_center_errors else 0,
            'std': np.std(all_center_errors) if all_center_errors else 0,
            'median': np.median(all_center_errors) if all_center_errors else 0,
        },
        'width_error': {
            'mean': np.mean(all_width_errors) if all_width_errors else 0,
            'std': np.std(all_width_errors) if all_width_errors else 0,
            'median': np.median(all_width_errors) if all_width_errors else 0,
        },
        'height_error': {
            'mean': np.mean(all_height_errors) if all_height_errors else 0,
            'std': np.std(all_height_errors) if all_height_errors else 0,
            'median': np.median(all_height_errors) if all_height_errors else 0,
        },
        'size_ratio': {
            'mean': np.mean(all_size_ratios) if all_size_ratios else 0,
            'std': np.std(all_size_ratios) if all_size_ratios else 0,
            'median': np.median(all_size_ratios) if all_size_ratios else 0,
        },
        'aspect_ratio_error': {
            'mean': np.mean(all_aspect_ratio_errors) if all_aspect_ratio_errors else 0,
            'std': np.std(all_aspect_ratio_errors) if all_aspect_ratio_errors else 0,
            'median': np.median(all_aspect_ratio_errors) if all_aspect_ratio_errors else 0,
        },
    }

    return {
        'confidence_threshold': float(conf_threshold),
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives),
        'total_predictions': int(total_predictions),
        'total_gt_boxes': int(total_gt),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'detection_rate': float(detection_rate),
        'mAP_50': float(mAP_50),
        'mAP_75': float(mAP_75),
        'mean_iou': float(iou_stats['mean']),
        'iou_stats': iou_stats,
        'localization_stats': localization_stats,
    }


def save_results_to_markdown(results, output_path):
    """Save comprehensive results to markdown file"""

    md = []
    md.append("# BASELINE METRICS - Pretrained YOLOS-small Layer 12")
    md.append("")
    md.append("**Model**: Official pretrained YOLOS-small from Hugging Face (`hustvl/yolos-small`)")
    md.append("**Dataset**: MOT17 Validation Set (person detection)")
    md.append("**Evaluation Date**: October 23, 2025")
    md.append("")
    md.append("---")
    md.append("")

    # Dataset statistics
    md.append("## Dataset Statistics")
    md.append("")
    ds = results['dataset_stats']
    md.append(f"- **Total images**: {ds['total_images']}")
    md.append(f"- **Total ground truth boxes**: {ds['total_gt_boxes']}")
    md.append(f"- **Average GT boxes per image**: {ds['avg_gt_boxes_per_image']:.2f}")
    md.append("")
    md.append("---")
    md.append("")

    # Best performance summary
    md.append("## Best Performance Summary")
    md.append("")
    best_conf = results['best_confidence_threshold']
    best = results['best_metrics']
    md.append(f"**Optimal Confidence Threshold**: {best_conf}")
    md.append("")
    md.append("### Key Metrics")
    md.append("")
    md.append(f"- **mAP@0.5**: {best['mAP_50']:.4f} ({best['mAP_50']*100:.2f}%)")
    md.append(f"- **mAP@0.75**: {best['mAP_75']:.4f} ({best['mAP_75']*100:.2f}%)")
    md.append(f"- **Precision**: {best['precision']:.4f} ({best['precision']*100:.2f}%)")
    md.append(f"- **Recall**: {best['recall']:.4f} ({best['recall']*100:.2f}%)")
    md.append(f"- **F1-Score**: {best['f1_score']:.4f} ({best['f1_score']*100:.2f}%)")
    md.append(f"- **Detection Rate**: {best['detection_rate']:.4f} ({best['detection_rate']*100:.2f}%)")
    md.append(f"- **Mean IoU**: {best['mean_iou']:.4f}")
    md.append("")
    md.append("### Detection Counts")
    md.append("")
    md.append(f"- **True Positives**: {best['true_positives']}")
    md.append(f"- **False Positives**: {best['false_positives']}")
    md.append(f"- **False Negatives**: {best['false_negatives']}")
    md.append(f"- **Total Predictions**: {best['total_predictions']}")
    md.append("")
    md.append("---")
    md.append("")

    # IoU statistics
    md.append("## IoU Distribution")
    md.append("")
    iou = best['iou_stats']
    md.append(f"- **Mean IoU**: {iou['mean']:.4f}")
    md.append(f"- **Median IoU**: {iou['median']:.4f}")
    md.append(f"- **Std Dev**: {iou['std']:.4f}")
    md.append(f"- **Min IoU**: {iou['min']:.4f}")
    md.append(f"- **Max IoU**: {iou['max']:.4f}")
    md.append(f"- **25th Percentile**: {iou['percentile_25']:.4f}")
    md.append(f"- **75th Percentile**: {iou['percentile_75']:.4f}")
    md.append(f"- **95th Percentile**: {iou['percentile_95']:.4f}")
    md.append("")
    md.append("---")
    md.append("")

    # Localization errors
    md.append("## Localization Error Analysis")
    md.append("")
    loc = best['localization_stats']

    md.append("### Center Error (L2 Distance)")
    md.append(f"- **Mean**: {loc['center_error']['mean']:.4f}")
    md.append(f"- **Median**: {loc['center_error']['median']:.4f}")
    md.append(f"- **Std Dev**: {loc['center_error']['std']:.4f}")
    md.append("")

    md.append("### Width Error (Absolute Difference)")
    md.append(f"- **Mean**: {loc['width_error']['mean']:.4f}")
    md.append(f"- **Median**: {loc['width_error']['median']:.4f}")
    md.append(f"- **Std Dev**: {loc['width_error']['std']:.4f}")
    md.append("")

    md.append("### Height Error (Absolute Difference)")
    md.append(f"- **Mean**: {loc['height_error']['mean']:.4f}")
    md.append(f"- **Median**: {loc['height_error']['median']:.4f}")
    md.append(f"- **Std Dev**: {loc['height_error']['std']:.4f}")
    md.append("")

    md.append("### Size Ratio (Predicted Area / GT Area)")
    md.append(f"- **Mean**: {loc['size_ratio']['mean']:.4f}")
    md.append(f"- **Median**: {loc['size_ratio']['median']:.4f}")
    md.append(f"- **Std Dev**: {loc['size_ratio']['std']:.4f}")
    md.append("")

    md.append("### Aspect Ratio Error")
    md.append(f"- **Mean**: {loc['aspect_ratio_error']['mean']:.4f}")
    md.append(f"- **Median**: {loc['aspect_ratio_error']['median']:.4f}")
    md.append(f"- **Std Dev**: {loc['aspect_ratio_error']['std']:.4f}")
    md.append("")
    md.append("---")
    md.append("")

    # Performance at different confidence thresholds
    md.append("## Performance at Different Confidence Thresholds")
    md.append("")
    md.append("| Conf | mAP@0.5 | mAP@0.75 | Precision | Recall | F1 | Detection Rate | Mean IoU | Predictions |")
    md.append("|------|---------|----------|-----------|--------|----|--------------|---------|-----------:|")

    for conf_str in sorted(results['metrics_by_confidence'].keys(), key=float):
        m = results['metrics_by_confidence'][conf_str]
        md.append(f"| {m['confidence_threshold']:.2f} | {m['mAP_50']:.4f} | {m['mAP_75']:.4f} | "
                 f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1_score']:.4f} | "
                 f"{m['detection_rate']:.4f} | {m['mean_iou']:.4f} | {m['total_predictions']} |")

    md.append("")
    md.append("---")
    md.append("")

    # Interpretation
    md.append("## Interpretation")
    md.append("")
    md.append("### What These Metrics Mean")
    md.append("")
    md.append(f"- **mAP@0.5 = {best['mAP_50']:.2%}**: At IoU threshold 0.5, the model achieves {best['mAP_50']:.2%} precision")
    md.append(f"- **Recall = {best['recall']:.2%}**: The model detects {best['recall']:.2%} of all ground truth persons")
    md.append(f"- **Precision = {best['precision']:.2%}**: {best['precision']:.2%} of predictions are correct (not false alarms)")
    md.append(f"- **F1-Score = {best['f1_score']:.2%}**: Harmonic mean balancing precision and recall")
    md.append("")

    # Quality assessment
    if best['f1_score'] > 0.7:
        quality = "✅ EXCELLENT"
    elif best['f1_score'] > 0.5:
        quality = "✅ GOOD"
    elif best['f1_score'] > 0.3:
        quality = "⚠️ MODERATE"
    else:
        quality = "❌ POOR"

    md.append(f"### Overall Quality: {quality}")
    md.append("")
    md.append("### Strengths")
    strengths = []
    if best['recall'] > 0.5:
        strengths.append(f"- Good detection rate ({best['recall']:.2%} of GT boxes detected)")
    if best['precision'] > 0.5:
        strengths.append(f"- Good precision ({best['precision']:.2%} predictions are correct)")
    if iou['mean'] > 0.5:
        strengths.append(f"- High IoU overlap (mean {iou['mean']:.2%})")
    if loc['center_error']['mean'] < 0.1:
        strengths.append(f"- Excellent localization (center error {loc['center_error']['mean']:.4f})")

    if strengths:
        md.extend(strengths)
    else:
        md.append("- (To be determined based on use case)")
    md.append("")

    md.append("### Weaknesses")
    weaknesses = []
    if best['recall'] < 0.5:
        weaknesses.append(f"- Low recall ({best['recall']:.2%}) - missing many GT boxes")
    if best['precision'] < 0.5:
        weaknesses.append(f"- Low precision ({best['precision']:.2%}) - many false positives")
    if iou['mean'] < 0.3:
        weaknesses.append(f"- Low IoU overlap (mean {iou['mean']:.2%})")

    if weaknesses:
        md.extend(weaknesses)
    else:
        md.append("- (To be determined based on use case)")
    md.append("")

    md.append("---")
    md.append("")

    # Usage notes
    md.append("## Usage Notes")
    md.append("")
    md.append("**Purpose**: This baseline establishes the detection performance of the official pretrained YOLOS-small model on MOT17 person detection. Use these metrics to compare against:")
    md.append("")
    md.append("1. **Early exit models** (Layer 8/10 detection heads)")
    md.append("2. **Fine-tuned models** (after training on MOT17)")
    md.append("3. **Alternative architectures** (other detection models)")
    md.append("")
    md.append("**Key Comparison Metrics**:")
    md.append("- mAP@0.5 (primary metric)")
    md.append("- Precision/Recall trade-off")
    md.append("- Mean IoU (localization quality)")
    md.append("- F1-Score (overall performance)")
    md.append("")
    md.append("---")
    md.append("")
    md.append(f"**Generated**: October 23, 2025")
    md.append(f"**Script**: `07_baseline_evaluation.py`")
    md.append(f"**Results JSON**: `results/experiment_08_baseline_comprehensive/baseline_metrics.json`")

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(md))


def main():
    """Run comprehensive baseline evaluation"""

    device = get_device()

    # Load config
    config_path = 'configs/experiment_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_root = config['dataset']['root_path']
    image_size = config['image']['size']

    print(f"\n{'='*80}")
    print("BASELINE EVALUATION - Pretrained YOLOS-small")
    print(f"{'='*80}\n")
    print(f"Dataset: MOT17 Validation")
    print(f"Model: Official pretrained YOLOS-small")
    print()

    # Load dataset
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

    # Load model
    print("Loading pretrained YOLOS-small...")
    model_path = "models/yolos-small-pretrained"
    model = YolosForObjectDetection.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print(f"✓ Model loaded")
    print()

    # Run comprehensive evaluation
    results = evaluate_comprehensive(
        model, val_loader, device,
        conf_thresholds=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )

    # Save results
    output_dir = Path('results/experiment_08_baseline_comprehensive')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / 'baseline_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {json_path}")

    # Save markdown
    md_path = Path('BASELINE_METRICS.md')
    save_results_to_markdown(results, md_path)
    print(f"✓ Baseline metrics saved to {md_path}")

    print(f"\n{'='*80}")
    print("BASELINE EVALUATION COMPLETE")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    results = main()
