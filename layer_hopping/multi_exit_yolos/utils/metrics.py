"""
Evaluation Metrics for Object Detection

Implements common detection metrics:
- mAP (Mean Average Precision)
- AP50, AP75
- Precision, Recall, F1 Score
"""

import torch
import numpy as np
from typing import List, Dict, Tuple


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes

    Args:
        boxes1: [N, 4] boxes in format [cx, cy, w, h], normalized
        boxes2: [M, 4] boxes in format [cx, cy, w, h], normalized

    Returns:
        [N, M] IoU matrix
    """
    # Convert to [x1, y1, x2, y2] format
    def box_cxcywh_to_xyxy(boxes):
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)

    boxes1_xyxy = box_cxcywh_to_xyxy(boxes1)
    boxes2_xyxy = box_cxcywh_to_xyxy(boxes2)

    # Compute IoU
    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])

    # Intersection
    lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Union
    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou


def compute_precision_recall(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.5
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score

    Args:
        predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
        targets: List of target dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for matching
        conf_threshold: Confidence threshold for predictions

    Returns:
        (precision, recall, f1_score)
    """
    total_tp = 0  # True positives
    total_fp = 0  # False positives
    total_fn = 0  # False negatives

    for pred, target in zip(predictions, targets):
        # Filter by confidence
        if len(pred['scores']) == 0:
            pred_boxes = torch.zeros((0, 4))
            pred_scores = torch.zeros(0)
        else:
            conf_mask = pred['scores'] >= conf_threshold
            pred_boxes = pred['boxes'][conf_mask]
            pred_scores = pred['scores'][conf_mask]

        target_boxes = target['boxes']

        # Count predictions and targets
        num_preds = len(pred_boxes)
        num_targets = len(target_boxes)

        if num_preds == 0 and num_targets == 0:
            continue
        elif num_preds == 0:
            total_fn += num_targets
            continue
        elif num_targets == 0:
            total_fp += num_preds
            continue

        # Compute IoU matrix
        iou_matrix = box_iou(pred_boxes, target_boxes)

        # Match predictions to targets (greedy matching)
        matched_targets = set()
        matched_preds = set()

        # Sort predictions by confidence
        sorted_indices = torch.argsort(pred_scores, descending=True)

        for pred_idx in sorted_indices:
            ious = iou_matrix[pred_idx]
            max_iou, max_target_idx = ious.max(dim=0)

            if max_iou >= iou_threshold and max_target_idx.item() not in matched_targets:
                # True positive
                total_tp += 1
                matched_targets.add(max_target_idx.item())
                matched_preds.add(pred_idx.item())
            else:
                # False positive
                total_fp += 1

        # Count false negatives (unmatched targets)
        total_fn += num_targets - len(matched_targets)

    # Compute metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1_score


def compute_ap(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5
) -> float:
    """
    Compute Average Precision (AP) at a given IoU threshold

    OPTIMIZED VERSION: Properly tracks image indices to avoid O(n³) complexity

    Args:
        predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
        targets: List of target dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for matching

    Returns:
        AP score
    """
    # Collect all predictions with image indices
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_image_ids = []

    for img_idx, pred in enumerate(predictions):
        if len(pred['boxes']) > 0:
            all_pred_boxes.append(pred['boxes'])
            all_pred_scores.append(pred['scores'])
            # Track which image each prediction belongs to
            all_pred_image_ids.extend([img_idx] * len(pred['boxes']))

    if len(all_pred_boxes) == 0:
        return 0.0

    # Concatenate all predictions
    all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
    all_pred_scores = torch.cat(all_pred_scores, dim=0)
    all_pred_image_ids = torch.tensor(all_pred_image_ids)

    # Sort by confidence
    sorted_indices = torch.argsort(all_pred_scores, descending=True)
    all_pred_boxes = all_pred_boxes[sorted_indices]
    all_pred_scores = all_pred_scores[sorted_indices]
    all_pred_image_ids = all_pred_image_ids[sorted_indices]

    # Total number of ground truth boxes
    num_gt = sum(len(target['boxes']) for target in targets)

    if num_gt == 0:
        return 0.0

    # Compute precision-recall curve
    tp = torch.zeros(len(all_pred_boxes))
    fp = torch.zeros(len(all_pred_boxes))

    matched_targets = [set() for _ in targets]

    # Process each prediction
    for i, (pred_box, pred_score, img_idx) in enumerate(zip(all_pred_boxes, all_pred_scores, all_pred_image_ids)):
        img_idx = img_idx.item()
        target_boxes = targets[img_idx]['boxes']

        if len(target_boxes) == 0:
            # No targets in this image - false positive
            fp[i] = 1
            continue

        # Compute IoU with all targets in the same image
        ious = box_iou(pred_box.unsqueeze(0), target_boxes).squeeze(0)
        max_iou, max_target_idx = ious.max(dim=0)

        # Check if match
        if max_iou >= iou_threshold and max_target_idx.item() not in matched_targets[img_idx]:
            tp[i] = 1
            matched_targets[img_idx].add(max_target_idx.item())
        else:
            fp[i] = 1

    # Compute precision and recall
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)

    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Compute AP (area under precision-recall curve)
    # Use 11-point interpolation
    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        if torch.sum(recalls >= t) == 0:
            p = 0
        else:
            p = torch.max(precisions[recalls >= t])
        ap += p / 11

    return ap.item()


def compute_map(
    predictions: List[Dict],
    targets: List[Dict],
    iou_thresholds: List[float] = None
) -> Dict[str, float]:
    """
    Compute mAP and other AP metrics

    Args:
        predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
        targets: List of target dicts with 'boxes', 'labels'
        iou_thresholds: List of IoU thresholds (default: [0.5, 0.55, ..., 0.95])

    Returns:
        Dictionary with mAP, AP50, AP75
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5 to 0.95

    # Compute AP at each threshold
    aps = []
    for iou_thresh in iou_thresholds:
        ap = compute_ap(predictions, targets, iou_thresh)
        aps.append(ap)

    # mAP is average over all thresholds
    map_score = np.mean(aps)

    # AP50 and AP75
    ap50 = compute_ap(predictions, targets, iou_threshold=0.5)
    ap75 = compute_ap(predictions, targets, iou_threshold=0.75)

    return {
        'mAP': map_score,
        'AP50': ap50,
        'AP75': ap75
    }


def evaluate_model(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Comprehensive evaluation of detection model

    Args:
        predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
        targets: List of target dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for matching
        conf_threshold: Confidence threshold for predictions

    Returns:
        Dictionary with all metrics
    """
    # Compute mAP metrics
    map_metrics = compute_map(predictions, targets)

    # Compute precision/recall/F1
    precision, recall, f1_score = compute_precision_recall(
        predictions, targets, iou_threshold, conf_threshold
    )

    return {
        'mAP': map_metrics['mAP'],
        'AP50': map_metrics['AP50'],
        'AP75': map_metrics['AP75'],
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
