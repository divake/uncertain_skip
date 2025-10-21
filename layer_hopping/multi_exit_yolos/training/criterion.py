"""
Detection Loss (SetCriterion) for Multi-Exit YOLOS

Adapted from official YOLOS/DETR implementation with support for auxiliary losses.
Computes losses for all exit layers (8, 10, 12) during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import box_ops
from utils.matcher import HungarianMatcher


class SetCriterion(nn.Module):
    """
    Loss computation for DETR-style detection

    Process:
    1. Hungarian matching between predictions and ground truth
    2. Compute losses for matched pairs (classification + bbox)
    3. If auxiliary outputs exist, repeat for intermediate layers

    Losses:
    - Classification loss (cross-entropy)
    - L1 bounding box loss
    - GIoU loss
    """

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float,
        losses: List[str]
    ):
        """
        Args:
            num_classes: Number of object categories (without no-object class)
            matcher: Hungarian matcher for bipartite matching
            weight_dict: Loss weights (e.g., {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2})
            eos_coef: Relative weight for no-object class
            losses: List of losses to compute ['labels', 'boxes', 'cardinality']
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        # Weight for empty/no-object class
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        Classification loss (NLL / Cross-Entropy)

        Args:
            outputs: Dict with 'pred_logits' [batch, num_queries, num_classes+1]
            targets: List of dicts with 'labels' [num_target_boxes]
            indices: Matched indices from Hungarian matcher
            num_boxes: Number of boxes for normalization
            log: Whether to log accuracy

        Returns:
            Dict with 'loss_ce' and optionally 'class_error'
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        # Get permutation indices
        idx = self._get_src_permutation_idx(indices)

        # Target classes: matched = actual class, unmatched = no-object
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,  # no-object class
            dtype=torch.int64,
            device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        # Cross-entropy loss
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            self.empty_weight
        )
        losses = {'loss_ce': loss_ce}

        if log:
            # Log classification accuracy
            losses['class_error'] = 100 - self._accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Cardinality error: absolute error in number of predicted objects

        This is for logging only, doesn't propagate gradients.
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device

        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)

        # Count predictions that are NOT no-object
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())

        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Bounding box losses: L1 + GIoU

        Args:
            outputs: Dict with 'pred_boxes' [batch, num_queries, 4]
            targets: List of dicts with 'boxes' [num_target_boxes, 4]
            indices: Matched indices
            num_boxes: Number of boxes for normalization

        Returns:
            Dict with 'loss_bbox' and 'loss_giou'
        """
        assert 'pred_boxes' in outputs

        idx = self._get_src_permutation_idx(indices)

        # Get matched predictions and targets
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # GIoU loss
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)
        ))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def _get_src_permutation_idx(self, indices):
        """Permute predictions following matched indices"""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """Permute targets following matched indices"""
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    @torch.no_grad()
    def _accuracy(self, output, target):
        """Compute classification accuracy"""
        pred = output.argmax(-1)
        acc = (pred == target).float().mean() * 100
        return [acc]

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """
        Dispatch to specific loss function

        Args:
            loss: Loss name ('labels', 'boxes', 'cardinality')
            outputs: Model outputs
            targets: Ground truth targets
            indices: Matched indices
            num_boxes: Number of boxes
            **kwargs: Additional arguments for specific losses

        Returns:
            Dict with computed losses
        """
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }

        assert loss in loss_map, f'Unknown loss: {loss}'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """
        Compute all losses

        Args:
            outputs: Model outputs dict with:
                - pred_logits: [batch, num_queries, num_classes+1]
                - pred_boxes: [batch, num_queries, 4]
                - aux_outputs: (optional) List of dicts for auxiliary exits

            targets: List of target dicts (len = batch_size) with:
                - labels: [num_objects] class labels
                - boxes: [num_objects, 4] bounding boxes (cx, cy, w, h) normalized

        Returns:
            Dict with all losses:
            - loss_ce, loss_bbox, loss_giou (main output)
            - loss_ce_0, loss_bbox_0, loss_giou_0 (auxiliary output 0 - Layer 8)
            - loss_ce_1, loss_bbox_1, loss_giou_1 (auxiliary output 1 - Layer 10)
            - class_error, cardinality_error (logging only)
        """

        # Separate auxiliary outputs from main output
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # 1. Compute losses for main output (Layer 12)
        indices = self.matcher(outputs_without_aux, targets)

        # Count total boxes for normalization
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes],
            dtype=torch.float,
            device=next(iter(outputs.values())).device
        )
        # Clamp to avoid division by zero
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all requested losses for main output
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs_without_aux, targets, indices, num_boxes))

        # 2. Compute auxiliary losses (Layers 8, 10)
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # Match for this auxiliary output
                indices = self.matcher(aux_outputs, targets)

                # Compute same losses
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Disable logging for auxiliary outputs
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)

                    # Add suffix to distinguish auxiliary losses
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def build_criterion(config: dict) -> SetCriterion:
    """
    Build SetCriterion from configuration

    Args:
        config: Configuration dict

    Returns:
        SetCriterion with Hungarian matcher
    """
    # Build matcher
    matcher = HungarianMatcher(
        cost_class=config['loss']['matcher']['cost_class'],
        cost_bbox=config['loss']['matcher']['cost_bbox'],
        cost_giou=config['loss']['matcher']['cost_giou']
    )

    # Loss weights
    weight_dict = config['loss']['weight_dict'].copy()

    # Add auxiliary loss weights (same as main loss)
    if config['loss']['aux_loss']:
        aux_weight_dict = {}
        # For each auxiliary output (layers 8, 10)
        num_aux_outputs = len(config['model']['exit_layers']) - 1  # Exclude layer 12
        for i in range(num_aux_outputs):
            aux_weight_dict.update({
                k + f'_{i}': v for k, v in weight_dict.items()
            })
        weight_dict.update(aux_weight_dict)

    # Build criterion
    criterion = SetCriterion(
        num_classes=config['model']['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=config['loss']['eos_coef'],
        losses=config['loss']['losses']
    )

    return criterion


if __name__ == "__main__":
    # Test the criterion
    print("Testing SetCriterion...")

    # Dummy config
    config = {
        'model': {
            'num_classes': 1,
            'exit_layers': [8, 10, 12]
        },
        'loss': {
            'losses': ['labels', 'boxes', 'cardinality'],
            'weight_dict': {
                'loss_ce': 1.0,
                'loss_bbox': 5.0,
                'loss_giou': 2.0
            },
            'aux_loss': True,
            'matcher': {
                'cost_class': 1.0,
                'cost_bbox': 5.0,
                'cost_giou': 2.0
            },
            'eos_coef': 0.1
        }
    }

    criterion = build_criterion(config)

    # Dummy outputs
    batch_size = 2
    num_queries = 100
    num_classes = 1

    outputs = {
        'pred_logits': torch.randn(batch_size, num_queries, num_classes + 1),
        'pred_boxes': torch.rand(batch_size, num_queries, 4),
        'aux_outputs': [
            {
                'pred_logits': torch.randn(batch_size, num_queries, num_classes + 1),
                'pred_boxes': torch.rand(batch_size, num_queries, 4)
            },
            {
                'pred_logits': torch.randn(batch_size, num_queries, num_classes + 1),
                'pred_boxes': torch.rand(batch_size, num_queries, 4)
            }
        ]
    }

    # Dummy targets
    targets = [
        {
            'labels': torch.tensor([0, 0, 0]),  # 3 people
            'boxes': torch.rand(3, 4)
        },
        {
            'labels': torch.tensor([0, 0]),  # 2 people
            'boxes': torch.rand(2, 4)
        }
    ]

    # Compute losses
    losses = criterion(outputs, targets)

    print("\nComputed losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    print("\n✓ Criterion test passed!")
