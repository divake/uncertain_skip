"""
Evaluation Script for Multi-Exit YOLOS

Evaluates all exit layers (8, 10, 12) and compares their performance.
Computes mAP, AP50, AP75, precision, recall, and F1 score per exit.
"""

import os
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import components
from models.multi_exit_yolos import build_multi_exit_yolos
from src.data_loader import MOT17DetectionDataset
from utils.metrics import evaluate_model


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    images = torch.stack(images, dim=0)
    return images, targets


@torch.no_grad()
def evaluate_exit_layer(
    model,
    dataloader: DataLoader,
    exit_layer: int,
    device: torch.device,
    conf_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate a single exit layer

    Args:
        model: Multi-exit YOLOS model
        dataloader: DataLoader for evaluation
        exit_layer: Exit layer to evaluate (8, 10, or 12)
        device: Device to use
        conf_threshold: Confidence threshold for predictions

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    print(f"\nEvaluating Layer {exit_layer}...")

    # Collect predictions and targets
    all_predictions = []
    all_targets = []

    for images, targets in tqdm(dataloader, desc=f"Layer {exit_layer}"):
        # Move to device
        images = images.to(device)

        # Forward pass (single exit)
        outputs = model(images, exit_layer=exit_layer)

        # Extract predictions
        pred_logits = outputs['pred_logits']  # [batch, num_queries, num_classes+1]
        pred_boxes = outputs['pred_boxes']    # [batch, num_queries, 4]

        # Convert to list of dicts
        batch_size = pred_logits.shape[0]
        for i in range(batch_size):
            # Get scores and labels
            pred_scores = pred_logits[i].softmax(-1)
            pred_labels = pred_scores.argmax(-1)

            # Filter out no-object class (last class)
            no_object_class = pred_scores.shape[-1] - 1
            is_object = pred_labels != no_object_class

            # Get confidence scores for predicted class
            confidence = pred_scores[torch.arange(len(pred_labels)), pred_labels]

            # Filter by object and confidence
            keep = is_object & (confidence >= conf_threshold)

            all_predictions.append({
                'boxes': pred_boxes[i][keep].cpu(),
                'scores': confidence[keep].cpu(),
                'labels': pred_labels[keep].cpu()
            })

            all_targets.append({
                'boxes': targets[i]['boxes'].cpu(),
                'labels': targets[i]['labels'].cpu()
            })

    # Compute metrics
    metrics = evaluate_model(
        all_predictions,
        all_targets,
        iou_threshold=0.5,
        conf_threshold=conf_threshold
    )

    return metrics


def main():
    """Main evaluation function"""

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Multi-Exit YOLOS')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (default: use config from checkpoint)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                       help='Confidence threshold for predictions')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda, set GPU with CUDA_VISIBLE_DEVICES)')

    args = parser.parse_args()

    # Load checkpoint
    print("=" * 80)
    print("Loading checkpoint...")
    print("=" * 80)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Load config
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = checkpoint['config']

    print(f"✓ Checkpoint loaded: {args.checkpoint}")
    print(f"  - Epoch: {checkpoint['epoch'] + 1}")
    print(f"  - Global step: {checkpoint['global_step']}")

    # Set device
    device = torch.device(args.device)

    # Build model
    print("\n" + "=" * 80)
    print("Building model...")
    print("=" * 80)
    model = build_multi_exit_yolos(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print("✓ Model loaded")

    # Build dataset
    print("\n" + "=" * 80)
    print("Loading validation dataset...")
    print("=" * 80)

    val_dataset = MOT17DetectionDataset(
        root_path=config['dataset']['root_path'],
        sequences=config['dataset']['val_sequences'],
        image_size=config['dataset']['image_size']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    print(f"✓ Validation samples: {len(val_dataset)}")
    print(f"✓ Batch size: {args.batch_size}")

    # Evaluate all exit layers
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    results = {}

    for exit_layer in config['model']['exit_layers']:
        metrics = evaluate_exit_layer(
            model,
            val_loader,
            exit_layer,
            device,
            conf_threshold=args.conf_threshold
        )

        results[f'layer_{exit_layer}'] = metrics

        print(f"\nLayer {exit_layer} Results:")
        print(f"  mAP:       {metrics['mAP']:.4f}")
        print(f"  AP50:      {metrics['AP50']:.4f}")
        print(f"  AP75:      {metrics['AP75']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")

    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"{'Layer':<10} {'mAP':<10} {'AP50':<10} {'AP75':<10} {'Precision':<12} {'Recall':<10} {'F1':<10}")
    print("-" * 80)

    for exit_layer in config['model']['exit_layers']:
        metrics = results[f'layer_{exit_layer}']
        print(f"{exit_layer:<10} "
              f"{metrics['mAP']:<10.4f} "
              f"{metrics['AP50']:<10.4f} "
              f"{metrics['AP75']:<10.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<10.4f} "
              f"{metrics['f1_score']:<10.4f}")

    # Save results
    checkpoint_dir = Path(args.checkpoint).parent
    results_path = checkpoint_dir / "evaluation_results.json"

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved: {results_path}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
