"""
Phase 2 Training Script for Multi-Exit YOLOS

Fine-tunes the entire model with layer-wise learning rates.
Includes per-epoch evaluation and smart checkpoint saving.
"""

import os
import sys
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models.multi_exit_yolos import build_multi_exit_yolos
from src.data_loader import MOT17DetectionDataset
from training.criterion import SetCriterion
from utils.matcher import HungarianMatcher
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


def build_optimizer_with_layer_wise_lr(model, config):
    """
    Build optimizer with layer-wise learning rates

    Different learning rates for different parts:
    - Early backbone layers (1-7): Very small LR
    - Mid backbone layers (8-9): Small LR
    - Late backbone layers (10-11): Medium LR
    - Layer 12: Medium LR
    - Detection heads: Large LR
    """
    lr_config = config['training']['layer_wise_lr']

    if not lr_config['enabled']:
        # Fallback to single LR if layer-wise disabled
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr_config['detection_heads'],
            weight_decay=config['training']['weight_decay']
        )

    # Group parameters by layer depth
    param_groups = []

    # Backbone layers 0-6 (indices 0-6, 7 layers total)
    if hasattr(model.backbone.encoder, 'layer'):
        early_backbone_params = []
        for i in range(7):  # Layers 0-6
            early_backbone_params.extend(list(model.backbone.encoder.layer[i].parameters()))

        if early_backbone_params:
            param_groups.append({
                'params': early_backbone_params,
                'lr': lr_config['backbone_early'],
                'name': 'backbone_early_layers_0-6'
            })

        # Backbone layers 7-8 (indices 7-8, 2 layers)
        mid_backbone_params = []
        for i in range(7, 9):  # Layers 7-8
            mid_backbone_params.extend(list(model.backbone.encoder.layer[i].parameters()))

        if mid_backbone_params:
            param_groups.append({
                'params': mid_backbone_params,
                'lr': lr_config['backbone_mid'],
                'name': 'backbone_mid_layers_7-8'
            })

        # Backbone layers 9-10 (indices 9-10, 2 layers)
        late_backbone_params = []
        for i in range(9, 11):  # Layers 9-10
            late_backbone_params.extend(list(model.backbone.encoder.layer[i].parameters()))

        if late_backbone_params:
            param_groups.append({
                'params': late_backbone_params,
                'lr': lr_config['backbone_late'],
                'name': 'backbone_late_layers_9-10'
            })

        # Layer 11 (index 11, layer 12 in 1-indexed)
        layer_12_params = list(model.backbone.encoder.layer[11].parameters())
        if layer_12_params:
            param_groups.append({
                'params': layer_12_params,
                'lr': lr_config['layer_12'],
                'name': 'backbone_layer_11'
            })

    # Detection heads (all three: layer 8, 10, 12)
    detection_head_params = []
    for name, param in model.named_parameters():
        if 'detection_heads' in name:
            detection_head_params.append(param)

    if detection_head_params:
        param_groups.append({
            'params': detection_head_params,
            'lr': lr_config['detection_heads'],
            'name': 'detection_heads'
        })

    print(f"\n{'='*80}")
    print("Optimizer Parameter Groups:")
    print(f"{'='*80}")
    for group in param_groups:
        num_params = sum(p.numel() for p in group['params'])
        print(f"  {group['name']:<30} LR: {group['lr']:.2e}  Params: {num_params:,}")
    print(f"{'='*80}\n")

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config['training']['weight_decay']
    )

    return optimizer


@torch.no_grad()
def evaluate_all_exits(model, dataloader, device, config):
    """
    Evaluate all exit layers and return metrics

    Returns:
        dict: Metrics for each exit layer
    """
    model.eval()

    conf_threshold = config['evaluation']['conf_threshold']
    iou_threshold = config['evaluation']['iou_threshold']

    results = {}

    for exit_layer in config['model']['exit_layers']:
        # Collect predictions and targets
        all_predictions = []
        all_targets = []

        for images, targets in tqdm(dataloader, desc=f"Eval Layer {exit_layer}", leave=False):
            images = images.to(device)

            # Forward pass (single exit)
            outputs = model(images, exit_layer=exit_layer)

            pred_logits = outputs['pred_logits']
            pred_boxes = outputs['pred_boxes']

            batch_size = pred_logits.shape[0]
            for i in range(batch_size):
                # Get scores and labels
                pred_scores = pred_logits[i].softmax(-1)
                pred_labels = pred_scores.argmax(-1)

                # Filter out no-object class
                no_object_class = pred_scores.shape[-1] - 1
                is_object = pred_labels != no_object_class

                # Get confidence scores
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
            iou_threshold=iou_threshold,
            conf_threshold=conf_threshold
        )

        results[f'layer_{exit_layer}'] = metrics

    return results


def train_one_epoch(model, criterion, dataloader, optimizer, scaler, device, epoch, config):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    num_batches = len(dataloader)

    log_every = config['logging']['log_every_n_steps']

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

    for batch_idx, (images, targets) in enumerate(pbar):
        # Move to device
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        # Forward pass with mixed precision
        with autocast(enabled=config['training']['use_amp']):
            # Get outputs from all exits
            outputs = model(images, output_all_exits=True)

            # Compute loss
            loss_dict = criterion(outputs, targets)

            # Weight losses by exit importance
            exit_weights = config['loss']['exit_weights']
            weight_dict = criterion.weight_dict

            # Compute total loss
            loss = 0
            for k in loss_dict.keys():
                if k in weight_dict:
                    # Main losses (layer 12)
                    loss += loss_dict[k] * weight_dict[k] * exit_weights['layer_12']
                elif 'aux' in k:
                    # Auxiliary losses (layers 8 and 10)
                    if '_0' in k:  # Layer 8
                        base_key = k.replace('_0', '')
                        loss += loss_dict[k] * weight_dict.get(base_key, 1.0) * exit_weights['layer_8']
                    elif '_1' in k:  # Layer 10
                        base_key = k.replace('_1', '')
                        loss += loss_dict[k] * weight_dict.get(base_key, 1.0) * exit_weights['layer_10']

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping
        if config['training']['clip_max_norm'] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['clip_max_norm']
            )

        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        total_loss += loss.item()

        # Update progress bar
        if (batch_idx + 1) % log_every == 0:
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """Main Phase 2 training function"""

    # Load config
    config_path = 'configs/phase2_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"{'='*80}")
    print("PHASE 2 TRAINING: Multi-Exit YOLOS Fine-Tuning")
    print(f"{'='*80}")
    print(f"Config: {config_path}")
    print(f"Device: {config['hardware']['device']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"{'='*80}\n")

    # Set device
    device = torch.device(config['hardware']['device'])

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config['logging']['output_dir']) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    # Build model
    print(f"{'='*80}")
    print("Building model...")
    print(f"{'='*80}")
    model = build_multi_exit_yolos(config)

    # Load Phase 1 checkpoint
    pretrained_path = config['training']['pretrained_checkpoint']
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"\nLoading Phase 1 checkpoint: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Phase 1 weights loaded successfully")
    else:
        print("\n⚠ Warning: No Phase 1 checkpoint found, starting from pretrained YOLOS")

    model.to(device)

    # Build criterion
    matcher = HungarianMatcher(
        cost_class=config['loss']['matcher']['cost_class'],
        cost_bbox=config['loss']['matcher']['cost_bbox'],
        cost_giou=config['loss']['matcher']['cost_giou']
    )

    criterion = SetCriterion(
        num_classes=config['model']['num_classes'],
        matcher=matcher,
        weight_dict=config['loss']['weight_dict'],
        eos_coef=config['loss']['eos_coef'],
        losses=config['loss']['losses']
    )
    criterion.to(device)

    # Build optimizer with layer-wise learning rates
    print(f"\n{'='*80}")
    print("Building optimizer...")
    print(f"{'='*80}")
    optimizer = build_optimizer_with_layer_wise_lr(model, config)

    # Build datasets
    print(f"\n{'='*80}")
    print("Loading datasets...")
    print(f"{'='*80}")

    train_dataset = MOT17DetectionDataset(
        root_path=config['dataset']['root_path'],
        sequences=config['dataset']['train_sequences'],
        image_size=config['dataset']['image_size']
    )

    val_dataset = MOT17DetectionDataset(
        root_path=config['dataset']['root_path'],
        sequences=config['dataset']['val_sequences'],
        image_size=config['dataset']['image_size']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        collate_fn=collate_fn
    )

    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}\n")

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['lr_drop'],
        gamma=config['training']['lr_drop_rate']
    )

    # Mixed precision scaler
    scaler = GradScaler(enabled=config['training']['use_amp'])

    # Training loop
    print(f"{'='*80}")
    print("TRAINING START")
    print(f"{'='*80}\n")

    best_map = 0.0
    training_log = []

    start_time = datetime.now()

    for epoch in range(config['training']['epochs']):
        epoch_start = datetime.now()

        # Train one epoch
        train_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, scaler, device, epoch, config
        )

        # Evaluate
        print(f"\nEvaluating epoch {epoch+1}...")
        eval_results = evaluate_all_exits(model, val_loader, device, config)

        # Compute average mAP across all exits
        avg_map = np.mean([
            eval_results[f'layer_{layer}']['mAP']
            for layer in config['model']['exit_layers']
        ])

        # Print results
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{config['training']['epochs']} Results")
        print(f"{'='*80}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Average mAP: {avg_map:.4f}")
        print()
        print(f"{'Layer':<10} {'mAP':<10} {'AP50':<10} {'AP75':<10} {'F1':<10}")
        print(f"{'-'*50}")

        for layer in config['model']['exit_layers']:
            metrics = eval_results[f'layer_{layer}']
            print(f"{layer:<10} "
                  f"{metrics['mAP']:<10.4f} "
                  f"{metrics['AP50']:<10.4f} "
                  f"{metrics['AP75']:<10.4f} "
                  f"{metrics['f1_score']:<10.4f}")

        print(f"{'='*80}\n")

        # Update learning rate
        scheduler.step()

        # Log epoch results
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'avg_map': avg_map,
            'eval_results': eval_results,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time': (datetime.now() - epoch_start).total_seconds()
        }
        training_log.append(epoch_log)

        # Save checkpoint if improved
        if avg_map > best_map:
            best_map = avg_map

            checkpoint_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'avg_map': avg_map,
                'eval_results': eval_results,
                'config': config
            }, checkpoint_path)

            print(f"✓ Best model saved (mAP: {avg_map:.4f} > {best_map:.4f})")
            print(f"  Checkpoint: {checkpoint_path}\n")

        # Save periodic checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'avg_map': avg_map,
                'eval_results': eval_results,
                'config': config
            }, checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}\n")

    # Training complete
    total_time = datetime.now() - start_time

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time}")
    print(f"Best average mAP: {best_map:.4f}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    # Save training log
    log_path = output_dir / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"✓ Training log saved: {log_path}")

    # Save config
    config_save_path = output_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✓ Config saved: {config_save_path}")


if __name__ == "__main__":
    main()
