"""
Training Script for Multi-Exit YOLOS

Complete training pipeline with extensive logging for multi-exit YOLOS fine-tuning.
Supports two-phase training strategy:
- Phase 1: Train only new detection heads (layers 8, 10) with frozen backbone
- Phase 2: Optional fine-tuning of entire model with small learning rate

Features:
- Extensive logging every 50 steps
- TensorBoard integration
- Checkpoint saving (every 5 epochs + best model)
- Mixed precision training (FP16)
- Gradient clipping
- Per-exit loss tracking
- Validation after each epoch
"""

import os
import sys
import yaml
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import multi-exit YOLOS components
from models.multi_exit_yolos import build_multi_exit_yolos
from training.criterion import build_criterion

# Import MOT17 data loader
from src.data_loader import MOT17DetectionDataset


class Trainer:
    """
    Trainer for Multi-Exit YOLOS

    Handles:
    - Training loop with extensive logging
    - Validation
    - Checkpointing
    - TensorBoard logging
    - Loss tracking per exit
    """

    def __init__(self, config: dict, output_dir: str):
        """
        Initialize trainer

        Args:
            config: Configuration dictionary from YAML
            output_dir: Directory for saving outputs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set device
        self.device = torch.device(config['hardware']['device'])

        # Set random seed for reproducibility
        self._set_seed(config['seed'])

        # Build model
        print("=" * 80)
        print("Building Multi-Exit YOLOS model...")
        print("=" * 80)
        self.model = build_multi_exit_yolos(config)
        self.model.to(self.device)

        # Build loss function
        print("\n" + "=" * 80)
        print("Building loss function (SetCriterion)...")
        print("=" * 80)
        self.criterion = build_criterion(config)
        self.criterion.to(self.device)

        # Build optimizer (only trainable parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config['training']['phase1']['learning_rate'],
            weight_decay=config['training']['phase1']['weight_decay']
        )

        # Build learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['training']['lr_drop'],
            gamma=config['training']['lr_drop_rate']
        )

        # Mixed precision training
        self.use_amp = config['training']['use_amp']
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient clipping
        self.clip_max_norm = config['training']['clip_max_norm']

        # Logging setup
        self.log_every = config['logging']['log_every_n_steps']
        self.save_every = config['training']['save_every']

        # TensorBoard
        if config['logging']['use_tensorboard']:
            tensorboard_dir = self.output_dir / "tensorboard"
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
        else:
            self.writer = None

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_map = 0.0

        # Loss tracking
        self.train_losses = []
        self.val_metrics = []

        print(f"\n✓ Trainer initialized")
        print(f"  - Device: {self.device}")
        print(f"  - Output directory: {self.output_dir}")
        print(f"  - Mixed precision: {self.use_amp}")
        print(f"  - Gradient clipping: {self.clip_max_norm}")

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        if self.config['deterministic']:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch

        Args:
            dataloader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary with average losses
        """
        self.model.train()
        self.criterion.train()

        # Loss accumulation
        epoch_losses = {
            'loss_total': 0.0,
            'loss_ce': 0.0,
            'loss_bbox': 0.0,
            'loss_giou': 0.0,
            'loss_ce_0': 0.0,    # Layer 8
            'loss_bbox_0': 0.0,
            'loss_giou_0': 0.0,
            'loss_ce_1': 0.0,    # Layer 10
            'loss_bbox_1': 0.0,
            'loss_giou_1': 0.0
        }

        num_batches = len(dataloader)

        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                       for k, v in t.items()} for t in targets]

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                # Model forward (returns all exits in training mode)
                outputs = self.model(images)

                # Compute losses
                loss_dict = self.criterion(outputs, targets)

                # Weighted sum of all losses
                losses = sum(loss_dict[k] * self.criterion.weight_dict[k]
                           for k in loss_dict.keys() if k in self.criterion.weight_dict)

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(losses).backward()

                # Gradient clipping
                if self.clip_max_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_max_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses.backward()

                # Gradient clipping
                if self.clip_max_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_max_norm
                    )

                self.optimizer.step()

            # Accumulate losses
            epoch_losses['loss_total'] += losses.item()
            for k in loss_dict.keys():
                if k in epoch_losses:
                    epoch_losses[k] += loss_dict[k].item()

            # Logging
            if (batch_idx + 1) % self.log_every == 0:
                current_lr = self.optimizer.param_groups[0]['lr']

                # Log to console
                log_str = (
                    f"[Epoch {epoch + 1}/{self.config['training']['phase1']['epochs']}] "
                    f"[Batch {batch_idx + 1}/{num_batches}] "
                    f"Loss: {losses.item():.4f} | "
                    f"CE: {loss_dict.get('loss_ce', 0):.4f} | "
                    f"BBox: {loss_dict.get('loss_bbox', 0):.4f} | "
                    f"GIoU: {loss_dict.get('loss_giou', 0):.4f} | "
                    f"LR: {current_lr:.6f}"
                )

                if self.clip_max_norm > 0:
                    log_str += f" | Grad: {grad_norm:.4f}"

                pbar.set_description(log_str)

                # Log to TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar('train/loss_total', losses.item(), self.global_step)
                    self.writer.add_scalar('train/learning_rate', current_lr, self.global_step)

                    for k, v in loss_dict.items():
                        self.writer.add_scalar(f'train/{k}', v.item(), self.global_step)

                    if self.clip_max_norm > 0:
                        self.writer.add_scalar('train/grad_norm', grad_norm, self.global_step)

            self.global_step += 1

        # Average losses over epoch
        for k in epoch_losses.keys():
            epoch_losses[k] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate on validation set

        Args:
            dataloader: Validation data loader
            epoch: Current epoch number

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        print("\n" + "=" * 80)
        print(f"Running validation (Epoch {epoch + 1})...")
        print("=" * 80)

        # Loss accumulation
        val_losses = {
            'loss_total': 0.0,
            'loss_ce': 0.0,
            'loss_bbox': 0.0,
            'loss_giou': 0.0,
            'loss_ce_0': 0.0,
            'loss_bbox_0': 0.0,
            'loss_giou_0': 0.0,
            'loss_ce_1': 0.0,
            'loss_bbox_1': 0.0,
            'loss_giou_1': 0.0
        }

        num_batches = len(dataloader)

        for images, targets in tqdm(dataloader, desc="Validation"):
            # Move to device
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                       for k, v in t.items()} for t in targets]

            # Forward pass - explicitly request all exits for validation
            outputs = self.model(images, output_all_exits=True)

            # Compute losses
            loss_dict = self.criterion(outputs, targets)
            losses = sum(loss_dict[k] * self.criterion.weight_dict[k]
                        for k in loss_dict.keys() if k in self.criterion.weight_dict)

            # Accumulate
            val_losses['loss_total'] += losses.item()
            for k in loss_dict.keys():
                if k in val_losses:
                    val_losses[k] += loss_dict[k].item()

        # Average losses
        for k in val_losses.keys():
            val_losses[k] /= num_batches

        # Print validation summary
        print(f"\nValidation Summary (Epoch {epoch + 1}):")
        print(f"  Total Loss: {val_losses['loss_total']:.4f}")
        print(f"  Layer 12 - CE: {val_losses['loss_ce']:.4f}, "
              f"BBox: {val_losses['loss_bbox']:.4f}, "
              f"GIoU: {val_losses['loss_giou']:.4f}")
        print(f"  Layer 8  - CE: {val_losses['loss_ce_0']:.4f}, "
              f"BBox: {val_losses['loss_bbox_0']:.4f}, "
              f"GIoU: {val_losses['loss_giou_0']:.4f}")
        print(f"  Layer 10 - CE: {val_losses['loss_ce_1']:.4f}, "
              f"BBox: {val_losses['loss_bbox_1']:.4f}, "
              f"GIoU: {val_losses['loss_giou_1']:.4f}")

        # Log to TensorBoard
        if self.writer is not None:
            for k, v in val_losses.items():
                self.writer.add_scalar(f'val/{k}', v, epoch)

        return val_losses

    def save_checkpoint(self, epoch: int, val_metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint

        Args:
            epoch: Current epoch
            val_metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'val_metrics': val_metrics,
            'best_val_map': self.best_val_map,
            'config': self.config
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved: {best_path}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        """
        Full training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
        """
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        print(f"Number of epochs: {num_epochs}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Batch size: {self.config['training']['phase1']['batch_size']}")
        print(f"Learning rate: {self.config['training']['phase1']['learning_rate']}")
        print("=" * 80)

        start_time = time.time()

        for epoch in range(num_epochs):
            self.epoch = epoch

            print(f"\n{'=' * 80}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'=' * 80}")

            # Train for one epoch
            train_losses = self.train_epoch(train_loader, epoch)

            # Log epoch summary
            print(f"\nEpoch {epoch + 1} Training Summary:")
            print(f"  Total Loss: {train_losses['loss_total']:.4f}")
            print(f"  Layer 12 - CE: {train_losses['loss_ce']:.4f}, "
                  f"BBox: {train_losses['loss_bbox']:.4f}, "
                  f"GIoU: {train_losses['loss_giou']:.4f}")
            print(f"  Layer 8  - CE: {train_losses['loss_ce_0']:.4f}, "
                  f"BBox: {train_losses['loss_bbox_0']:.4f}, "
                  f"GIoU: {train_losses['loss_giou_0']:.4f}")
            print(f"  Layer 10 - CE: {train_losses['loss_ce_1']:.4f}, "
                  f"BBox: {train_losses['loss_bbox_1']:.4f}, "
                  f"GIoU: {train_losses['loss_giou_1']:.4f}")

            # Validation
            val_losses = self.validate(val_loader, epoch)

            # Update learning rate
            self.lr_scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(epoch, val_losses)

            # Check if best model (based on total validation loss)
            current_val_loss = val_losses['loss_total']
            if epoch == 0 or current_val_loss < self.best_val_map:
                self.best_val_map = current_val_loss
                self.save_checkpoint(epoch, val_losses, is_best=True)

            # Save training history
            self.train_losses.append(train_losses)
            self.val_metrics.append(val_losses)

        # Training complete
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Total training time: {hours}h {minutes}m")
        print(f"Best validation loss: {self.best_val_map:.4f}")
        print(f"Checkpoints saved to: {self.output_dir}")
        print("=" * 80)

        # Save final training log
        self._save_training_log()

        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()

    def _save_training_log(self):
        """Save detailed training log to JSON"""
        log_data = {
            'config': self.config,
            'training_summary': {
                'total_epochs': self.epoch + 1,
                'total_steps': self.global_step,
                'best_val_loss': self.best_val_map
            },
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics
        }

        log_path = self.output_dir / "training_log.json"
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"✓ Training log saved: {log_path}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def collate_fn(batch):
    """
    Custom collate function for DataLoader

    Handles variable number of boxes per image
    """
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    # Stack images
    images = torch.stack(images, dim=0)

    return images, targets


def main():
    """Main training function"""

    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "multi_exit_config.yaml"
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config['logging']['output_dir']) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output directory
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)

    print(f"\nOutput directory: {output_dir}")

    # Build datasets
    print("\n" + "=" * 80)
    print("Loading datasets...")
    print("=" * 80)

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

    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")

    # Build data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['phase1']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        collate_fn=collate_fn,
        drop_last=True  # Drop last incomplete batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['phase1']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        collate_fn=collate_fn
    )

    # Build trainer
    trainer = Trainer(config, output_dir)

    # Start training
    num_epochs = config['training']['phase1']['epochs']
    trainer.train(train_loader, val_loader, num_epochs)

    print("\n✓ Training script completed successfully!")


if __name__ == "__main__":
    main()
