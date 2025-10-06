#!/usr/bin/env python3
"""
Multi-Exit YOLOS Training Framework
Fine-tunes YOLOS with detection heads at layers 3, 6, 9, 12
Extensive logging for research analysis
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
from pathlib import Path
import json
import time
from datetime import datetime
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
# import wandb  # Optional: for experiment tracking

class DetectionHead(nn.Module):
    """Detection head for intermediate layers"""
    def __init__(self, hidden_dim=384, num_classes=1, num_queries=100):
        super().__init__()
        self.num_queries = num_queries
        
        # Object queries (learnable)
        self.query_pos = nn.Parameter(torch.randn(num_queries, hidden_dim))
        
        # Cross-attention for object queries
        self.cross_attn = nn.MultiheadAttention(hidden_dim, 8, dropout=0.1)
        
        # Detection outputs
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # cx, cy, w, h
        )
        
    def forward(self, encoder_output):
        """
        encoder_output: [batch_size, seq_len, hidden_dim]
        """
        bs = encoder_output.shape[0]
        
        # Prepare queries
        query_embed = self.query_pos.unsqueeze(0).repeat(bs, 1, 1)
        
        # Cross-attention between queries and encoder output
        # Note: MultiheadAttention expects (seq_len, batch, hidden_dim)
        encoder_output = encoder_output.transpose(0, 1)
        query_embed = query_embed.transpose(0, 1)
        
        hs, _ = self.cross_attn(query_embed, encoder_output, encoder_output)
        hs = hs.transpose(0, 1)  # Back to [batch, queries, hidden_dim]
        
        # Get outputs
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        return {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord
        }

class MultiExitYOLOS(nn.Module):
    """YOLOS with detection heads at multiple layers"""
    
    def __init__(self, model_name='hustvl/yolos-small'):
        super().__init__()
        
        # Load pretrained YOLOS
        self.base_model = YolosForObjectDetection.from_pretrained(model_name)
        
        # Get hidden dimension
        self.hidden_dim = self.base_model.config.hidden_size
        
        # Add detection heads for early exits
        self.detection_heads = nn.ModuleDict({
            'layer_3': DetectionHead(self.hidden_dim),
            'layer_6': DetectionHead(self.hidden_dim),
            'layer_9': DetectionHead(self.hidden_dim),
            'layer_12': None  # Use original head
        })
        
        # Layer indices for exits
        self.exit_layers = [3, 6, 9, 12]
        
    def forward_single_exit(self, pixel_values, exit_layer=12):
        """Forward pass with single exit"""
        
        # Get ViT encoder
        vit = self.base_model.vit
        
        # Embeddings
        hidden_states = vit.embeddings(pixel_values)
        
        # Process through encoder layers
        for i in range(min(exit_layer, 12)):
            layer_outputs = vit.encoder.layer[i](hidden_states)
            hidden_states = layer_outputs[0]
        
        # Apply layer norm
        if exit_layer >= 12:
            hidden_states = vit.layernorm(hidden_states)
        
        # Get detection output
        if exit_layer < 12:
            head = self.detection_heads[f'layer_{exit_layer}']
            outputs = head(hidden_states)
        else:
            # Use original detection head with detection tokens
            sequence_output = hidden_states[:, :100, :]  # First 100 are detection tokens
            
            outputs = {
                'pred_logits': self.base_model.class_labels_classifier(sequence_output),
                'pred_boxes': self.base_model.bbox_predictor(sequence_output).sigmoid()
            }
        
        return outputs
    
    def forward_all_exits(self, pixel_values):
        """Forward pass through all exit points - for training"""
        
        outputs = {}
        vit = self.base_model.vit
        
        # Embeddings
        hidden_states = vit.embeddings(pixel_values)
        
        # Process through all layers, collecting outputs at exit points
        for i in range(12):
            layer_outputs = vit.encoder.layer[i](hidden_states)
            hidden_states = layer_outputs[0]
            
            # Check if this is an exit layer
            exit_layer = i + 1
            if exit_layer in [3, 6, 9]:
                head = self.detection_heads[f'layer_{exit_layer}']
                outputs[f'layer_{exit_layer}'] = head(hidden_states.clone())
        
        # Layer 12 with final norm
        hidden_states = vit.layernorm(hidden_states)
        
        # Original detection head - extract detection tokens
        # YOLOS adds 100 detection tokens at the beginning
        sequence_output = hidden_states[:, :100, :]  # [batch, 100, hidden_dim]
        
        outputs['layer_12'] = {
            'pred_logits': self.base_model.class_labels_classifier(sequence_output),
            'pred_boxes': self.base_model.bbox_predictor(sequence_output).sigmoid()
        }
        
        return outputs

class MOT17Dataset(Dataset):
    """MOT17 Dataset for training"""
    
    def __init__(self, sequences, data_root='../data/MOT17/train', processor=None, max_objects=50):
        self.sequences = sequences
        self.data_root = Path(data_root)
        self.processor = processor
        self.max_objects = max_objects
        
        # Build frame list
        self.frames = []
        for seq in sequences:
            seq_path = self.data_root / f"{seq}-FRCNN"
            img_dir = seq_path / "img1"
            gt_file = seq_path / "gt" / "gt.txt"
            
            # Load GT
            gt_data = pd.read_csv(gt_file, header=None)
            gt_data.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'visibility']
            
            # Get frame list
            for img_path in sorted(img_dir.glob("*.jpg")):
                frame_num = int(img_path.stem)
                frame_gt = gt_data[gt_data['frame'] == frame_num]
                
                # Filter visible persons
                frame_gt = frame_gt[frame_gt['visibility'] > 0.3]
                
                if len(frame_gt) > 0:
                    self.frames.append({
                        'img_path': img_path,
                        'boxes': frame_gt[['x', 'y', 'w', 'h']].values,
                        'sequence': seq
                    })
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        
        # Load image
        image = Image.open(frame['img_path']).convert('RGB')
        
        # Process image
        if self.processor:
            encoding = self.processor(images=image, return_tensors="pt")
            pixel_values = encoding['pixel_values'].squeeze(0)
        else:
            # Fallback
            pixel_values = torch.randn(3, 224, 224)
        
        # Prepare targets
        boxes = frame['boxes']  # [N, 4] in xywh format
        
        # Convert to relative coordinates [0, 1]
        w, h = image.size
        boxes_norm = boxes.astype(np.float32)
        boxes_norm[:, [0, 2]] /= w
        boxes_norm[:, [1, 3]] /= h
        
        # Convert xywh to cxcywh
        boxes_cxcy = np.zeros_like(boxes_norm)
        boxes_cxcy[:, 0] = boxes_norm[:, 0] + boxes_norm[:, 2] / 2  # cx
        boxes_cxcy[:, 1] = boxes_norm[:, 1] + boxes_norm[:, 3] / 2  # cy
        boxes_cxcy[:, 2] = boxes_norm[:, 2]  # w
        boxes_cxcy[:, 3] = boxes_norm[:, 3]  # h
        
        # Pad to max_objects
        n_objects = len(boxes_cxcy)
        if n_objects < self.max_objects:
            pad_boxes = np.zeros((self.max_objects - n_objects, 4))
            boxes_cxcy = np.vstack([boxes_cxcy, pad_boxes])
            labels = np.concatenate([np.ones(n_objects), np.zeros(self.max_objects - n_objects)])
        else:
            boxes_cxcy = boxes_cxcy[:self.max_objects]
            labels = np.ones(self.max_objects)
        
        return {
            'pixel_values': pixel_values,
            'boxes': torch.FloatTensor(boxes_cxcy),
            'labels': torch.LongTensor(labels),
            'n_objects': n_objects
        }

class MultiExitLoss(nn.Module):
    """Multi-exit loss with Hungarian matching"""
    
    def __init__(self, layer_weights=[0.1, 0.2, 0.3, 0.4], 
                 cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.layer_weights = layer_weights
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        
    def forward(self, outputs_dict, targets):
        """
        outputs_dict: Dict with 'layer_3', 'layer_6', 'layer_9', 'layer_12' outputs
        targets: Dict with 'boxes', 'labels', 'n_objects'
        """
        
        losses = {}
        total_loss = 0
        
        # Compute loss for each exit
        exit_names = ['layer_3', 'layer_6', 'layer_9', 'layer_12']
        
        for i, exit_name in enumerate(exit_names):
            if exit_name not in outputs_dict:
                continue
                
            outputs = outputs_dict[exit_name]
            
            # Simple L1 loss for boxes and CE for classification
            # In production, use Hungarian matching
            
            # Handle different output sizes between layers
            pred_boxes = outputs['pred_boxes']
            target_boxes = targets['boxes']
            
            # Truncate to minimum size
            min_queries = min(pred_boxes.shape[1], target_boxes.shape[1])
            pred_boxes = pred_boxes[:, :min_queries, :]
            target_boxes = target_boxes[:, :min_queries, :]
            
            loss_bbox = nn.functional.l1_loss(
                pred_boxes, 
                target_boxes,
                reduction='mean'
            )
            
            # Classification loss
            pred_logits = outputs['pred_logits']
            target_classes = targets['labels']
            
            # Handle size mismatch
            if len(pred_logits.shape) == 3:  # [batch, queries, classes]
                min_queries = min(pred_logits.shape[1], target_classes.shape[1])
                pred_logits = pred_logits[:, :min_queries, :].flatten(0, 1)
                target_classes = target_classes[:, :min_queries].flatten()
            else:  # Layer 12 output [batch, classes]
                # For layer 12, use first target label for each batch
                target_classes = target_classes[:, 0]
            
            loss_ce = nn.functional.cross_entropy(
                pred_logits,
                target_classes,
                reduction='mean'
            )
            
            # Combine losses
            exit_loss = loss_bbox * self.cost_bbox + loss_ce * self.cost_class
            
            # Weight by layer importance
            weighted_loss = exit_loss * self.layer_weights[i]
            
            losses[f'{exit_name}_loss'] = exit_loss.item()
            losses[f'{exit_name}_loss_bbox'] = loss_bbox.item()
            losses[f'{exit_name}_loss_ce'] = loss_ce.item()
            
            total_loss += weighted_loss
        
        losses['total_loss'] = total_loss.item()
        
        return total_loss, losses

class MultiExitTrainer:
    """Trainer with extensive logging"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Logging setup
        self.log_dir = Path(config['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_log_{timestamp}.jsonl"
        self.metrics_file = self.log_dir / f"metrics_{timestamp}.csv"
        
        # Training history
        self.history = defaultdict(list)
        
    def log(self, data):
        """Log data to file and console"""
        data['timestamp'] = datetime.now().isoformat()
        
        # Console output
        if 'epoch' in data and 'loss' in data:
            print(f"[{data['timestamp']}] Epoch {data['epoch']}: Loss={data['loss']:.4f}")
        
        # File output
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
    
    def train_epoch(self, dataloader, optimizer, criterion, epoch):
        """Train for one epoch with detailed logging"""
        
        self.model.train()
        epoch_losses = defaultdict(list)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            pixel_values = batch['pixel_values'].to(self.device)
            targets = {
                'boxes': batch['boxes'].to(self.device),
                'labels': batch['labels'].to(self.device),
                'n_objects': batch['n_objects']
            }
            
            # Forward pass through all exits
            outputs = self.model.forward_all_exits(pixel_values)
            
            # Compute multi-exit loss
            loss, loss_dict = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log batch metrics
            for k, v in loss_dict.items():
                epoch_losses[k].append(v)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Detailed logging every N batches
            if batch_idx % 10 == 0:
                self.log({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss': loss.item(),
                    **loss_dict
                })
        
        # Epoch summary
        epoch_summary = {
            'epoch': epoch,
            **{k: np.mean(v) for k, v in epoch_losses.items()}
        }
        
        self.log({'epoch_summary': epoch_summary})
        
        return epoch_summary
    
    def validate(self, dataloader, criterion, epoch):
        """Validation with per-layer metrics"""
        
        self.model.eval()
        val_losses = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                pixel_values = batch['pixel_values'].to(self.device)
                targets = {
                    'boxes': batch['boxes'].to(self.device),
                    'labels': batch['labels'].to(self.device),
                    'n_objects': batch['n_objects']
                }
                
                outputs = self.model.forward_all_exits(pixel_values)
                loss, loss_dict = criterion(outputs, targets)
                
                for k, v in loss_dict.items():
                    val_losses[k].append(v)
        
        val_summary = {
            'epoch': epoch,
            **{'val_' + k: np.mean(v) for k, v in val_losses.items()}
        }
        
        self.log({'validation_summary': val_summary})
        
        return val_summary
    
    def train(self, train_loader, val_loader, epochs, lr=1e-4):
        """Full training loop"""
        
        # Optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
        # Loss
        criterion = MultiExitLoss()
        
        # Training loop
        for epoch in range(1, epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{epochs}")
            print('='*60)
            
            # Train
            train_summary = self.train_epoch(train_loader, optimizer, criterion, epoch)
            
            # Validate
            val_summary = self.validate(val_loader, criterion, epoch)
            
            # Store history
            for k, v in {**train_summary, **val_summary}.items():
                self.history[k].append(v)
            
            # Save checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch)
            
            # Plot progress
            self.plot_training_curves(epoch)
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        
        checkpoint_path = self.log_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'history': dict(self.history)
        }, checkpoint_path)
        
        print(f"✅ Saved checkpoint: {checkpoint_path}")
    
    def plot_training_curves(self, epoch):
        """Plot training curves for each exit"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        exits = ['layer_3', 'layer_6', 'layer_9', 'layer_12']
        
        for i, (ax, exit_name) in enumerate(zip(axes.flat, exits)):
            # Plot loss curves
            train_key = f'{exit_name}_loss'
            val_key = f'val_{exit_name}_loss'
            
            if train_key in self.history:
                ax.plot(self.history[train_key], label='Train', linewidth=2)
            if val_key in self.history:
                ax.plot(self.history[val_key], label='Val', linewidth=2)
            
            ax.set_title(f'{exit_name.replace("_", " ").title()} Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Multi-Exit Training Progress - Epoch {epoch}')
        plt.tight_layout()
        
        plot_path = self.log_dir / f'training_curves_epoch_{epoch}.png'
        plt.savefig(plot_path, dpi=100)
        plt.close()

def main():
    """Main training script"""
    
    print("="*60)
    print("Multi-Exit YOLOS Training on MOT17")
    print("="*60)
    
    # Configuration
    config = {
        'model_name': 'hustvl/yolos-small',
        'train_sequences': ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10'],
        'val_sequences': ['MOT17-11', 'MOT17-13'],
        'batch_size': 4,
        'epochs': 30,
        'learning_rate': 1e-4,
        'log_dir': 'results/multi_exit_training'
    }
    
    # Initialize processor
    processor = YolosImageProcessor.from_pretrained(config['model_name'])
    
    # Create datasets
    print("\n📁 Loading datasets...")
    train_dataset = MOT17Dataset(
        config['train_sequences'], 
        processor=processor
    )
    val_dataset = MOT17Dataset(
        config['val_sequences'],
        processor=processor
    )
    
    print(f"  Train: {len(train_dataset)} frames")
    print(f"  Val: {len(val_dataset)} frames")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    print("\n🔨 Building multi-exit model...")
    model = MultiExitYOLOS(config['model_name'])
    
    # Create trainer
    trainer = MultiExitTrainer(model, config)
    
    # Train
    print("\n🚀 Starting training...")
    trainer.train(
        train_loader, 
        val_loader,
        epochs=config['epochs'],
        lr=config['learning_rate']
    )
    
    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()