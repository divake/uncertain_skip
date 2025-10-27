"""
Fine-tune YOLOS Layer 12 Detection Head on MOT17

Uses HuggingFace Transformers API with official YOLOS hyperparameters.
Focuses on fine-tuning ONLY the detection heads (freeze backbone).

Key Features:
- Uses official YOLOS hyperparameters (lr, loss weights, scheduler)
- Validation after each epoch
- Saves only best model (based on validation mAP)
- Comprehensive logging
"""

import os
import json
import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import YolosForObjectDetection, YolosImageProcessor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tempfile


class MOT17Dataset(Dataset):
    """MOT17 dataset in COCO format for YOLOS fine-tuning"""

    def __init__(self, img_folder, ann_file, image_processor, image_size=800, augment=False):
        self.img_folder = Path(img_folder)
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.image_processor = image_processor
        self.image_size = image_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]

        # Load image
        file_parts = img_info['file_name'].split('/')
        if len(file_parts) == 2:
            seq_name, frame_name = file_parts
            path = self.img_folder / seq_name / "img1" / frame_name
        else:
            path = self.img_folder / img_info['file_name']

        image = Image.open(path).convert('RGB')

        # Process image with YOLOS processor
        # The processor expects PIL image and annotations in COCO format
        encoding = self.image_processor(
            images=image,
            annotations={'image_id': img_id, 'annotations': anns},
            return_tensors="pt"
        )

        # Remove batch dimension
        pixel_values = encoding['pixel_values'].squeeze(0)

        # Prepare target from processor output
        # Include original image size for proper bbox scaling during evaluation
        # CRITICAL FIX: Remap class labels to 0 (person is class 0 in COCO-91)
        # The processor keeps the category_id from annotations (1 for person)
        # But YOLOS expects class 0 for person
        class_labels = encoding['labels'][0]['class_labels']
        # Remap: category_id 1 (person) -> class 0
        class_labels = torch.zeros_like(class_labels)

        target = {
            'image_id': torch.tensor([img_id]),
            'boxes': encoding['labels'][0]['boxes'],
            'class_labels': class_labels,
            'orig_size': torch.tensor([img_info['height'], img_info['width']])  # (height, width)
        }

        return pixel_values, target


def collate_fn(batch):
    """Custom collate function for batching with padding"""
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Find max dimensions in batch
    max_h = max([pv.shape[1] for pv in pixel_values])
    max_w = max([pv.shape[2] for pv in pixel_values])

    # Pad all images to same size
    padded_values = []
    for pv in pixel_values:
        c, h, w = pv.shape
        padded = torch.zeros((c, max_h, max_w), dtype=pv.dtype)
        padded[:, :h, :w] = pv
        padded_values.append(padded)

    # Stack padded tensors
    pixel_values = torch.stack(padded_values, dim=0)

    return pixel_values, labels


def compute_map(model, dataloader, device, coco_gt, image_processor):
    """Compute mAP using COCO evaluation"""
    model.eval()

    results = []

    debug_batch_count = 0
    with torch.no_grad():
        for pixel_values, targets in tqdm(dataloader, desc="Evaluating"):
            pixel_values = pixel_values.to(device)

            outputs = model(pixel_values=pixel_values)

            # Debug: Print raw model outputs for first batch
            if debug_batch_count == 0:
                print(f"\nDEBUG Batch 0:")
                print(f"  logits shape: {outputs.logits.shape}")
                print(f"  logits min/max: {outputs.logits.min().item():.4f} / {outputs.logits.max().item():.4f}")
                print(f"  pred_boxes shape: {outputs.pred_boxes.shape}")
                print(f"  pred_boxes min/max: {outputs.pred_boxes.min().item():.4f} / {outputs.pred_boxes.max().item():.4f}")
                # Get class probabilities using softmax for person class (class 0)
                probs = outputs.logits.softmax(-1)[0, :, :-1]  # [num_queries, num_classes], exclude no-object
                person_probs = probs[:, 0]  # Person is class 0
                max_probs, _ = probs.max(-1)  # Max prob across all classes
                print(f"  Person class prob: min={person_probs.min().item():.4f}, max={person_probs.max().item():.4f}, mean={person_probs.mean().item():.4f}")
                print(f"  Max class prob per query: min={max_probs.min().item():.4f}, max={max_probs.max().item():.4f}, mean={max_probs.mean().item():.4f}")
                print(f"  Num person predictions > 0.01: {(person_probs > 0.01).sum().item()}")
                print(f"  Num person predictions > 0.1: {(person_probs > 0.1).sum().item()}")
                debug_batch_count += 1

            # Post-process predictions
            # Get actual image sizes from targets
            target_sizes = torch.stack([target['orig_size'] for target in targets]).to(device)
            results_batch = image_processor.post_process_object_detection(
                outputs,
                threshold=0.05,  # Reasonable threshold for pretrained model
                target_sizes=target_sizes
            )

            # Convert to COCO format
            for i, (result, target) in enumerate(zip(results_batch, targets)):
                image_id = target['image_id'].item()

                scores = result['scores'].cpu().numpy()
                labels = result['labels'].cpu().numpy()
                boxes = result['boxes'].cpu().numpy()

                for score, label, box in zip(scores, labels, boxes):
                    # FILTER: Only keep person class (class 0 in COCO-91)
                    if int(label) != 0:
                        continue

                    x1, y1, x2, y2 = box
                    # Person is category_id 1 in COCO format
                    category_id = 1
                    results.append({
                        'image_id': image_id,
                        'category_id': category_id,
                        'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        'score': float(score)
                    })

    # Evaluate using COCO API
    if len(results) == 0:
        print("\nWARNING: No predictions to evaluate!")
        return 0.0, 0.0, [0.0] * 12

    # Save results to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(results, f)
        results_file = f.name

    print(f"\nSaved {len(results)} predictions to {results_file}")

    coco_dt = coco_gt.loadRes(results_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map_score = coco_eval.stats[0]  # mAP @ IoU=0.50:0.95
    map50 = coco_eval.stats[1]  # mAP @ IoU=0.50

    os.unlink(results_file)

    return map_score, map50, coco_eval.stats


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for pixel_values, targets in progress_bar:
        pixel_values = pixel_values.to(device)

        # Move targets to device
        labels = []
        for target in targets:
            labels.append({
                'class_labels': target['class_labels'].to(device),
                'boxes': target['boxes'].to(device)
            })

        # Forward pass
        outputs = model(pixel_values=pixel_values, labels=labels)

        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (official YOLOS uses 0.1)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def main(args):
    print("\n" + "="*80)
    print("FINE-TUNING YOLOS LAYER 12 ON MOT17")
    print("="*80)
    print(f"Task: Person detection (single class)")
    print(f"Strategy: Fine-tune detection heads only (freeze backbone)")
    print(f"Using: HuggingFace Transformers + Official YOLOS hyperparameters")
    print()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model and processor
    print("Loading pretrained YOLOS-small from HuggingFace...")
    # Keep the pretrained 91-class detection head (person is class 0 in COCO)
    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")

    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-small")
    model.to(device)

    # Freeze backbone if requested
    if args.freeze_backbone:
        print("\n" + "="*80)
        print("FREEZING BACKBONE")
        print("="*80)

        frozen_params = 0
        trainable_params = 0

        for name, param in model.named_parameters():
            if "vit" in name:  # Backbone
                param.requires_grad = False
                frozen_params += param.numel()
            else:  # Detection heads
                param.requires_grad = True
                trainable_params += param.numel()

        print(f"Frozen parameters (backbone): {frozen_params:,}")
        print(f"Trainable parameters (detection heads): {trainable_params:,}")
        print(f"Percentage trainable: {100 * trainable_params / (frozen_params + trainable_params):.2f}%")
        print()

    # Count total trainable parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {n_parameters:,}")
    print()

    # Load datasets
    print("Loading datasets...")
    train_dataset = MOT17Dataset(
        args.coco_path,
        args.coco_json_train,
        image_processor,
        image_size=args.image_size,
        augment=True
    )

    val_dataset = MOT17Dataset(
        args.coco_path,
        args.coco_json_val,
        image_processor,
        image_size=args.image_size,
        augment=False
    )

    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")
    print()

    # Dataloaders
    # Use persistent_workers to avoid recreating workers each epoch
    # Use num_workers=0 temporarily to debug worker crashes
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid worker crashes
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid worker crashes
        collate_fn=collate_fn,
        pin_memory=True
    )

    # COCO API for evaluation
    coco_gt = val_dataset.coco

    # Optimizer (official YOLOS uses AdamW with lr=1e-4, weight_decay=1e-4)
    # We use lr=5e-5 (lower) for fine-tuning
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Scheduler (warmup cosine)
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.lr_drop,
        T_mult=1,
        eta_min=args.min_lr
    )

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_file = output_dir / 'config.json'
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Configuration saved to {config_file}")
    print()

    # Training loop
    print("="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Output directory: {output_dir}")
    print()

    best_map = 0.0
    best_epoch = -1
    training_log = []

    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*80}\n")

        # Training
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch+1)

        # Validation
        print(f"\nRunning validation...")
        map_score, map50, coco_stats = compute_map(model, val_loader, device, coco_gt, image_processor)

        # Step scheduler
        scheduler.step()

        # Log stats
        log_stats = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'mAP': map_score,
            'mAP50': map50,
            'coco_stats': coco_stats.tolist() if hasattr(coco_stats, 'tolist') else coco_stats,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'n_parameters': n_parameters,
        }

        training_log.append(log_stats)

        # Save log
        log_file = output_dir / 'training_log.json'
        with open(log_file, 'w') as f:
            json.dump(training_log, f, indent=2)

        # Print summary
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1} Summary")
        print(f"{'='*80}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"mAP (0.50:0.95): {map_score:.6f} ({map_score*100:.4f}%)")
        print(f"mAP50: {map50:.6f} ({map50*100:.4f}%)")
        print(f"Best mAP: {best_map:.6f} ({best_map*100:.4f}%) at epoch {best_epoch+1}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*80}\n")

        # Save best model only
        if map_score > best_map:
            best_map = map_score
            best_epoch = epoch

            best_model_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'mAP': map_score,
                'mAP50': map50,
                'args': vars(args),
            }, best_model_path)

            print(f"✅ New best model saved! mAP: {map_score:.4f}, mAP50: {map50:.4f}")
            print(f"   Saved to: {best_model_path}\n")

    # Final summary
    print("\n" + "="*80)
    print("FINE-TUNING COMPLETE")
    print("="*80)
    print(f"Best mAP (0.50:0.95): {best_map:.4f} (epoch {best_epoch+1})")
    print(f"Best model: {output_dir / 'best_model.pth'}")
    print(f"Training log: {output_dir / 'training_log.json'}")
    print(f"Total epochs: {args.epochs}")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('YOLOS Layer 12 fine-tuning')

    # Model
    parser.add_argument('--freeze_backbone', action='store_true', default=True)

    # Training
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='Learning rate (official: 1e-4, we use 5e-5 for fine-tuning)')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='Weight decay (official YOLOS value)')
    parser.add_argument('--lr_drop', default=40, type=int,
                        help='LR drop epoch for scheduler')
    parser.add_argument('--min_lr', default=1e-7, type=float)
    parser.add_argument('--image_size', default=800, type=int)

    # Dataset
    parser.add_argument('--coco_path', default='/ssd_4TB/divake/uncertain_skip/data/MOT17/train')
    parser.add_argument('--coco_json_train', default='data/mot17_coco/train_coco.json')
    parser.add_argument('--coco_json_val', default='data/mot17_coco/val_coco.json')

    # System
    parser.add_argument('--output_dir', default='models/yolos_finetuned_layer12')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    args = parser.parse_args()

    main(args)
