"""
Phase 2: Fine-tune YOLOS with Unfrozen Backbone

Continues from Phase 1 checkpoint (frozen backbone, 14.9% mAP).
Unfreezes the entire model and trains end-to-end with lower learning rate.

Key differences from Phase 1:
- Load Phase 1 checkpoint as starting point
- Unfreeze ALL parameters (backbone + detection heads)
- Use lower learning rate (1e-5) to avoid catastrophic forgetting
- Train for 10 epochs with early stopping
- Save best model based on validation mAP
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
        encoding = self.image_processor(
            images=image,
            annotations={'image_id': img_id, 'annotations': anns},
            return_tensors="pt"
        )

        # Remove batch dimension
        pixel_values = encoding['pixel_values'].squeeze(0)

        # Remap class labels to 0 (person is class 0 in COCO-91)
        class_labels = encoding['labels'][0]['class_labels']
        class_labels = torch.zeros_like(class_labels)

        target = {
            'image_id': torch.tensor([img_id]),
            'boxes': encoding['labels'][0]['boxes'],
            'class_labels': class_labels,
            'orig_size': torch.tensor([img_info['height'], img_info['width']])
        }

        return pixel_values, target


def collate_fn(batch):
    """Collate function with proper padding for variable-size images"""
    # Find max dimensions in batch
    max_h = max([item[0].shape[1] for item in batch])
    max_w = max([item[0].shape[2] for item in batch])

    # Pad all images to max dimensions
    padded_images = []
    for item in batch:
        img = item[0]  # [3, H, W]
        c, h, w = img.shape

        # Create padded tensor
        padded = torch.zeros(c, max_h, max_w, dtype=img.dtype)
        padded[:, :h, :w] = img
        padded_images.append(padded)

    pixel_values = torch.stack(padded_images)
    targets = [item[1] for item in batch]
    return pixel_values, targets


def compute_map(model, dataloader, coco_gt, device, threshold=0.05):
    """Compute mAP on validation set"""
    model.eval()
    results = []

    for pixel_values, targets in tqdm(dataloader, desc="Validation", leave=False):
        pixel_values = pixel_values.to(device)

        with torch.no_grad():
            outputs = model(pixel_values)

        logits = outputs.logits
        pred_boxes = outputs.pred_boxes

        for i, target in enumerate(targets):
            image_id = target['image_id'].item()
            orig_h, orig_w = target['orig_size'].tolist()

            # Get person class probabilities
            class_probs = logits[i].softmax(-1)
            person_probs = class_probs[:, 0]  # Person is class 0

            # Get class labels
            max_probs, labels = class_probs[:, :-1].max(-1)

            # Filter by threshold
            keep = person_probs >= threshold

            if keep.sum() == 0:
                continue

            filtered_boxes = pred_boxes[i][keep]
            filtered_scores = person_probs[keep]
            filtered_labels = labels[keep]

            # Convert to COCO format
            for score, label, box in zip(filtered_scores, filtered_labels, filtered_boxes):
                # Only keep person class (class 0)
                if int(label) != 0:
                    continue

                cx, cy, w, h = box.tolist()

                # Convert normalized [cx, cy, w, h] to absolute [x, y, w, h]
                x = (cx - w/2) * orig_w
                y = (cy - h/2) * orig_h
                w_abs = w * orig_w
                h_abs = h * orig_h

                results.append({
                    'image_id': image_id,
                    'category_id': 1,  # Person
                    'bbox': [float(x), float(y), float(w_abs), float(h_abs)],
                    'score': float(score)
                })

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

    map_score = coco_eval.stats[0]
    map50 = coco_eval.stats[1]

    os.unlink(results_file)

    return map_score, map50, coco_eval.stats


def train_phase2(args):
    """Phase 2 training: Unfreeze backbone and train end-to-end"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load image processor
    print("\nLoading YOLOS processor...")
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-small")

    # Load model architecture
    print("Loading YOLOS model architecture...")
    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")

    # Load Phase 1 checkpoint
    print(f"\nLoading Phase 1 checkpoint: {args.phase1_checkpoint}")
    checkpoint = torch.load(args.phase1_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded Phase 1 model from Epoch {checkpoint['epoch']}")
    print(f"Phase 1 best mAP: {checkpoint.get('best_map', 'N/A')}")

    model.to(device)

    # ALL parameters are trainable (backbone + detection heads)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    print("Phase 2: Training ENTIRE model (backbone + detection heads)")
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

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    coco_gt = val_dataset.coco

    # Optimizer with LOWER learning rate for end-to-end training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.1
    )

    # Training loop
    best_map = checkpoint.get('best_map', 0.0)  # Start from Phase 1 best mAP
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_log = []

    print(f"{'='*80}")
    print(f"Starting Phase 2 Training")
    print(f"{'='*80}\n")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Output dir: {output_dir}")
    print()

    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}\n")

        # Training
        model.train()
        epoch_loss = 0
        num_batches = 0

        for pixel_values, targets in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            pixel_values = pixel_values.to(device)

            # Prepare targets
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Validation
        print(f"\nValidation...")
        val_map, val_map50, coco_stats = compute_map(
            model, val_loader, coco_gt, device, threshold=0.05
        )

        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Val mAP: {val_map:.4f} ({val_map*100:.2f}%)")
        print(f"  Val mAP@0.50: {val_map50:.4f} ({val_map50*100:.2f}%)")
        print(f"  Learning Rate: {current_lr:.2e}")

        # Save best model
        if val_map > best_map:
            best_map = val_map
            checkpoint_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_map': best_map,
            }, checkpoint_path)
            print(f"  ✓ New best model saved! mAP: {best_map:.4f}")

        # Log results
        training_log.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'val_map': val_map,
            'val_map50': val_map50,
            'learning_rate': current_lr,
            'coco_stats': [float(s) for s in coco_stats]
        })

        # Save training log
        with open(output_dir / 'training_log.json', 'w') as f:
            json.dump(training_log, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Phase 2 Training Complete!")
    print(f"{'='*80}\n")
    print(f"Best validation mAP: {best_map:.4f} ({best_map*100:.2f}%)")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")
    print(f"Training log saved to: {output_dir / 'training_log.json'}")


def main():
    parser = argparse.ArgumentParser('YOLOS Phase 2 fine-tuning - Unfreeze backbone')

    # Model parameters
    parser.add_argument('--phase1_checkpoint',
                        default='models/yolos_finetuned_layer12/best_model.pth',
                        help='Path to Phase 1 checkpoint')

    # Dataset parameters
    parser.add_argument('--coco_path', default='/ssd_4TB/divake/uncertain_skip/data/MOT17/train')
    parser.add_argument('--coco_json_train', default='data/mot17_coco/train_coco.json')
    parser.add_argument('--coco_json_val', default='data/mot17_coco/val_coco.json')
    parser.add_argument('--image_size', type=int, default=800)

    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate (lower for end-to-end training)')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=0)

    # Output
    parser.add_argument('--output_dir', default='models/yolos_finetuned_phase2')

    args = parser.parse_args()

    train_phase2(args)


if __name__ == '__main__':
    main()
