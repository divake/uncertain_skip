"""
Detailed evaluation of fine-tuned YOLOS model.

Metrics:
- mAP@0.50
- mAP@0.75
- Precision
- Recall
- F1 Score
"""

import torch
import numpy as np
from transformers import YolosForObjectDetection, YolosImageProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import tempfile
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# MOT17 COCO annotations path
TRAIN_ANNOTATIONS = 'data/mot17_coco/train_coco.json'
VAL_ANNOTATIONS = 'data/mot17_coco/val_coco.json'
IMG_FOLDER = '/ssd_4TB/divake/uncertain_skip/data/MOT17/train'

# Model paths
CHECKPOINT_PATH = 'models/yolos_finetuned_layer12/best_model.pth'


class MOT17Dataset(torch.utils.data.Dataset):
    """MOT17 dataset in COCO format"""

    def __init__(self, img_folder, coco_json_path, processor):
        self.img_folder = img_folder
        self.coco = COCO(coco_json_path)
        self.processor = processor
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]

        # Construct image path
        from pathlib import Path
        file_parts = image_info['file_name'].split('/')
        if len(file_parts) == 2:
            seq_name, frame_name = file_parts
            image_path = Path(self.img_folder) / seq_name / "img1" / frame_name
        else:
            image_path = Path(self.img_folder) / image_info['file_name']

        # Load image
        from PIL import Image
        image = Image.open(image_path).convert('RGB')

        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        # Convert to YOLOS format
        boxes = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            # Convert to [cx, cy, w, h] normalized
            cx = (x + w/2) / image_info['width']
            cy = (y + h/2) / image_info['height']
            w_norm = w / image_info['width']
            h_norm = h / image_info['height']
            boxes.append([cx, cy, w_norm, h_norm])

        # Process with YOLOS processor
        encoding = self.processor(
            images=image,
            annotations=[{
                'image_id': image_id,
                'annotations': annotations
            }],
            return_tensors='pt'
        )

        # Prepare target
        if len(boxes) > 0:
            # Remap to class 0 (person)
            class_labels = torch.zeros(len(boxes), dtype=torch.long)
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        else:
            class_labels = torch.zeros(0, dtype=torch.long)
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        target = {
            'image_id': torch.tensor([image_id]),
            'boxes': boxes_tensor,
            'class_labels': class_labels,
            'orig_size': torch.tensor([image_info['height'], image_info['width']])
        }

        pixel_values = encoding['pixel_values'].squeeze(0)

        return pixel_values, target


def collate_fn(batch):
    """Collate function for dataloader"""
    pixel_values = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return pixel_values, targets


def evaluate_model(model, dataloader, coco_gt, device, conf_threshold=0.05):
    """
    Evaluate model and compute detailed metrics.

    Returns:
        dict: All evaluation metrics
    """
    model.eval()

    all_predictions = []

    print(f"\nRunning inference on validation set...")

    for pixel_values, targets in tqdm(dataloader, desc="Evaluating"):
        pixel_values = pixel_values.to(device)

        with torch.no_grad():
            outputs = model(pixel_values)

        # Process predictions
        logits = outputs.logits  # [B, 100, 92]
        pred_boxes = outputs.pred_boxes  # [B, 100, 4]

        for i, target in enumerate(targets):
            image_id = target['image_id'].item()
            orig_h, orig_w = target['orig_size'].tolist()

            # Get class probabilities
            class_probs = logits[i].softmax(-1)  # [100, 92]
            person_probs = class_probs[:, 0]  # Person is class 0

            # Get all class labels and max probabilities
            max_probs, labels = class_probs[:, :-1].max(-1)  # Exclude no-object class

            # Filter predictions
            keep = person_probs >= conf_threshold

            if keep.sum() == 0:
                continue

            filtered_boxes = pred_boxes[i][keep]
            filtered_scores = person_probs[keep]
            filtered_labels = labels[keep]

            # Convert boxes to COCO format [x, y, w, h] in absolute coordinates
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

                all_predictions.append({
                    'image_id': image_id,
                    'category_id': 1,  # Person category in COCO
                    'bbox': [float(x), float(y), float(w_abs), float(h_abs)],
                    'score': float(score)
                })

    if len(all_predictions) == 0:
        print("\nWARNING: No predictions generated!")
        return None

    print(f"\nGenerated {len(all_predictions)} predictions")

    # Save predictions to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(all_predictions, f)
        results_file = f.name

    print(f"Saved predictions to: {results_file}")

    # Run COCO evaluation
    print("\nRunning COCO evaluation...")
    coco_dt = coco_gt.loadRes(results_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract metrics
    stats = coco_eval.stats

    # Compute Precision, Recall, F1 at IoU=0.5
    print("\nComputing Precision, Recall, F1...")

    # Get precision and recall at IoU=0.5
    precision = coco_eval.eval['precision'][0, :, 0, 0, 2]  # IoU=0.5, all areas, maxDets=100
    recall = coco_eval.eval['recall'][0, 0, 0, 2]  # IoU=0.5, all areas, maxDets=100

    # Average precision and recall across recall/precision thresholds
    valid_precision = precision[precision > -1]
    avg_precision_50 = valid_precision.mean() if len(valid_precision) > 0 else 0

    # Recall at IoU=0.5
    avg_recall_50 = recall if recall > -1 else 0

    # F1 Score
    if avg_precision_50 + avg_recall_50 > 0:
        f1_score_50 = 2 * (avg_precision_50 * avg_recall_50) / (avg_precision_50 + avg_recall_50)
    else:
        f1_score_50 = 0

    # Also compute at IoU=0.75
    precision_75 = coco_eval.eval['precision'][5, :, 0, 0, 2]  # IoU=0.75
    recall_75 = coco_eval.eval['recall'][5, 0, 0, 2]

    valid_precision_75 = precision_75[precision_75 > -1]
    avg_precision_75 = valid_precision_75.mean() if len(valid_precision_75) > 0 else 0
    avg_recall_75 = recall_75 if recall_75 > -1 else 0

    if avg_precision_75 + avg_recall_75 > 0:
        f1_score_75 = 2 * (avg_precision_75 * avg_recall_75) / (avg_precision_75 + avg_recall_75)
    else:
        f1_score_75 = 0

    # Clean up
    os.unlink(results_file)

    results = {
        'mAP': float(stats[0]),  # mAP@0.50:0.95
        'mAP@0.50': float(stats[1]),  # mAP@0.50
        'mAP@0.75': float(stats[2]),  # mAP@0.75
        'mAP_small': float(stats[3]),  # mAP for small objects
        'mAP_medium': float(stats[4]),  # mAP for medium objects
        'mAP_large': float(stats[5]),  # mAP for large objects
        'Precision@0.50': float(avg_precision_50),
        'Recall@0.50': float(avg_recall_50),
        'F1@0.50': float(f1_score_50),
        'Precision@0.75': float(avg_precision_75),
        'Recall@0.75': float(avg_recall_75),
        'F1@0.75': float(f1_score_75),
        'num_predictions': len(all_predictions),
    }

    return results


def main():
    """Main evaluation function"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load processor
    print("\nLoading YOLOS processor...")
    processor = YolosImageProcessor.from_pretrained("hustvl/yolos-small")

    # Load model
    print("Loading fine-tuned model...")
    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint from: {CHECKPOINT_PATH}")
    print(f"  Epoch: {checkpoint['epoch']}")

    # Load validation dataset
    print("\nLoading validation dataset...")
    val_dataset = MOT17Dataset(IMG_FOLDER, VAL_ANNOTATIONS, processor)
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    print(f"Validation set: {len(val_dataset)} images")

    # Load COCO ground truth
    coco_gt = COCO(VAL_ANNOTATIONS)

    # Evaluate
    results = evaluate_model(model, val_loader, coco_gt, device, conf_threshold=0.05)

    if results is None:
        print("\nEvaluation failed!")
        return

    # Print results
    print(f"\n{'='*80}")
    print("FINE-TUNED MODEL EVALUATION RESULTS")
    print(f"{'='*80}\n")

    print(f"mAP Metrics:")
    print(f"  mAP@0.50:0.95: {results['mAP']:.4f} ({results['mAP']*100:.2f}%)")
    print(f"  mAP@0.50:      {results['mAP@0.50']:.4f} ({results['mAP@0.50']*100:.2f}%)")
    print(f"  mAP@0.75:      {results['mAP@0.75']:.4f} ({results['mAP@0.75']*100:.2f}%)")
    print()

    print(f"Object Size Breakdown:")
    print(f"  mAP (small):   {results['mAP_small']:.4f} ({results['mAP_small']*100:.2f}%)")
    print(f"  mAP (medium):  {results['mAP_medium']:.4f} ({results['mAP_medium']*100:.2f}%)")
    print(f"  mAP (large):   {results['mAP_large']:.4f} ({results['mAP_large']*100:.2f}%)")
    print()

    print(f"Precision, Recall, F1 @ IoU=0.50:")
    print(f"  Precision:     {results['Precision@0.50']:.4f} ({results['Precision@0.50']*100:.2f}%)")
    print(f"  Recall:        {results['Recall@0.50']:.4f} ({results['Recall@0.50']*100:.2f}%)")
    print(f"  F1 Score:      {results['F1@0.50']:.4f} ({results['F1@0.50']*100:.2f}%)")
    print()

    print(f"Precision, Recall, F1 @ IoU=0.75:")
    print(f"  Precision:     {results['Precision@0.75']:.4f} ({results['Precision@0.75']*100:.2f}%)")
    print(f"  Recall:        {results['Recall@0.75']:.4f} ({results['Recall@0.75']*100:.2f}%)")
    print(f"  F1 Score:      {results['F1@0.75']:.4f} ({results['F1@0.75']*100:.2f}%)")
    print()

    print(f"Total predictions: {results['num_predictions']}")

    # Save results
    output_file = 'models/yolos_finetuned_layer12/detailed_evaluation.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
