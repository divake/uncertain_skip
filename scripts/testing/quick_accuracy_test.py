#!/usr/bin/env python3
"""
Quick accuracy test for all models with key metrics
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from ultralytics import YOLO
from tqdm import tqdm

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x, y, w, h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to corners
    box1_x2, box1_y2 = x1 + w1, y1 + h1
    box2_x2, box2_y2 = x2 + w2, y2 + h2
    
    # Intersection
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
        
    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union = w1 * h1 + w2 * h2 - intersection
    
    return intersection / union if union > 0 else 0

def evaluate_model(model_name, num_frames=50):
    """Quick evaluation of a model"""
    
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    # Load model
    model = YOLO(f'{model_name}.pt')
    model.to('cuda')
    
    # Load ground truth
    sequence = 'MOT17-02-FRCNN'
    gt_path = Path(f"data/MOT17/train/{sequence}/gt/gt.txt")
    gt_data = pd.read_csv(gt_path, header=None,
                          names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis'])
    gt_data = gt_data[(gt_data['class'] == 1) & (gt_data['vis'] > 0.3)]
    
    # Load images
    img_path = Path(f"data/MOT17/train/{sequence}/img1")
    images = sorted(img_path.glob("*.jpg"))[:num_frames]
    
    # Metrics
    tp, fp, fn = 0, 0, 0
    total_detections = 0
    confidences = []
    
    for frame_idx, img_file in enumerate(tqdm(images, desc=model_name)):
        frame_num = frame_idx + 1
        
        # Load image
        img = cv2.imread(str(img_file))
        
        # Get ground truth
        frame_gt = gt_data[gt_data['frame'] == frame_num]
        gt_boxes = frame_gt[['x', 'y', 'w', 'h']].values
        
        # Run detection
        results = model(img, conf=0.25, device='cuda', verbose=False)
        
        # Extract person detections
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls) == 0:  # Person
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf)
                    detections.append([x1, y1, x2-x1, y2-y1])
                    confidences.append(conf)
        
        total_detections += len(detections)
        
        # Match detections to GT
        matched_gt = set()
        for det in detections:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_boxes):
                iou = calculate_iou(det, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou > 0.5 and best_gt_idx not in matched_gt:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        fn += len(gt_boxes) - len(matched_gt)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_conf = np.mean(confidences) if confidences else 0
    
    # Model size
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    
    return {
        'model': model_name,
        'params_M': params,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'avg_confidence': avg_conf,
        'detections_per_frame': total_detections / num_frames
    }

def main():
    print("="*80)
    print("QUICK ACCURACY TEST - YOLOV8 MODELS ON MOT17")
    print("="*80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    
    models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
    results = []
    
    for model_name in models:
        result = evaluate_model(model_name, num_frames=50)
        results.append(result)
    
    # Create comparison table
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("ACCURACY COMPARISON RESULTS")
    print("="*80)
    print("\n| Model | Params | Precision | Recall | F1    | Det/Frame | Quality Score |")
    print("|-------|--------|-----------|--------|-------|-----------|---------------|")
    
    for _, row in df.iterrows():
        # Quality score = F1 * sqrt(precision * recall)
        quality = row['f1'] * np.sqrt(row['precision'] * row['recall'])
        print(f"| {row['model']:5s} | {row['params_M']:5.1f}M | {row['precision']:9.3f} | "
              f"{row['recall']:6.3f} | {row['f1']:5.3f} | {row['detections_per_frame']:9.1f} | "
              f"{quality:13.3f} |")
    
    print("\n🎯 KEY FINDINGS:")
    
    # Best model by metric
    print(f"Best Precision: {df.loc[df['precision'].idxmax(), 'model']} ({df['precision'].max():.3f})")
    print(f"Best Recall: {df.loc[df['recall'].idxmax(), 'model']} ({df['recall'].max():.3f})")
    print(f"Best F1: {df.loc[df['f1'].idxmax(), 'model']} ({df['f1'].max():.3f})")
    
    # Calculate improvement from nano to xlarge
    nano_f1 = df[df['model'] == 'yolov8n']['f1'].values[0]
    xlarge_f1 = df[df['model'] == 'yolov8x']['f1'].values[0]
    improvement = (xlarge_f1 - nano_f1) / nano_f1 * 100
    
    print(f"\nF1 Improvement (nano→xlarge): {improvement:.1f}%")
    
    # Save results
    output_file = Path("results/baseline/quick_accuracy_results.csv")
    df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to {output_file}")
    
    return df

if __name__ == "__main__":
    results = main()