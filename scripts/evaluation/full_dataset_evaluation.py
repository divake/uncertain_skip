#!/usr/bin/env python3
"""
Full dataset evaluation for all models on entire MOT17-02 sequence
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
import time
import json

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

def evaluate_model(model_name, num_frames=600):
    """Full evaluation of a model on MOT17-02"""
    
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on {num_frames} frames")
    print(f"{'='*60}")
    
    # Load model
    model = YOLO(f'models/{model_name}.pt')
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
    inference_times = []
    
    # Per-frame metrics for analysis
    frame_metrics = []
    
    for frame_idx, img_file in enumerate(tqdm(images, desc=f"{model_name} detection")):
        frame_num = frame_idx + 1
        
        # Load image
        img = cv2.imread(str(img_file))
        
        # Get ground truth
        frame_gt = gt_data[gt_data['frame'] == frame_num]
        gt_boxes = frame_gt[['x', 'y', 'w', 'h']].values
        
        # Run detection with timing
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        results = model(img, conf=0.25, device='cuda', verbose=False)
        
        torch.cuda.synchronize()
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        inference_times.append(inference_time)
        
        # Extract person detections
        detections = []
        frame_confidences = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls) == 0:  # Person
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf)
                    detections.append([x1, y1, x2-x1, y2-y1])
                    confidences.append(conf)
                    frame_confidences.append(conf)
        
        total_detections += len(detections)
        
        # Match detections to GT
        frame_tp, frame_fp = 0, 0
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
                frame_tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
                frame_fp += 1
        
        frame_fn = len(gt_boxes) - len(matched_gt)
        fn += frame_fn
        
        # Store frame metrics
        frame_metrics.append({
            'frame': frame_num,
            'tp': frame_tp,
            'fp': frame_fp,
            'fn': frame_fn,
            'gt_count': len(gt_boxes),
            'det_count': len(detections),
            'avg_conf': np.mean(frame_confidences) if frame_confidences else 0,
            'inference_ms': inference_time
        })
    
    # Calculate overall metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_conf = np.mean(confidences) if confidences else 0
    
    # Model size
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    
    # Timing stats
    avg_inference = np.mean(inference_times)
    std_inference = np.std(inference_times)
    fps = 1000 / avg_inference if avg_inference > 0 else 0
    
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
        'detections_per_frame': total_detections / num_frames,
        'avg_inference_ms': avg_inference,
        'std_inference_ms': std_inference,
        'fps': fps,
        'total_frames': num_frames,
        'frame_metrics': frame_metrics
    }

def main():
    print("="*80)
    print("FULL DATASET EVALUATION - YOLOV8 MODELS ON MOT17-02")
    print("="*80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Dataset: MOT17-02-FRCNN (600 frames)")
    
    models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
    results = []
    
    for model_name in models:
        result = evaluate_model(model_name, num_frames=600)
        
        # Save frame-level metrics
        frame_df = pd.DataFrame(result['frame_metrics'])
        frame_df.to_csv(f"results/baseline/{model_name}_frame_metrics_600.csv", index=False)
        
        # Remove frame_metrics from main results (too large for summary)
        result_summary = {k: v for k, v in result.items() if k != 'frame_metrics'}
        results.append(result_summary)
    
    # Create comparison table
    df = pd.DataFrame(results)
    
    print("\n" + "="*100)
    print("FULL EVALUATION RESULTS (600 FRAMES)")
    print("="*100)
    print("\n| Model | Params | Precision | Recall | F1    | TP     | FP     | FN     | Det/Frame | FPS   |")
    print("|-------|--------|-----------|--------|-------|--------|--------|--------|-----------|-------|")
    
    for _, row in df.iterrows():
        print(f"| {row['model']:5s} | {row['params_M']:5.1f}M | {row['precision']:9.3f} | "
              f"{row['recall']:6.3f} | {row['f1']:5.3f} | {row['tp']:6d} | {row['fp']:6d} | "
              f"{row['fn']:6d} | {row['detections_per_frame']:9.1f} | {row['fps']:5.1f} |")
    
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
    
    # Detection reliability
    print(f"\nTotal Ground Truth: {df.iloc[0]['tp'] + df.iloc[0]['fn']}")
    print(f"Best Detection Coverage: {df['recall'].max():.1%} ({df.loc[df['recall'].idxmax(), 'model']})")
    
    # Save results
    output_file = Path("results/baseline/full_evaluation_600frames.csv")
    df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to {output_file}")
    
    # Save JSON summary
    json_summary = {
        'evaluation_info': {
            'dataset': 'MOT17-02-FRCNN',
            'frames': 600,
            'confidence_threshold': 0.25,
            'iou_threshold': 0.5
        },
        'models': df.to_dict('records'),
        'best_metrics': {
            'precision': {'model': df.loc[df['precision'].idxmax(), 'model'], 
                         'value': float(df['precision'].max())},
            'recall': {'model': df.loc[df['recall'].idxmax(), 'model'], 
                      'value': float(df['recall'].max())},
            'f1': {'model': df.loc[df['f1'].idxmax(), 'model'], 
                   'value': float(df['f1'].max())}
        }
    }
    
    with open("results/baseline/full_evaluation_600frames.json", 'w') as f:
        json.dump(json_summary, f, indent=2)
    
    return df

if __name__ == "__main__":
    results = main()