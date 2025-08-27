#!/usr/bin/env python3
"""
Compare fixed model performance vs adaptive model selection
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd

class FixedModelTracker:
    """Track with a single fixed model for comparison"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = YOLO(f'models/{model_name}.pt')
        self.model.to('cuda')
        self.iou_threshold = 0.3
        
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 < xi1 or yi2 < yi1:
            return 0.0
            
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def track_sequence(self, img_path, initial_bbox, max_frames=600):
        """Track object through sequence with fixed model"""
        images = sorted(Path(img_path).glob("*.jpg"))[:max_frames]
        
        results = []
        current_bbox = initial_bbox
        lost_frames = 0
        
        for frame_idx, img_file in enumerate(images):
            frame = cv2.imread(str(img_file))
            
            # Run detection
            detections = self.model(frame, conf=0.25, device='cuda', verbose=False)
            
            # Find best match
            best_match = None
            best_iou = 0
            best_conf = 0
            
            if detections[0].boxes is not None:
                for box in detections[0].boxes:
                    if int(box.cls) == 0:  # Person
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        det_bbox = [x1, y1, x2-x1, y2-y1]
                        
                        iou = self.calculate_iou(current_bbox, det_bbox)
                        
                        if iou > best_iou and iou > self.iou_threshold:
                            best_iou = iou
                            best_match = det_bbox
                            best_conf = float(box.conf)
            
            if best_match is not None:
                current_bbox = best_match
                confidence = best_conf
                status = 'tracked'
                lost_frames = 0
            else:
                confidence = 0.0
                lost_frames += 1
                status = 'lost' if lost_frames > 5 else 'searching'
            
            results.append({
                'frame': frame_idx,
                'confidence': confidence,
                'status': status
            })
            
            # Stop if object lost
            if lost_frames > 5:
                break
        
        return results

def run_comparison():
    """Run comprehensive comparison between fixed and adaptive tracking"""
    
    print("="*60)
    print("FIXED vs ADAPTIVE MODEL COMPARISON")
    print("="*60)
    
    # Load adaptive results
    adaptive_results_file = Path("results/adaptive/tracking_results.json")
    if adaptive_results_file.exists():
        with open(adaptive_results_file, 'r') as f:
            adaptive_data = json.load(f)
    else:
        print("Please run enhanced_adaptive_tracker.py first!")
        return
    
    # Get initial object from adaptive tracking
    img_path = "data/MOT17/train/MOT17-02-FRCNN/img1"
    first_frame = cv2.imread(str(sorted(Path(img_path).glob("*.jpg"))[0]))
    
    # Use medium confidence object selection
    temp_model = YOLO('models/yolov8m.pt')
    temp_model.to('cuda')
    results = temp_model(first_frame, conf=0.25, device='cuda', verbose=False)
    
    initial_bbox = None
    if results[0].boxes is not None:
        candidates = []
        for box in results[0].boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                candidates.append({
                    'bbox': [x1, y1, x2-x1, y2-y1],
                    'confidence': conf
                })
        
        if candidates:
            # Select medium confidence object
            target_conf = 0.6
            selected = min(candidates, key=lambda x: abs(x['confidence'] - target_conf))
            initial_bbox = selected['bbox']
    
    if initial_bbox is None:
        print("No initial object found!")
        return
    
    # Run fixed model tracking for each model
    models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
    fixed_results = {}
    
    for model_name in models:
        print(f"\nTesting fixed {model_name}...")
        tracker = FixedModelTracker(model_name)
        results = tracker.track_sequence(img_path, initial_bbox, max_frames=600)
        
        # Calculate metrics
        tracked = [r for r in results if r['status'] == 'tracked']
        tracking_rate = len(tracked) / len(results) if results else 0
        avg_confidence = np.mean([r['confidence'] for r in tracked if r['confidence'] > 0])
        
        fixed_results[model_name] = {
            'results': results,
            'tracking_rate': tracking_rate,
            'avg_confidence': avg_confidence,
            'total_frames': len(results),
            'tracked_frames': len(tracked)
        }
        
        print(f"  Tracking rate: {tracking_rate:.2%}")
        print(f"  Avg confidence: {avg_confidence:.3f}")
        print(f"  Frames before lost: {len(results)}")
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Fixed vs Adaptive Model Comparison', fontsize=16, fontweight='bold')
    
    # 1. Tracking Success Rate
    ax = axes[0, 0]
    models_list = list(fixed_results.keys())
    tracking_rates = [fixed_results[m]['tracking_rate'] for m in models_list]
    adaptive_rate = adaptive_data['summary']['tracking_rate']
    
    bars = ax.bar(models_list + ['Adaptive'], 
                   tracking_rates + [adaptive_rate],
                   color=['green', 'yellow', 'orange', 'magenta', 'red', 'blue'])
    ax.set_ylabel('Tracking Success Rate')
    ax.set_title('Tracking Success Rate Comparison')
    ax.set_ylim(0, 1.0)
    for bar, rate in zip(bars, tracking_rates + [adaptive_rate]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.2%}', ha='center', va='bottom')
    ax.grid(True, alpha=0.3)
    
    # 2. Average Confidence
    ax = axes[0, 1]
    avg_confidences = [fixed_results[m]['avg_confidence'] for m in models_list]
    adaptive_conf = adaptive_data['summary']['avg_confidence']
    
    bars = ax.bar(models_list + ['Adaptive'],
                   avg_confidences + [adaptive_conf],
                   color=['green', 'yellow', 'orange', 'magenta', 'red', 'blue'])
    ax.set_ylabel('Average Confidence')
    ax.set_title('Average Tracking Confidence')
    ax.set_ylim(0, 1.0)
    for bar, conf in zip(bars, avg_confidences + [adaptive_conf]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{conf:.3f}', ha='center', va='bottom')
    ax.grid(True, alpha=0.3)
    
    # 3. Frames Tracked
    ax = axes[1, 0]
    frames_tracked = [fixed_results[m]['tracked_frames'] for m in models_list]
    adaptive_frames = adaptive_data['summary']['tracked_frames']
    
    bars = ax.bar(models_list + ['Adaptive'],
                   frames_tracked + [adaptive_frames],
                   color=['green', 'yellow', 'orange', 'magenta', 'red', 'blue'])
    ax.set_ylabel('Frames Successfully Tracked')
    ax.set_title('Total Frames Tracked (out of 600)')
    for bar, frames in zip(bars, frames_tracked + [adaptive_frames]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{frames}', ha='center', va='bottom')
    ax.grid(True, alpha=0.3)
    
    # 4. Model Efficiency (Parameters vs Performance)
    ax = axes[1, 1]
    model_params = [3.2, 11.2, 25.9, 43.7, 68.2]  # Model parameters in millions
    adaptive_params = adaptive_data['summary']['avg_model_params']
    
    # Plot fixed models
    for i, model in enumerate(models_list):
        ax.scatter(model_params[i], fixed_results[model]['tracking_rate'],
                  s=200, alpha=0.7, label=model)
    
    # Plot adaptive
    ax.scatter(adaptive_params, adaptive_rate, 
              s=300, color='blue', marker='*', label='Adaptive', 
              edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Model Parameters (M)')
    ax.set_ylabel('Tracking Success Rate')
    ax.set_title('Model Efficiency: Parameters vs Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison plot
    output_path = Path("results/adaptive/fixed_vs_adaptive_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to {output_path}")
    
    # Create summary table
    summary = pd.DataFrame({
        'Model': models_list + ['Adaptive'],
        'Parameters (M)': model_params + [adaptive_params],
        'Tracking Rate': tracking_rates + [adaptive_rate],
        'Avg Confidence': avg_confidences + [adaptive_conf],
        'Frames Tracked': frames_tracked + [adaptive_frames]
    })
    
    # Calculate efficiency score (tracking_rate / params)
    summary['Efficiency'] = summary['Tracking Rate'] / summary['Parameters (M)']
    
    # Save summary table
    summary.to_csv("results/adaptive/comparison_summary.csv", index=False)
    
    print("\n📊 COMPARISON SUMMARY TABLE:")
    print(summary.to_string(index=False))
    
    return summary

if __name__ == "__main__":
    summary = run_comparison()