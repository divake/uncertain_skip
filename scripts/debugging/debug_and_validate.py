#!/usr/bin/env python3
"""
Debug and validate YOLOv8 performance issues
- Verify GPU usage and identify bottlenecks
- Measure detection quality against ground truth
- Proper benchmarking with timing breakdown
"""

import torch
import numpy as np
import pandas as pd
import time
from pathlib import Path
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class PerformanceDebugger:
    def __init__(self):
        # GPU verification
        print("="*80)
        print("GPU AND SYSTEM VERIFICATION")
        print("="*80)
        
        # Check CUDA
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"Current GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            
            # Set optimization flags
            torch.backends.cudnn.benchmark = True
            print("✓ Set cudnn.benchmark = True for optimization")
        else:
            print("⚠️ WARNING: Running on CPU - expect slow performance!")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timing_data = defaultdict(list)
        
    def profile_inference_pipeline(self, model_name='yolov8n', num_frames=100):
        """Profile each component of the inference pipeline"""
        
        print(f"\n{'='*80}")
        print(f"PROFILING {model_name.upper()} INFERENCE PIPELINE")
        print(f"{'='*80}")
        
        # Load model
        model = YOLO(f"{model_name}.pt")
        
        # Verify model is on GPU
        print(f"Model device: {next(model.model.parameters()).device}")
        
        # Create dummy frames for consistent testing
        print(f"\nCreating {num_frames} test frames...")
        test_frames = []
        for i in range(num_frames):
            # Create frames with varying complexity
            frame = np.ones((640, 640, 3), dtype=np.uint8) * 128
            # Add random rectangles to simulate objects
            num_objects = np.random.randint(5, 20)
            for _ in range(num_objects):
                x1, y1 = np.random.randint(0, 500, 2)
                x2, y2 = x1 + np.random.randint(50, 150), y1 + np.random.randint(50, 150)
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            test_frames.append(frame)
        
        # Warm-up GPU (important!)
        print("\nWarming up GPU...")
        for _ in range(10):
            _ = model(test_frames[0], verbose=False)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # Profile each component
        print(f"\nProfiling {num_frames} frames...")
        
        total_times = {
            'preprocessing': [],
            'inference': [],
            'postprocessing': [],
            'total': []
        }
        
        for i, frame in enumerate(test_frames):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Total time
            total_start = time.perf_counter()
            
            # Run inference (this includes all steps internally)
            results = model(frame, conf=0.25, iou=0.45, verbose=False)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            total_time = time.perf_counter() - total_start
            total_times['total'].append(total_time)
            
            # Extract detections
            num_detections = 0
            if results[0].boxes is not None:
                num_detections = len(results[0].boxes)
            
            if i % 20 == 0:
                print(f"  Frame {i+1}/{num_frames}: {total_time*1000:.2f}ms, {num_detections} detections")
        
        # Calculate statistics
        avg_total = np.mean(total_times['total'])
        std_total = np.std(total_times['total'])
        fps = 1.0 / avg_total
        
        print(f"\n📊 TIMING RESULTS FOR {model_name}:")
        print(f"  Average inference time: {avg_total*1000:.2f} ± {std_total*1000:.2f} ms")
        print(f"  FPS: {fps:.1f}")
        print(f"  Min time: {min(total_times['total'])*1000:.2f} ms")
        print(f"  Max time: {max(total_times['total'])*1000:.2f} ms")
        
        # Check GPU memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Peak GPU memory: {memory_used:.3f} GB")
            torch.cuda.reset_peak_memory_stats()
        
        return {
            'model': model_name,
            'avg_time_ms': avg_total * 1000,
            'std_time_ms': std_total * 1000,
            'fps': fps,
            'num_frames': num_frames
        }
    
    def validate_detection_quality(self, model_name='yolov8n', sequence='MOT17-02-FRCNN', max_frames=100):
        """Validate detection quality against ground truth"""
        
        print(f"\n{'='*80}")
        print(f"VALIDATING DETECTION QUALITY: {model_name}")
        print(f"{'='*80}")
        
        # Load model
        model = YOLO(f"{model_name}.pt")
        
        # Load ground truth
        gt_path = Path(f"data/MOT17/train/{sequence}/gt/gt.txt")
        if not gt_path.exists():
            print(f"❌ Ground truth not found at {gt_path}")
            return None
        
        # Load ground truth annotations
        gt_data = pd.read_csv(gt_path, header=None, 
                              names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis'])
        
        # Filter for persons only (class 1 in MOT)
        gt_data = gt_data[gt_data['class'] == 1]
        
        # Load images
        img_path = Path(f"data/MOT17/train/{sequence}/img1")
        images = sorted(img_path.glob("*.jpg"))[:max_frames]
        
        print(f"Processing {len(images)} frames from {sequence}...")
        
        # Detection statistics
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        total_gt = 0
        total_detections = 0
        
        iou_threshold = 0.5  # Standard for object detection
        
        for frame_idx, img_file in enumerate(images):
            frame_num = frame_idx + 1
            
            # Load image
            img = cv2.imread(str(img_file))
            
            # Get ground truth for this frame
            frame_gt = gt_data[gt_data['frame'] == frame_num]
            gt_boxes = frame_gt[['x', 'y', 'w', 'h']].values
            
            # Run detection
            results = model(img, conf=0.25, verbose=False)
            
            # Extract person detections
            detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    if int(box.cls) == 0:  # Person class in COCO
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append([x1, y1, x2-x1, y2-y1])
            
            detections = np.array(detections) if detections else np.empty((0, 4))
            
            # Match detections to ground truth
            matched_gt = set()
            matched_det = set()
            
            for det_idx, det in enumerate(detections):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gt_boxes):
                    iou = self.calculate_iou(det, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou > iou_threshold and best_gt_idx not in matched_gt:
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
                    matched_det.add(det_idx)
                else:
                    false_positives += 1
            
            false_negatives += len(gt_boxes) - len(matched_gt)
            total_gt += len(gt_boxes)
            total_detections += len(detections)
            
            if frame_idx % 20 == 0:
                print(f"  Frame {frame_num}: {len(detections)} detections, {len(gt_boxes)} GT")
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n📊 DETECTION QUALITY METRICS:")
        print(f"  True Positives:  {true_positives}")
        print(f"  False Positives: {false_positives}")
        print(f"  False Negatives: {false_negatives}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1 Score:  {f1_score:.3f}")
        print(f"  Avg Detections/Frame: {total_detections/len(images):.1f}")
        print(f"  Avg GT/Frame: {total_gt/len(images):.1f}")
        
        return {
            'model': model_name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': true_positives,
            'fp': false_positives,
            'fn': false_negatives,
            'avg_detections': total_detections/len(images),
            'avg_gt': total_gt/len(images)
        }
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes in [x, y, w, h] format"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to [x1, y1, x2, y2]
        box1_x2 = x1 + w1
        box1_y2 = y1 + h1
        box2_x2 = x2 + w2
        box2_y2 = y2 + h2
        
        # Calculate intersection
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)
        
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        
        intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def run_comprehensive_debug(self):
        """Run complete debugging suite"""
        
        models_to_test = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
        
        # Performance profiling
        print("\n" + "="*80)
        print("PHASE 1: PERFORMANCE PROFILING")
        print("="*80)
        
        performance_results = []
        for model in models_to_test:
            result = self.profile_inference_pipeline(model, num_frames=100)
            performance_results.append(result)
        
        # Detection quality validation
        print("\n" + "="*80)
        print("PHASE 2: DETECTION QUALITY VALIDATION")
        print("="*80)
        
        quality_results = []
        for model in models_to_test:
            result = self.validate_detection_quality(model, max_frames=50)
            if result:
                quality_results.append(result)
        
        # Create comparison table
        self.create_comparison_table(performance_results, quality_results)
        
        return performance_results, quality_results
    
    def create_comparison_table(self, perf_results, quality_results):
        """Create comprehensive comparison table"""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE COMPARISON")
        print("="*80)
        
        # Merge results
        perf_df = pd.DataFrame(perf_results)
        quality_df = pd.DataFrame(quality_results)
        
        merged = pd.merge(perf_df, quality_df, on='model')
        
        # Calculate efficiency score
        merged['efficiency_score'] = (merged['f1_score'] * merged['fps']) / 100
        
        print("\n📊 COMPLETE PERFORMANCE & QUALITY METRICS:")
        print("="*80)
        print("| Model   | FPS   | Time(ms) | Precision | Recall | F1    | Efficiency |")
        print("|---------|-------|----------|-----------|--------|-------|------------|")
        
        for _, row in merged.iterrows():
            print(f"| {row['model']:7s} | {row['fps']:5.1f} | {row['avg_time_ms']:8.1f} | "
                  f"{row['precision']:9.3f} | {row['recall']:6.3f} | {row['f1_score']:5.3f} | "
                  f"{row['efficiency_score']:10.3f} |")
        
        # Save results
        output_file = Path("results/baseline/debug_validation_results.csv")
        merged.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to {output_file}")
        
        # Key insights
        print("\n🔍 KEY INSIGHTS:")
        
        # FPS spread
        fps_spread = merged['fps'].max() / merged['fps'].min()
        print(f"  - FPS Spread: {fps_spread:.1f}x (should be 5-10x)")
        
        # Quality vs Speed tradeoff
        best_quality = merged.loc[merged['f1_score'].idxmax()]
        best_speed = merged.loc[merged['fps'].idxmax()]
        best_efficiency = merged.loc[merged['efficiency_score'].idxmax()]
        
        print(f"  - Best Quality: {best_quality['model']} (F1: {best_quality['f1_score']:.3f})")
        print(f"  - Best Speed: {best_speed['model']} (FPS: {best_speed['fps']:.1f})")
        print(f"  - Best Efficiency: {best_efficiency['model']} (Score: {best_efficiency['efficiency_score']:.3f})")
        
        return merged


def main():
    """Run debugging and validation"""
    debugger = PerformanceDebugger()
    perf_results, quality_results = debugger.run_comprehensive_debug()
    
    print("\n" + "="*80)
    print("DEBUGGING COMPLETE")
    print("="*80)
    print("\nNext steps based on findings:")
    print("1. If FPS spread < 2x: Investigate CPU bottleneck or I/O issues")
    print("2. If precision < 0.5: Need to adjust confidence thresholds")
    print("3. If recall < 0.3: Models missing too many objects")
    print("4. Compare efficiency scores to select optimal models for adaptive strategy")

if __name__ == "__main__":
    main()