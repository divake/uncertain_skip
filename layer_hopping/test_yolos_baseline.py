#!/usr/bin/env python3
"""
Test pretrained YOLOS on MOT17 dataset with extensive logging
Baseline evaluation before any fine-tuning
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1

import torch
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

class YOLOSBaselineTester:
    def __init__(self, model_size='small', device='cuda'):
        """Initialize YOLOS model and processor"""
        
        print("=" * 60)
        print(f"YOLOS Baseline Testing on MOT17")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        self.device = device
        self.model_size = model_size
        
        # Create results directory
        self.results_dir = Path('results/baseline')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.log_file = self.results_dir / f'baseline_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        self.metrics_file = self.results_dir / f'baseline_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        print(f"\n📝 Logging to: {self.log_file}")
        print(f"📊 Metrics will be saved to: {self.metrics_file}")
        
        # Load model and processor
        print(f"\n🔄 Loading YOLOS-{model_size} model...")
        model_name = f"hustvl/yolos-{model_size}"
        
        start_time = time.time()
        self.processor = YolosImageProcessor.from_pretrained(model_name)
        self.model = YolosForObjectDetection.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n📈 Model Statistics:")
        print(f"  Total parameters: {total_params/1e6:.2f}M")
        print(f"  Trainable parameters: {trainable_params/1e6:.2f}M")
        print(f"  Device: {device}")
        
        # Log initial info
        self.log("YOLOS Baseline Testing Started")
        self.log(f"Model: {model_name}")
        self.log(f"Total params: {total_params/1e6:.2f}M")
        self.log(f"Device: {device}")
        
        # Initialize metrics
        self.all_metrics = {
            'per_sequence': {},
            'per_frame': defaultdict(list),
            'timing': defaultdict(list),
            'model_info': {
                'name': model_name,
                'total_params': total_params,
                'trainable_params': trainable_params
            }
        }
    
    def log(self, message):
        """Log message to file and console"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def process_single_image(self, image_path):
        """Process single image and return detections with timing"""
        
        # Load and preprocess image
        start_preprocess = time.time()
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        preprocess_time = time.time() - start_preprocess
        
        # Model inference
        start_inference = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
        inference_time = time.time() - start_inference
        
        # Post-processing
        start_postprocess = time.time()
        # Target sizes (height, width) for post-processing
        target_sizes = torch.tensor([image.size[::-1]])  # PIL uses (width, height)
        results = self.processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes.to(self.device), 
            threshold=0.5  # Confidence threshold
        )[0]
        postprocess_time = time.time() - start_postprocess
        
        # Extract results
        boxes = results['boxes'].cpu().numpy()  # [x1, y1, x2, y2]
        scores = results['scores'].cpu().numpy()
        labels = results['labels'].cpu().numpy()
        
        # Filter for person class (COCO class 1)
        person_mask = labels == 1
        person_boxes = boxes[person_mask]
        person_scores = scores[person_mask]
        
        return {
            'boxes': person_boxes,
            'scores': person_scores,
            'n_detections': len(person_boxes),
            'timing': {
                'preprocess': preprocess_time,
                'inference': inference_time,
                'postprocess': postprocess_time,
                'total': preprocess_time + inference_time + postprocess_time
            }
        }
    
    def evaluate_sequence(self, sequence_name, max_frames=None):
        """Evaluate model on a single MOT17 sequence"""
        
        self.log(f"\n{'='*40}")
        self.log(f"Evaluating sequence: {sequence_name}")
        self.log(f"{'='*40}")
        
        # Paths
        seq_path = Path(f"../data/MOT17/train/{sequence_name}-FRCNN")
        img_dir = seq_path / "img1"
        gt_file = seq_path / "gt" / "gt.txt"
        
        # Load ground truth
        gt_data = pd.read_csv(gt_file, header=None)
        gt_data.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'visibility']
        
        # Get image list
        img_files = sorted(img_dir.glob("*.jpg"))
        if max_frames:
            img_files = img_files[:max_frames]
        
        self.log(f"Processing {len(img_files)} frames...")
        
        # Process each frame
        sequence_results = []
        frame_metrics = []
        
        for frame_idx, img_path in enumerate(tqdm(img_files, desc=sequence_name)):
            frame_num = frame_idx + 1
            
            # Get detections
            det_result = self.process_single_image(img_path)
            
            # Get ground truth for this frame
            frame_gt = gt_data[gt_data['frame'] == frame_num]
            gt_boxes = frame_gt[['x', 'y', 'w', 'h']].values
            # Convert from [x, y, w, h] to [x1, y1, x2, y2]
            gt_boxes_xyxy = np.column_stack([
                gt_boxes[:, 0],
                gt_boxes[:, 1],
                gt_boxes[:, 0] + gt_boxes[:, 2],
                gt_boxes[:, 1] + gt_boxes[:, 3]
            ])
            
            # Calculate frame-level metrics
            n_gt = len(gt_boxes_xyxy)
            n_det = det_result['n_detections']
            
            # Simple IoU matching for TP/FP/FN
            tp, fp, fn = self.calculate_tp_fp_fn(det_result['boxes'], gt_boxes_xyxy)
            
            frame_metric = {
                'frame': frame_num,
                'n_gt': n_gt,
                'n_det': n_det,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'timing': det_result['timing']
            }
            
            frame_metrics.append(frame_metric)
            
            # Store for sequence-level metrics
            sequence_results.append({
                'frame': frame_num,
                'detections': det_result,
                'ground_truth': gt_boxes_xyxy
            })
            
            # Log every 50 frames
            if frame_num % 50 == 0:
                avg_time = np.mean([fm['timing']['total'] for fm in frame_metrics[-50:]])
                avg_prec = np.mean([fm['precision'] for fm in frame_metrics[-50:]])
                avg_rec = np.mean([fm['recall'] for fm in frame_metrics[-50:]])
                
                self.log(f"  Frame {frame_num}: "
                        f"Avg time={avg_time:.3f}s, "
                        f"Prec={avg_prec:.3f}, "
                        f"Rec={avg_rec:.3f}")
        
        # Calculate sequence-level metrics
        seq_metrics = self.calculate_sequence_metrics(frame_metrics)
        
        # Store in all_metrics
        self.all_metrics['per_sequence'][sequence_name] = seq_metrics
        self.all_metrics['per_frame'][sequence_name] = frame_metrics
        
        # Log summary
        self.log(f"\n📊 Sequence {sequence_name} Summary:")
        self.log(f"  Total frames: {len(frame_metrics)}")
        self.log(f"  Average FPS: {1.0/seq_metrics['avg_time_per_frame']:.2f}")
        self.log(f"  Average Precision: {seq_metrics['avg_precision']:.3f}")
        self.log(f"  Average Recall: {seq_metrics['avg_recall']:.3f}")
        self.log(f"  F1 Score: {seq_metrics['f1_score']:.3f}")
        
        return seq_metrics
    
    def calculate_tp_fp_fn(self, pred_boxes, gt_boxes, iou_threshold=0.5):
        """Calculate TP, FP, FN using IoU matching"""
        
        if len(pred_boxes) == 0 and len(gt_boxes) == 0:
            return 0, 0, 0
        if len(pred_boxes) == 0:
            return 0, 0, len(gt_boxes)
        if len(gt_boxes) == 0:
            return 0, len(pred_boxes), 0
        
        # Calculate IoU matrix
        ious = self.calculate_iou_matrix(pred_boxes, gt_boxes)
        
        # Greedy matching
        tp = 0
        matched_gt = set()
        matched_pred = set()
        
        # Sort by IoU in descending order
        iou_pairs = []
        for i in range(len(pred_boxes)):
            for j in range(len(gt_boxes)):
                if ious[i, j] >= iou_threshold:
                    iou_pairs.append((ious[i, j], i, j))
        
        iou_pairs.sort(reverse=True)
        
        for iou, pred_idx, gt_idx in iou_pairs:
            if pred_idx not in matched_pred and gt_idx not in matched_gt:
                tp += 1
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)
        
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        
        return tp, fp, fn
    
    def calculate_iou_matrix(self, boxes1, boxes2):
        """Calculate IoU matrix between two sets of boxes"""
        
        n1 = len(boxes1)
        n2 = len(boxes2)
        iou_matrix = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                iou_matrix[i, j] = self.calculate_iou(boxes1[i], boxes2[j])
        
        return iou_matrix
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes [x1, y1, x2, y2]"""
        
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_sequence_metrics(self, frame_metrics):
        """Calculate aggregate metrics for a sequence"""
        
        total_tp = sum(fm['tp'] for fm in frame_metrics)
        total_fp = sum(fm['fp'] for fm in frame_metrics)
        total_fn = sum(fm['fn'] for fm in frame_metrics)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Timing statistics
        all_timings = [fm['timing']['total'] for fm in frame_metrics]
        
        return {
            'total_frames': len(frame_metrics),
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'avg_precision': precision,
            'avg_recall': recall,
            'f1_score': f1,
            'avg_time_per_frame': np.mean(all_timings),
            'std_time_per_frame': np.std(all_timings),
            'min_time_per_frame': np.min(all_timings),
            'max_time_per_frame': np.max(all_timings),
            'fps': 1.0 / np.mean(all_timings)
        }
    
    def run_full_evaluation(self, sequences=None, max_frames_per_seq=None):
        """Run evaluation on multiple sequences"""
        
        if sequences is None:
            # Use validation sequences
            sequences = ['MOT17-11', 'MOT17-13']
        
        self.log(f"\n{'='*60}")
        self.log(f"Starting Full Evaluation")
        self.log(f"Sequences: {sequences}")
        self.log(f"Max frames per sequence: {max_frames_per_seq or 'All'}")
        self.log(f"{'='*60}")
        
        overall_start = time.time()
        
        for seq in sequences:
            seq_metrics = self.evaluate_sequence(seq, max_frames_per_seq)
        
        total_time = time.time() - overall_start
        
        # Calculate overall metrics
        self.calculate_overall_metrics()
        
        self.log(f"\n{'='*60}")
        self.log(f"Evaluation Complete")
        self.log(f"Total time: {total_time:.2f} seconds")
        self.log(f"{'='*60}")
        
        # Save all metrics
        self.save_metrics()
        
        return self.all_metrics
    
    def calculate_overall_metrics(self):
        """Calculate metrics across all sequences"""
        
        if not self.all_metrics['per_sequence']:
            return
        
        # Aggregate across sequences
        total_tp = sum(s['total_tp'] for s in self.all_metrics['per_sequence'].values())
        total_fp = sum(s['total_fp'] for s in self.all_metrics['per_sequence'].values())
        total_fn = sum(s['total_fn'] for s in self.all_metrics['per_sequence'].values())
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        # Average timing
        all_fps = [s['fps'] for s in self.all_metrics['per_sequence'].values()]
        
        self.all_metrics['overall'] = {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'avg_fps': np.mean(all_fps),
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn
        }
        
        self.log(f"\n📊 Overall Metrics:")
        self.log(f"  Precision: {overall_precision:.3f}")
        self.log(f"  Recall: {overall_recall:.3f}")
        self.log(f"  F1 Score: {overall_f1:.3f}")
        self.log(f"  Average FPS: {np.mean(all_fps):.2f}")
    
    def save_metrics(self):
        """Save all metrics to JSON file"""
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Recursively convert all numpy types
        import json
        metrics_json = json.loads(json.dumps(self.all_metrics, default=convert_numpy))
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        self.log(f"\n✅ Metrics saved to: {self.metrics_file}")
        
        # Also save frame-level metrics as CSV for analysis
        for seq_name, frame_metrics in self.all_metrics['per_frame'].items():
            df = pd.DataFrame(frame_metrics)
            csv_file = self.results_dir / f'frame_metrics_{seq_name}.csv'
            df.to_csv(csv_file, index=False)
            self.log(f"  Frame metrics saved: {csv_file}")


if __name__ == "__main__":
    # Test YOLOS baseline on validation sequences
    tester = YOLOSBaselineTester(model_size='small', device='cuda')
    
    # Run on validation sequences with limited frames for quick test
    # Use None for max_frames_per_seq to process all frames
    metrics = tester.run_full_evaluation(
        sequences=['MOT17-11', 'MOT17-13'],
        max_frames_per_seq=100  # Start with 100 frames for quick test
    )
    
    print("\n✅ Baseline testing complete!")
    print(f"Check results in: layer_hopping/results/baseline/")