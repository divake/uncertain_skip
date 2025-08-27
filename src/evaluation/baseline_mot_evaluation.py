#!/usr/bin/env python3
"""
Comprehensive YOLOv8 Baseline Evaluation on MOT17 Dataset
Tests YOLOv8 models (nano, small, medium, large, xlarge) on MOT17 sequences
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ultralytics import YOLO
import motmetrics as mm
import cv2

from src.tracking.sort import Sort
from src.utils.mot_utils import (
    load_mot_gt, save_mot_results, convert_bbox_format,
    calculate_iou, format_detection_for_tracking
)
from src.visualization.plot_results import create_comparison_plots


class YOLOBaselineEvaluator:
    def __init__(self, data_dir="data/MOT17", output_dir="results/baseline"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # YOLO model variants to test
        self.model_variants = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
        
        # MOT17 sequences to evaluate (FRCNN variants only)
        self.test_sequences = [
            "MOT17-02-FRCNN",  # 600 frames, indoor mall
            "MOT17-04-FRCNN",  # 1050 frames, street view
            "MOT17-05-FRCNN"   # 837 frames, night scene
        ]
        
        # Detection parameters (consistent across all models)
        self.conf_threshold = 0.25
        self.nms_threshold = 0.45
        self.input_size = 640
        
        # Tracking parameters
        self.max_age = 5  # Delete track if no detection for 5 frames
        self.min_hits = 3  # Minimum hits to start tracking
        self.iou_threshold = 0.3  # IoU threshold for association
        
        # Device configuration - use less loaded GPU if available
        if torch.cuda.is_available():
            # Check for multiple GPUs and use the less loaded one
            if torch.cuda.device_count() > 1:
                self.device = torch.device('cuda:1')
            else:
                self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        
        # Results storage
        self.all_results = []
        
    def download_and_load_model(self, model_name):
        """Download and load YOLOv8 model"""
        print(f"\nLoading {model_name} model...")
        model_path = f"models/yolo_weights/{model_name}.pt"
        
        # Create directory if not exists
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Load model (will download if not present)
        model = YOLO(f"{model_name}.pt")
        model.to(self.device)
        
        # Get model info
        model_info = {
            'name': model_name,
            'parameters': sum(p.numel() for p in model.model.parameters()),
            'size_mb': sum(p.numel() * p.element_size() for p in model.model.parameters()) / (1024**2)
        }
        
        return model, model_info
    
    def process_sequence(self, model, model_name, sequence_name):
        """Process a single MOT17 sequence with given model"""
        print(f"\nProcessing {sequence_name} with {model_name}")
        
        # Paths
        seq_path = self.data_dir / "train" / sequence_name
        img_path = seq_path / "img1"
        gt_path = seq_path / "gt" / "gt.txt"
        det_path = seq_path / "det" / "det.txt"
        
        # Check if sequence exists
        if not seq_path.exists():
            print(f"Warning: Sequence {sequence_name} not found at {seq_path}")
            return None
        
        # Load ground truth
        gt_data = load_mot_gt(gt_path)
        
        # Get image list
        images = sorted(img_path.glob("*.jpg"))
        if not images:
            print(f"Warning: No images found in {img_path}")
            return None
        
        # Initialize tracker
        tracker = Sort(max_age=self.max_age, min_hits=self.min_hits, iou_threshold=self.iou_threshold)
        
        # Storage for results
        detections_all = []
        tracks_all = []
        timing_info = []
        memory_info = []
        
        # Process each frame
        for frame_idx, img_file in enumerate(tqdm(images, desc=f"Processing {sequence_name}")):
            frame_num = frame_idx + 1
            
            # Read image
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            # Measure GPU memory before inference
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated() / (1024**3)  # GB
            else:
                memory_before = 0
            
            # Run detection with timing
            start_time = time.time()
            
            # YOLOv8 inference
            results = model(img, conf=self.conf_threshold, iou=self.nms_threshold, 
                          imgsz=self.input_size, verbose=False)
            
            # Extract detections (class 0 is person in COCO)
            detections = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        if int(box.cls) == 0:  # Person class
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf)
                            
                            # Convert to MOT format (x, y, w, h)
                            bbox = [x1, y1, x2-x1, y2-y1, conf]
                            detections.append(bbox)
            
            # Convert to numpy array for tracker
            if detections:
                detections_np = np.array(detections)
            else:
                detections_np = np.empty((0, 5))
            
            # Update tracker
            tracks = tracker.update(detections_np)
            
            # Measure time and memory
            inference_time = time.time() - start_time
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated() / (1024**3)
                memory_used = memory_after - memory_before
            else:
                memory_used = 0
            
            timing_info.append(inference_time)
            memory_info.append(memory_used)
            
            # Store detections and tracks
            for det in detections:
                det_entry = [frame_num] + det[:4] + [det[4], -1, -1, -1]
                detections_all.append(det_entry)
            
            for track in tracks:
                track_id = int(track[4])
                bbox = track[:4]
                track_entry = [frame_num, track_id] + bbox.tolist() + [1.0, -1, -1, -1]
                tracks_all.append(track_entry)
        
        # Calculate metrics
        print(f"Calculating MOT metrics for {sequence_name}...")
        metrics = self.calculate_mot_metrics(tracks_all, gt_data, len(images))
        
        # Add performance metrics
        metrics['avg_inference_time'] = np.mean(timing_info)
        metrics['fps'] = 1.0 / metrics['avg_inference_time']
        metrics['max_memory_gb'] = np.max(memory_info) if memory_info else 0
        metrics['total_frames'] = len(images)
        metrics['total_detections'] = len(detections_all)
        metrics['avg_detections_per_frame'] = len(detections_all) / len(images)
        
        # Save results
        self.save_sequence_results(model_name, sequence_name, detections_all, tracks_all, metrics)
        
        return metrics
    
    def _compute_iou_distance_matrix(self, gt_bboxes, track_bboxes):
        """Compute IoU distance matrix between ground truth and tracked bounding boxes"""
        # gt_bboxes and track_bboxes are in format [x, y, w, h]
        iou_matrix = np.zeros((len(gt_bboxes), len(track_bboxes)))
        
        for i, gt_bbox in enumerate(gt_bboxes):
            for j, track_bbox in enumerate(track_bboxes):
                # Convert to x1, y1, x2, y2 format
                gt_x1, gt_y1 = gt_bbox[0], gt_bbox[1]
                gt_x2, gt_y2 = gt_x1 + gt_bbox[2], gt_y1 + gt_bbox[3]
                
                track_x1, track_y1 = track_bbox[0], track_bbox[1]
                track_x2, track_y2 = track_x1 + track_bbox[2], track_y1 + track_bbox[3]
                
                # Calculate intersection
                int_x1 = max(gt_x1, track_x1)
                int_y1 = max(gt_y1, track_y1)
                int_x2 = min(gt_x2, track_x2)
                int_y2 = min(gt_y2, track_y2)
                
                if int_x2 > int_x1 and int_y2 > int_y1:
                    intersection = (int_x2 - int_x1) * (int_y2 - int_y1)
                else:
                    intersection = 0
                
                # Calculate union
                gt_area = gt_bbox[2] * gt_bbox[3]
                track_area = track_bbox[2] * track_bbox[3]
                union = gt_area + track_area - intersection
                
                # Calculate IoU
                if union > 0:
                    iou = intersection / union
                else:
                    iou = 0
                
                # Convert IoU to distance (1 - IoU)
                iou_matrix[i, j] = 1 - iou
        
        return iou_matrix
    
    def calculate_mot_metrics(self, tracks, ground_truth, num_frames):
        """Calculate MOT metrics using py-motmetrics"""
        
        # Create accumulator
        acc = mm.MOTAccumulator(auto_id=True)
        
        # Process each frame
        for frame_num in range(1, num_frames + 1):
            # Get tracks for this frame
            frame_tracks = [t for t in tracks if t[0] == frame_num]
            frame_gt = [g for g in ground_truth if g[0] == frame_num]
            
            # Extract IDs and bboxes
            track_ids = [int(t[1]) for t in frame_tracks]
            track_bboxes = np.array([t[2:6] for t in frame_tracks]) if frame_tracks else np.empty((0, 4))
            
            gt_ids = [int(g[1]) for g in frame_gt]
            gt_bboxes = np.array([g[2:6] for g in frame_gt]) if frame_gt else np.empty((0, 4))
            
            # Calculate distances (IoU-based)
            if len(track_bboxes) > 0 and len(gt_bboxes) > 0:
                # Fix for numpy 2.0 compatibility
                gt_bboxes_array = np.asarray(gt_bboxes, dtype=np.float64)
                track_bboxes_array = np.asarray(track_bboxes, dtype=np.float64)
                # Calculate IoU manually to avoid motmetrics numpy issue
                distances = self._compute_iou_distance_matrix(gt_bboxes_array, track_bboxes_array)
            else:
                distances = np.empty((len(gt_bboxes), len(track_bboxes)))
            
            # Update accumulator
            acc.update(gt_ids, track_ids, distances)
        
        # Compute metrics
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=['mota', 'motp', 'num_switches', 'num_false_positives',
                                          'num_misses', 'num_detections', 'num_objects',
                                          'num_predictions', 'mostly_tracked', 'mostly_lost',
                                          'partially_tracked', 'num_fragmentations', 'idf1'],
                            name='acc')
        
        # Extract metrics
        metrics = {
            'mota': summary['mota'].values[0] * 100,  # Convert to percentage
            'motp': summary['motp'].values[0],
            'idf1': summary['idf1'].values[0] * 100,
            'mt': summary['mostly_tracked'].values[0],
            'ml': summary['mostly_lost'].values[0],
            'fp': summary['num_false_positives'].values[0],
            'fn': summary['num_misses'].values[0],
            'id_switches': summary['num_switches'].values[0],
            'fragmentations': summary['num_fragmentations'].values[0],
            'num_objects': summary['num_objects'].values[0],
            'num_predictions': summary['num_predictions'].values[0]
        }
        
        # Calculate percentages for MT and ML
        if metrics['num_objects'] > 0:
            metrics['mt_percentage'] = (metrics['mt'] / metrics['num_objects']) * 100
            metrics['ml_percentage'] = (metrics['ml'] / metrics['num_objects']) * 100
        else:
            metrics['mt_percentage'] = 0
            metrics['ml_percentage'] = 0
        
        return metrics
    
    def save_sequence_results(self, model_name, sequence_name, detections, tracks, metrics):
        """Save results for a single sequence"""
        
        # Create output directory for this model-sequence combination
        output_path = self.output_dir / f"{model_name}_{sequence_name}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detections in MOT format
        det_file = output_path / "det.txt"
        with open(det_file, 'w') as f:
            for det in detections:
                line = ','.join(map(str, det))
                f.write(line + '\n')
        
        # Save tracks in MOT format
        track_file = output_path / "tracks.txt"
        with open(track_file, 'w') as f:
            for track in tracks:
                line = ','.join(map(str, track))
                f.write(line + '\n')
        
        # Save metrics (convert numpy types to native Python types for JSON)
        metrics_file = output_path / "metrics.json"
        import json
        
        # Convert numpy types to native Python types
        metrics_json = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.int64)):
                metrics_json[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                metrics_json[key] = float(value)
            else:
                metrics_json[key] = value
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def run_full_evaluation(self):
        """Run evaluation for all models on all sequences"""
        
        print("\n" + "="*80)
        print("Starting YOLOv8 Baseline Evaluation on MOT17")
        print("="*80)
        
        # Results DataFrame
        results_data = []
        
        for model_name in self.model_variants:
            # Clear GPU cache before loading new model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load model
            model, model_info = self.download_and_load_model(model_name)
            
            for sequence_name in self.test_sequences:
                # Process sequence
                metrics = self.process_sequence(model, model_name, sequence_name)
                
                if metrics:
                    # Combine all results
                    result_entry = {
                        'model': model_name,
                        'sequence': sequence_name,
                        'model_params_M': model_info['parameters'] / 1e6,
                        'model_size_mb': model_info['size_mb'],
                        **metrics
                    }
                    
                    results_data.append(result_entry)
                    self.all_results.append(result_entry)
                    
                    # Print summary for this run
                    print(f"\nResults for {model_name} on {sequence_name}:")
                    print(f"  MOTA: {metrics['mota']:.2f}%")
                    print(f"  MOTP: {metrics['motp']:.4f}")
                    print(f"  IDF1: {metrics['idf1']:.2f}%")
                    print(f"  FPS: {metrics['fps']:.2f}")
                    print(f"  Memory: {metrics['max_memory_gb']:.2f} GB")
            
            # Clear model from memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Save all results to CSV
        df_results = pd.DataFrame(results_data)
        
        # Save detailed results
        detailed_csv = self.output_dir / f"detailed_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        df_results.to_csv(detailed_csv, index=False)
        print(f"\nDetailed results saved to {detailed_csv}")
        
        # Create summary table
        self.create_summary_table(df_results)
        
        # Create visualization plots
        self.create_visualizations(df_results)
        
        return df_results
    
    def create_summary_table(self, df_results):
        """Create summary table comparing all models"""
        
        # Group by model and calculate mean metrics
        summary = df_results.groupby('model').agg({
            'model_params_M': 'first',
            'model_size_mb': 'first',
            'mota': 'mean',
            'motp': 'mean',
            'idf1': 'mean',
            'mt_percentage': 'mean',
            'ml_percentage': 'mean',
            'fp': 'sum',
            'fn': 'sum',
            'id_switches': 'sum',
            'fps': 'mean',
            'max_memory_gb': 'max',
            'avg_detections_per_frame': 'mean'
        }).round(2)
        
        # Save summary
        summary_csv = self.output_dir / "summary_all_models.csv"
        summary.to_csv(summary_csv)
        print(f"\nSummary table saved to {summary_csv}")
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY: Average Performance Across All Sequences")
        print("="*80)
        print(summary.to_string())
        
        return summary
    
    def create_visualizations(self, df_results):
        """Create visualization plots"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('YOLOv8 Models Performance on MOT17', fontsize=16, fontweight='bold')
        
        # 1. MOTA vs FPS (Speed-Accuracy Tradeoff)
        ax = axes[0, 0]
        for model in self.model_variants:
            model_data = df_results[df_results['model'] == model]
            ax.scatter(model_data['fps'], model_data['mota'], label=model, s=100)
        ax.set_xlabel('FPS (Frames per Second)')
        ax.set_ylabel('MOTA (%)')
        ax.set_title('Speed vs Accuracy Tradeoff')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Model Size vs Performance
        ax = axes[0, 1]
        summary_by_model = df_results.groupby('model').agg({
            'model_params_M': 'first',
            'mota': 'mean',
            'idf1': 'mean'
        })
        x = summary_by_model['model_params_M']
        ax.plot(x, summary_by_model['mota'], 'o-', label='MOTA', markersize=8)
        ax.plot(x, summary_by_model['idf1'], 's-', label='IDF1', markersize=8)
        ax.set_xlabel('Model Parameters (Millions)')
        ax.set_ylabel('Score (%)')
        ax.set_title('Model Size vs Tracking Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Memory Usage by Model
        ax = axes[0, 2]
        memory_data = df_results.groupby('model')['max_memory_gb'].max()
        bars = ax.bar(range(len(memory_data)), memory_data.values)
        ax.set_xticks(range(len(memory_data)))
        ax.set_xticklabels(memory_data.index, rotation=45)
        ax.set_ylabel('GPU Memory (GB)')
        ax.set_title('Maximum GPU Memory Usage')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Color bars based on memory usage
        for i, (bar, val) in enumerate(zip(bars, memory_data.values)):
            if val < 1:
                bar.set_color('green')
            elif val < 2:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
        
        # 4. Performance by Sequence
        ax = axes[1, 0]
        pivot_mota = df_results.pivot(index='sequence', columns='model', values='mota')
        pivot_mota.plot(kind='bar', ax=ax)
        ax.set_ylabel('MOTA (%)')
        ax.set_title('MOTA Performance by Sequence')
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 5. ID Switches and Fragmentations
        ax = axes[1, 1]
        error_metrics = df_results.groupby('model').agg({
            'id_switches': 'sum',
            'fragmentations': 'sum'
        })
        x_pos = np.arange(len(error_metrics))
        width = 0.35
        ax.bar(x_pos - width/2, error_metrics['id_switches'], width, label='ID Switches')
        ax.bar(x_pos + width/2, error_metrics['fragmentations'], width, label='Fragmentations')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(error_metrics.index, rotation=45)
        ax.set_ylabel('Count')
        ax.set_title('Tracking Errors by Model')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 6. Comprehensive Performance Radar Chart
        ax = axes[1, 2]
        from math import pi
        
        # Select metrics for radar chart
        metrics_for_radar = ['mota', 'idf1', 'mt_percentage', 'fps']
        
        # Normalize metrics to 0-100 scale
        normalized_data = {}
        for model in self.model_variants[:3]:  # Show only first 3 models for clarity
            model_data = df_results[df_results['model'] == model].mean()
            normalized_data[model] = [
                model_data['mota'],
                model_data['idf1'],
                model_data['mt_percentage'],
                min(100, model_data['fps'] / 1.5)  # Normalize FPS (assume 150 FPS = 100%)
            ]
        
        # Number of variables
        categories = ['MOTA', 'IDF1', 'MT%', 'FPS (norm)']
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        # Initialize radar chart
        ax = plt.subplot(2, 3, 6, projection='polar')
        
        # Plot data for each model
        for model, values in normalized_data.items():
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title('Comprehensive Performance (n, s, m models)', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plot_file = self.output_dir / "performance_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to {plot_file}")
        plt.close()
    
    def generate_report(self, df_results):
        """Generate a comprehensive evaluation report"""
        
        report_file = self.output_dir / "evaluation_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# YOLOv8 Baseline Evaluation Report on MOT17\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Find best models for different criteria
            best_accuracy = df_results.groupby('model')['mota'].mean().idxmax()
            best_speed = df_results.groupby('model')['fps'].mean().idxmax()
            
            f.write(f"- **Best Accuracy**: {best_accuracy} (MOTA: {df_results[df_results['model']==best_accuracy]['mota'].mean():.2f}%)\n")
            f.write(f"- **Best Speed**: {best_speed} (FPS: {df_results[df_results['model']==best_speed]['fps'].mean():.2f})\n\n")
            
            f.write("## Detailed Results by Model\n\n")
            
            for model in self.model_variants:
                model_data = df_results[df_results['model'] == model]
                if not model_data.empty:
                    f.write(f"### {model.upper()}\n\n")
                    f.write(f"- **Parameters**: {model_data['model_params_M'].iloc[0]:.2f}M\n")
                    f.write(f"- **Model Size**: {model_data['model_size_mb'].iloc[0]:.2f} MB\n")
                    f.write(f"- **Average MOTA**: {model_data['mota'].mean():.2f}%\n")
                    f.write(f"- **Average IDF1**: {model_data['idf1'].mean():.2f}%\n")
                    f.write(f"- **Average FPS**: {model_data['fps'].mean():.2f}\n")
                    f.write(f"- **Max GPU Memory**: {model_data['max_memory_gb'].max():.2f} GB\n\n")
            
            f.write("## Sequence-Specific Performance\n\n")
            
            for seq in self.test_sequences:
                seq_data = df_results[df_results['sequence'] == seq]
                if not seq_data.empty:
                    f.write(f"### {seq}\n\n")
                    f.write(seq_data[['model', 'mota', 'idf1', 'fps']].to_markdown(index=False))
                    f.write("\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("Based on the evaluation results:\n\n")
            f.write("1. **For Real-time Applications**: Use YOLOv8n or YOLOv8s\n")
            f.write("2. **For High Accuracy**: Use YOLOv8l or YOLOv8x\n")
            f.write("3. **Balanced Performance**: YOLOv8m offers good compromise\n")
            f.write("4. **Adaptive Strategy**: Switch between models based on scene complexity\n")
        
        print(f"\nEvaluation report saved to {report_file}")


def main():
    """Main execution function"""
    
    # Create evaluator
    evaluator = YOLOBaselineEvaluator()
    
    # Run full evaluation
    results_df = evaluator.run_full_evaluation()
    
    # Generate report
    evaluator.generate_report(results_df)
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    print(f"Results saved in: {evaluator.output_dir}")
    print("\nNext steps:")
    print("1. Review the evaluation report and visualizations")
    print("2. Analyze the speed-accuracy tradeoffs")
    print("3. Design adaptive selection strategy based on these baselines")
    

if __name__ == "__main__":
    main()