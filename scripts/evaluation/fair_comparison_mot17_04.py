#!/usr/bin/env python3
"""
Fair comparison: Test ALL models (fixed + adaptive) on the SAME object in MOT17-04
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm import tqdm

class FairComparison:
    """Run fair comparison on MOT17-04 with same object for all models"""
    
    def __init__(self):
        self.sequence = "MOT17-04-FRCNN"
        self.img_path = Path(f"data/MOT17/train/{self.sequence}/img1")
        self.max_frames = 1050
        self.iou_threshold = 0.3
        self.confidence_threshold = 0.25
        
        # Models to test
        self.model_names = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
        
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
    
    def select_initial_object(self, strategy='high_confidence'):
        """Select the same initial object for all models"""
        print("Selecting initial object...")
        
        # Load first frame
        first_frame = cv2.imread(str(sorted(self.img_path.glob("*.jpg"))[0]))
        
        # Use YOLOv8m to select initial object (middle ground model)
        model = YOLO('models/yolov8m.pt')
        model.to('cuda')
        
        results = model(first_frame, conf=0.25, device='cuda', verbose=False)
        
        if results[0].boxes is None:
            return None
            
        candidates = []
        for box in results[0].boxes:
            if int(box.cls) == 0:  # Person class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                area = (x2 - x1) * (y2 - y1)
                candidates.append({
                    'bbox': [x1, y1, x2-x1, y2-y1],
                    'confidence': conf,
                    'area': area
                })
        
        if not candidates:
            return None
            
        if strategy == 'high_confidence':
            selected = max(candidates, key=lambda x: x['confidence'])
        elif strategy == 'medium_confidence':
            target_conf = 0.6
            selected = min(candidates, key=lambda x: abs(x['confidence'] - target_conf))
        else:
            selected = candidates[0]
            
        print(f"Selected object with confidence {selected['confidence']:.3f}")
        return selected['bbox']
    
    def track_with_fixed_model(self, model_name, initial_bbox):
        """Track object with a fixed model"""
        print(f"\nTesting {model_name}...")
        
        # Load model
        model = YOLO(f'models/{model_name}.pt')
        model.to('cuda')
        
        # Load images
        images = sorted(self.img_path.glob("*.jpg"))[:self.max_frames]
        
        # Tracking variables
        current_bbox = initial_bbox.copy()
        results = []
        lost_frames = 0
        max_lost = 5
        confidences = []
        inference_times = []
        
        for frame_idx, img_file in enumerate(tqdm(images, desc=model_name)):
            frame = cv2.imread(str(img_file))
            
            # Time inference
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            # Run detection
            detections = model(frame, conf=self.confidence_threshold, 
                             device='cuda', verbose=False)
            
            torch.cuda.synchronize()
            inference_time = (time.perf_counter() - start) * 1000  # ms
            inference_times.append(inference_time)
            
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
                confidences.append(confidence)
                status = 'tracked'
                lost_frames = 0
            else:
                confidence = 0.0
                lost_frames += 1
                status = 'lost' if lost_frames > max_lost else 'searching'
            
            results.append({
                'frame': frame_idx,
                'confidence': confidence,
                'status': status,
                'inference_ms': inference_time
            })
            
            # Stop if object lost
            if lost_frames > max_lost:
                print(f"  Object lost at frame {frame_idx}")
                break
        
        # Calculate metrics
        tracked = [r for r in results if r['status'] == 'tracked']
        tracking_rate = len(tracked) / len(results) if results else 0
        avg_confidence = np.mean(confidences) if confidences else 0
        avg_inference = np.mean(inference_times) if inference_times else 0
        
        model_params = sum(p.numel() for p in model.model.parameters()) / 1e6
        
        return {
            'model': model_name,
            'parameters_M': model_params,
            'total_frames': len(results),
            'tracked_frames': len(tracked),
            'tracking_rate': tracking_rate,
            'avg_confidence': avg_confidence,
            'avg_inference_ms': avg_inference,
            'fps': 1000 / avg_inference if avg_inference > 0 else 0,
            'lost_at_frame': len(results),
            'frame_results': results[:100]  # Save first 100 frames
        }
    
    def track_with_adaptive_model(self, initial_bbox):
        """Track with adaptive model switching"""
        print(f"\nTesting Adaptive Model...")
        
        # Import adaptive tracker
        from enhanced_adaptive_tracker import EnhancedAdaptiveTracker
        
        # Create tracker
        tracker = EnhancedAdaptiveTracker()
        
        # Load images
        images = sorted(self.img_path.glob("*.jpg"))[:self.max_frames]
        
        # Initialize tracking state
        from enhanced_adaptive_tracker import TrackingState
        state = TrackingState(
            frame_idx=0,
            bbox=np.array(initial_bbox),
            confidence=0.8,  # Initial confidence
            model_used='yolov8n',
            uncertainty=0.0
        )
        
        results = []
        lost_frames = 0
        confidences = []
        model_switches = []
        model_usage = {}
        inference_times = []
        
        for frame_idx, img_file in enumerate(tqdm(images, desc="Adaptive")):
            frame = cv2.imread(str(img_file))
            
            # Update state
            state.frame_idx = frame_idx
            state.frames_since_switch += 1
            if state.switch_cooldown > 0:
                state.switch_cooldown -= 1
            
            # Get adaptive model
            new_model = tracker.select_adaptive_model_bidirectional(state)
            
            # Check for model switch
            if new_model != state.model_used:
                model_switches.append({
                    'frame': frame_idx,
                    'from': state.model_used,
                    'to': new_model
                })
                state.model_used = new_model
                state.frames_since_switch = 0
                state.switch_cooldown = tracker.cooldown
            
            # Track model usage
            if state.model_used not in model_usage:
                model_usage[state.model_used] = 0
            model_usage[state.model_used] += 1
            
            # Time inference
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            # Track object
            new_bbox, confidence = tracker.track_object_in_frame(
                frame, state.bbox, state.model_used)
            
            torch.cuda.synchronize()
            inference_time = (time.perf_counter() - start) * 1000
            inference_times.append(inference_time)
            
            if new_bbox is not None:
                state.bbox = np.array(new_bbox)
                state.confidence = confidence
                state.confidence_history.append(confidence)
                confidences.append(confidence)
                state.uncertainty = tracker.calculate_uncertainty(state.confidence_history)
                lost_frames = 0
                status = 'tracked'
            else:
                lost_frames += 1
                confidence = 0.0
                status = 'lost' if lost_frames > tracker.max_lost_frames else 'searching'
            
            results.append({
                'frame': frame_idx,
                'model': state.model_used,
                'confidence': confidence,
                'status': status,
                'inference_ms': inference_time
            })
            
            # Stop if lost
            if lost_frames > tracker.max_lost_frames:
                print(f"  Object lost at frame {frame_idx}")
                break
        
        # Calculate metrics
        tracked = [r for r in results if r['status'] == 'tracked']
        tracking_rate = len(tracked) / len(results) if results else 0
        avg_confidence = np.mean(confidences) if confidences else 0
        avg_inference = np.mean(inference_times) if inference_times else 0
        
        # Calculate average parameters
        total_params = sum(
            model_usage.get(m, 0) * tracker.config['models']['parameters'][m]
            for m in tracker.model_list
        )
        avg_params = total_params / sum(model_usage.values()) if model_usage else 0
        
        return {
            'model': 'Adaptive',
            'parameters_M': avg_params,
            'total_frames': len(results),
            'tracked_frames': len(tracked),
            'tracking_rate': tracking_rate,
            'avg_confidence': avg_confidence,
            'avg_inference_ms': avg_inference,
            'fps': 1000 / avg_inference if avg_inference > 0 else 0,
            'lost_at_frame': len(results),
            'model_switches': len(model_switches),
            'model_usage': model_usage,
            'switches': model_switches[:10],  # First 10 switches
            'frame_results': results[:100]
        }
    
    def run_comparison(self):
        """Run fair comparison on all models"""
        
        # Select initial object (same for all models)
        initial_bbox = self.select_initial_object('high_confidence')
        
        if initial_bbox is None:
            print("No object found!")
            return None
        
        results = {}
        
        # Test fixed models
        for model_name in self.model_names:
            result = self.track_with_fixed_model(model_name, initial_bbox)
            results[model_name] = result
            print(f"  {model_name}: {result['tracking_rate']:.1%} success, "
                  f"{result['tracked_frames']}/{result['total_frames']} frames")
        
        # Test adaptive model
        adaptive_result = self.track_with_adaptive_model(initial_bbox)
        results['Adaptive'] = adaptive_result
        print(f"  Adaptive: {adaptive_result['tracking_rate']:.1%} success, "
              f"{adaptive_result['tracked_frames']}/{adaptive_result['total_frames']} frames, "
              f"{adaptive_result.get('model_switches', 0)} switches")
        
        return results
    
    def create_comparison_plot(self, results):
        """Create fair comparison visualization"""
        
        # Prepare data
        models = list(results.keys())
        tracking_rates = [results[m]['tracking_rate'] for m in models]
        avg_confidences = [results[m]['avg_confidence'] for m in models]
        tracked_frames = [results[m]['tracked_frames'] for m in models]
        total_frames = [results[m]['total_frames'] for m in models]
        model_params = [results[m]['parameters_M'] for m in models]
        fps_values = [results[m]['fps'] for m in models]
        
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Fair Comparison: All Models on MOT17-04 (Same Object)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Create subplot grid
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3, top=0.92, bottom=0.08)
        axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
        
        # Colors
        colors = ['green', 'yellow', 'orange', 'magenta', 'red', 'blue']
        
        # 1. Tracking Success Rate
        ax = axes[0]
        bars = ax.bar(models, tracking_rates, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Tracking Success Rate')
        ax.set_title('Tracking Success Rate')
        ax.set_ylim(0, 1.05)
        for bar, rate in zip(bars, tracking_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 2. Frames Tracked
        ax = axes[1]
        bars = ax.bar(models, tracked_frames, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Frames Successfully Tracked')
        ax.set_title('Total Frames Tracked (Same Object)')
        for bar, frames, total in zip(bars, tracked_frames, total_frames):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{frames}/{total}', ha='center', va='bottom', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 3. Average Confidence
        ax = axes[2]
        bars = ax.bar(models, avg_confidences, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Average Confidence')
        ax.set_title('Tracking Confidence')
        ax.set_ylim(0, 1.0)
        for bar, conf in zip(bars, avg_confidences):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{conf:.3f}', ha='center', va='bottom')
        ax.grid(True, alpha=0.3)
        
        # 4. FPS Performance
        ax = axes[3]
        bars = ax.bar(models, fps_values, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Frames Per Second')
        ax.set_title('Inference Speed (FPS)')
        for bar, fps in zip(bars, fps_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{fps:.1f}', ha='center', va='bottom')
        ax.grid(True, alpha=0.3)
        
        # 5. Model Efficiency
        ax = axes[4]
        for i, model in enumerate(models):
            marker = '*' if model == 'Adaptive' else 'o'
            size = 350 if model == 'Adaptive' else 150
            ax.scatter(model_params[i], tracking_rates[i], s=size, 
                      color=colors[i], alpha=0.7, marker=marker,
                      edgecolor='black', linewidth=1.5, label=model)
        ax.set_xlabel('Model Parameters (M)')
        ax.set_ylabel('Tracking Success Rate')
        ax.set_title('Efficiency: Parameters vs Performance')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 6. Summary Statistics
        ax = axes[5]
        ax.axis('off')
        
        # Create summary text
        summary_text = "📊 KEY FINDINGS:\n\n"
        
        # Best tracking rate
        best_rate_idx = np.argmax(tracking_rates)
        summary_text += f"✅ Best Tracking Rate:\n{models[best_rate_idx]} ({tracking_rates[best_rate_idx]:.1%})\n\n"
        
        # Most frames tracked
        best_frames_idx = np.argmax(tracked_frames)
        summary_text += f"📹 Most Frames Tracked:\n{models[best_frames_idx]} ({tracked_frames[best_frames_idx]} frames)\n\n"
        
        # Adaptive specific
        if 'Adaptive' in results:
            adaptive = results['Adaptive']
            summary_text += f"🔄 Adaptive Model:\n"
            summary_text += f"• {adaptive.get('model_switches', 0)} model switches\n"
            summary_text += f"• {adaptive['parameters_M']:.1f}M avg params\n"
            summary_text += f"• {adaptive['tracking_rate']:.1%} success rate\n\n"
        
        # Most efficient
        efficiency = [rate/params for rate, params in zip(tracking_rates, model_params)]
        best_eff_idx = np.argmax(efficiency)
        summary_text += f"⚡ Most Efficient:\n{models[best_eff_idx]} "
        summary_text += f"({efficiency[best_eff_idx]:.3f} rate/param)"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.savefig('results/adaptive/fair_comparison_mot17_04.png', 
                   dpi=150, bbox_inches='tight')
        print("\n✓ Saved fair comparison plot to results/adaptive/fair_comparison_mot17_04.png")
        
        return fig

def main():
    print("="*60)
    print("FAIR COMPARISON: ALL MODELS ON MOT17-04")
    print("="*60)
    
    comparison = FairComparison()
    results = comparison.run_comparison()
    
    if results:
        # Save results
        output_dir = Path("results/adaptive")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        with open(output_dir / "fair_comparison_results.json", "w") as f:
            # Convert numpy types for JSON serialization
            json_results = {}
            for model, data in results.items():
                json_results[model] = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else 
                        float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in data.items()
                }
            json.dump(json_results, f, indent=2)
        
        # Create comparison plot
        comparison.create_comparison_plot(results)
        
        # Create summary CSV
        summary_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Parameters (M)': [r['parameters_M'] for r in results.values()],
            'Tracking Rate': [r['tracking_rate'] for r in results.values()],
            'Frames Tracked': [r['tracked_frames'] for r in results.values()],
            'Total Frames': [r['total_frames'] for r in results.values()],
            'Avg Confidence': [r['avg_confidence'] for r in results.values()],
            'FPS': [r['fps'] for r in results.values()]
        })
        summary_df.to_csv(output_dir / "fair_comparison_summary.csv", index=False)
        
        print("\n" + "="*60)
        print("FAIR COMPARISON SUMMARY")
        print("="*60)
        print(summary_df.to_string(index=False))
        
        return results

if __name__ == "__main__":
    results = main()