#!/usr/bin/env python3
"""
Enhanced Adaptive Single Object Tracker with Video Generation
Demonstrates bidirectional model switching based on tracking difficulty
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import cv2
import torch
import yaml
from ultralytics import YOLO
from pathlib import Path
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from collections import deque
import time
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.core.rl_model_selector import RLModelSelector

@dataclass
class TrackingState:
    """Enhanced tracking state with uncertainty metrics"""
    frame_idx: int
    bbox: np.ndarray
    confidence: float
    model_used: str
    uncertainty: float
    confidence_history: List[float] = field(default_factory=list)
    model_history: List[str] = field(default_factory=list)
    switch_cooldown: int = 0
    frames_since_switch: int = 0
    
class EnhancedAdaptiveTracker:
    """
    Enhanced tracker with bidirectional switching and uncertainty metrics
    """
    
    def __init__(self, config_path='configs/adaptive_tracking_config.yaml', use_rl=False, rl_weights_path=None):
        """Initialize with configuration file
        
        Args:
            config_path: Path to configuration file
            use_rl: Whether to use RL-based model selection instead of rule-based
            rl_weights_path: Path to pretrained RL weights (optional)
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load models
        self.models = {}
        self.model_list = self.config['models']['available']
        print("Loading models...")
        for name in self.model_list:
            self.models[name] = YOLO(f'models/{name}.pt')
            self.models[name].to('cuda')
            print(f"  ✓ {name} loaded")
        
        # Initialize RL model selector if requested
        self.use_rl = use_rl
        if use_rl:
            print("\n🤖 Using RL-based model selection")
            self.rl_selector = RLModelSelector(
                model_names=self.model_list,
                pretrained_path=rl_weights_path,
                training_mode=True  # Collect experiences for analysis
            )
        else:
            print("\n📏 Using rule-based model selection")
        
        # Model colors for visualization
        self.model_colors = {}
        for model, color in self.config['models']['colors'].items():
            # Convert BGR to RGB for matplotlib, keep BGR for OpenCV
            self.model_colors[model] = color
        
        # Tracking parameters
        self.iou_threshold = self.config['tracking']['iou_threshold']
        self.max_lost_frames = self.config['tracking']['max_lost_frames']
        
        # Adaptive thresholds
        self.conf_thresholds = self.config['adaptive_thresholds']['confidence_levels']
        self.variance_threshold = self.config['adaptive_thresholds']['uncertainty']['variance_threshold']
        self.smoothing_window = self.config['adaptive_thresholds']['uncertainty']['smoothing_window']
        self.switch_delay = self.config['adaptive_thresholds']['hysteresis']['frames_before_switch']
        self.cooldown = self.config['adaptive_thresholds']['hysteresis']['cooldown_frames']
        
        # For video generation
        self.video_writer = None
        
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
    
    def calculate_uncertainty(self, confidence_history, window=5):
        """Calculate tracking uncertainty based on confidence variance"""
        if len(confidence_history) < window:
            return 0.0
        
        recent = confidence_history[-window:]
        variance = np.var(recent)
        mean_conf = np.mean(recent)
        
        # Uncertainty is high when variance is high or mean confidence is low
        uncertainty = variance + (1 - mean_conf) * 0.3
        return min(uncertainty, 1.0)
    
    def select_adaptive_model_bidirectional(self, state: TrackingState):
        """
        Select model with bidirectional switching based on confidence and uncertainty
        """
        current_model = state.model_used
        current_idx = self.model_list.index(current_model)
        
        # Check if we're still in cooldown
        if state.switch_cooldown > 0:
            return current_model
            
        # Get recent confidence and uncertainty
        if len(state.confidence_history) < 3:
            return current_model
            
        recent_conf = np.mean(state.confidence_history[-self.smoothing_window:])
        uncertainty = self.calculate_uncertainty(state.confidence_history)
        
        # Determine desired model based on confidence levels
        if recent_conf >= self.conf_thresholds['very_high'] and uncertainty < 0.1:
            # Very confident and stable - switch DOWN to smallest model
            desired_model = 'yolov8n'
        elif recent_conf >= self.conf_thresholds['high'] and uncertainty < 0.2:
            # Confident - can use smaller model
            desired_model = 'yolov8s'
        elif recent_conf >= self.conf_thresholds['medium']:
            # Medium confidence - use medium model
            desired_model = 'yolov8m'
        elif recent_conf >= self.conf_thresholds['low']:
            # Low confidence - use larger model
            desired_model = 'yolov8l'
        else:
            # Very low confidence - use largest model
            desired_model = 'yolov8x'
        
        # Add hysteresis logic
        desired_idx = self.model_list.index(desired_model)
        
        # Only switch if the change is significant (at least 1 model apart)
        # or if uncertainty is high
        if abs(desired_idx - current_idx) > 0 or uncertainty > self.variance_threshold:
            if state.frames_since_switch >= self.switch_delay:
                return desired_model
        
        return current_model
    
    def select_initial_object(self, frame):
        """Select initial object based on configuration"""
        strategy = self.config['object_selection']['strategy']
        model = self.models[self.config['models']['starting_model']]
        
        results = model(frame, conf=0.25, device='cuda', verbose=False)
        
        if results[0].boxes is None:
            return None
            
        candidates = []
        for box in results[0].boxes:
            if int(box.cls) == 0:  # Person
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
            
        if strategy == 'medium_confidence':
            target_conf = self.config['object_selection']['target_confidence']
            selected = min(candidates, key=lambda x: abs(x['confidence'] - target_conf))
        elif strategy == 'high_confidence':
            selected = max(candidates, key=lambda x: x['confidence'])
        elif strategy == 'largest':
            selected = max(candidates, key=lambda x: x['area'])
        else:
            selected = candidates[0]
            
        return selected
    
    def track_object_in_frame(self, frame, bbox, model_name):
        """Track object in current frame"""
        model = self.models[model_name]
        results = model(frame, conf=self.config['tracking']['confidence_threshold'], 
                       device='cuda', verbose=False)
        
        if results[0].boxes is None:
            return None, 0.0
            
        best_match = None
        best_iou = 0
        best_conf = 0
        
        for box in results[0].boxes:
            if int(box.cls) == 0:  # Person
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                det_bbox = [x1, y1, x2-x1, y2-y1]
                
                iou = self.calculate_iou(bbox, det_bbox)
                
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_match = det_bbox
                    best_conf = float(box.conf)
        
        return best_match, best_conf
    
    def draw_frame_visualization(self, frame, state: TrackingState, fps=None):
        """Draw bounding box and information on frame"""
        if state.bbox is not None:
            x, y, w, h = state.bbox.astype(int)
            color = self.model_colors[state.model_used]
            
            # Draw bounding box with model-specific color
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # Prepare text information
            model_info = f"Model: {state.model_used}"
            conf_info = f"Conf: {state.confidence:.3f}"
            uncertainty_info = f"Uncertainty: {state.uncertainty:.3f}"
            
            # Draw background rectangles for text
            cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (300, 120), color, 2)
            
            # Draw text
            cv2.putText(frame, model_info, (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, conf_info, (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, uncertainty_info, (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if fps:
                fps_info = f"FPS: {fps:.1f}"
                cv2.putText(frame, fps_info, (frame.shape[1] - 150, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def run_adaptive_tracking(self, save_video=True):
        """Run adaptive tracking with video generation"""
        # Load configuration
        img_path = Path(self.config['dataset']['path'])
        max_frames = self.config['dataset']['max_frames']
        starting_model = self.config['models']['starting_model']
        
        # Load images
        images = sorted(img_path.glob("*.jpg"))[:max_frames]
        print(f"Processing {len(images)} frames...")
        
        # Initialize tracking
        first_frame = cv2.imread(str(images[0]))
        initial_obj = self.select_initial_object(first_frame)
        
        if initial_obj is None:
            print("No object found!")
            return None
            
        # Initialize state
        state = TrackingState(
            frame_idx=0,
            bbox=np.array(initial_obj['bbox']),
            confidence=initial_obj['confidence'],
            model_used=starting_model,
            uncertainty=0.0
        )
        
        print(f"Initial object: confidence={state.confidence:.3f}")
        print(f"Starting with model: {starting_model}")
        
        # Video writer setup
        if save_video:
            output_path = Path(self.config['visualization']['output_video_path'])
            # Change extension to .webm for GitHub compatibility
            output_path = output_path.with_suffix('.webm')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Use VP80 codec for WebM format (GitHub compatible)
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
            self.video_writer = cv2.VideoWriter(
                str(output_path), fourcc, 
                self.config['visualization']['fps'],
                (first_frame.shape[1], first_frame.shape[0])
            )
        
        # Tracking loop
        results = []
        lost_frames = 0
        
        for frame_idx, img_path in enumerate(images):
            frame = cv2.imread(str(img_path))
            start_time = time.time()
            
            # Update state
            state.frame_idx = frame_idx
            state.frames_since_switch += 1
            if state.switch_cooldown > 0:
                state.switch_cooldown -= 1
            
            # Get adaptive model using RL or rule-based selection
            if self.use_rl:
                # For RL, we need to calculate IoU from previous frame
                # This is a simplified version - in production you'd track the previous bbox
                iou_value = 0.8 if frame_idx > 0 and len(state.confidence_history) > 0 and state.confidence_history[-1] > 0 else 0.0
                
                # Use RL-based selection
                new_model = self.rl_selector.select_adaptive_model_rl(
                    confidence=state.confidence,
                    confidence_history=state.confidence_history,
                    current_model=state.model_used,
                    cooldown_frames=state.switch_cooldown,
                    iou=iou_value,
                    bbox=state.bbox,
                    frame_shape=frame.shape[:2]
                )
            else:
                # Use original rule-based selection
                new_model = self.select_adaptive_model_bidirectional(state)
            
            # Check for model switch
            if new_model != state.model_used:
                old_model = state.model_used
                state.model_used = new_model
                state.frames_since_switch = 0
                state.switch_cooldown = self.cooldown
                print(f"Frame {frame_idx}: Switching {old_model} → {new_model} "
                      f"(conf={state.confidence:.3f}, uncertainty={state.uncertainty:.3f})")
            
            # Track object
            new_bbox, confidence = self.track_object_in_frame(frame, state.bbox, state.model_used)
            
            if new_bbox is not None:
                state.bbox = np.array(new_bbox)
                state.confidence = confidence
                state.confidence_history.append(confidence)
                state.model_history.append(state.model_used)
                state.uncertainty = self.calculate_uncertainty(state.confidence_history)
                lost_frames = 0
                status = 'tracked'
            else:
                lost_frames += 1
                confidence = 0.0
                state.confidence_history.append(0.0)
                status = 'lost' if lost_frames > self.max_lost_frames else 'searching'
            
            # Calculate FPS
            inference_time = time.time() - start_time
            fps = 1.0 / inference_time if inference_time > 0 else 0
            
            # Draw visualization
            vis_frame = self.draw_frame_visualization(frame.copy(), state, fps)
            
            # Write to video
            if save_video and self.video_writer:
                self.video_writer.write(vis_frame)
            
            # Save results
            results.append({
                'frame': frame_idx,
                'model': state.model_used,
                'confidence': confidence,
                'uncertainty': state.uncertainty,
                'bbox': state.bbox.tolist() if status == 'tracked' else None,
                'status': status,
                'fps': fps
            })
            
            # Progress update
            if frame_idx % 50 == 0:
                print(f"Frame {frame_idx}: model={state.model_used}, "
                      f"conf={confidence:.3f}, uncertainty={state.uncertainty:.3f}")
            
            # Stop if lost
            if lost_frames > self.max_lost_frames:
                print(f"Object lost after frame {frame_idx}")
                break
        
        # Cleanup
        if self.video_writer:
            self.video_writer.release()
            print(f"✓ Video saved to {self.config['visualization']['output_video_path']}")
        
        # Save RL experiences if using RL mode
        if self.use_rl and hasattr(self.rl_selector, 'experiences'):
            exp_path = Path(self.config['output']['results_dir']) / 'rl_experiences.json'
            self.rl_selector.save_experiences(str(exp_path))
            
            # Also save the current RL model weights
            weights_path = Path(self.config['output']['results_dir']) / 'rl_model_weights.pth'
            self.rl_selector.save_weights(str(weights_path))
        
        return results, state
    
    def generate_analysis_plots(self, results, state: TrackingState):
        """Generate analysis plots"""
        frames = [r['frame'] for r in results]
        confidences = [r['confidence'] for r in results]
        uncertainties = [r['uncertainty'] for r in results]
        models = [r['model'] for r in results]
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.suptitle('Adaptive Single Object Tracking Analysis', fontsize=16, fontweight='bold')
        
        # 1. Confidence over time with model switches
        ax = axes[0]
        ax.plot(frames, confidences, 'b-', linewidth=2, label='Confidence')
        ax.axhline(y=self.conf_thresholds['very_high'], color='g', linestyle='--', alpha=0.5, label='Very High')
        ax.axhline(y=self.conf_thresholds['high'], color='y', linestyle='--', alpha=0.5, label='High')
        ax.axhline(y=self.conf_thresholds['medium'], color='orange', linestyle='--', alpha=0.5, label='Medium')
        ax.axhline(y=self.conf_thresholds['low'], color='r', linestyle='--', alpha=0.5, label='Low')
        
        # Mark model switches
        for i in range(1, len(models)):
            if models[i] != models[i-1]:
                ax.axvline(x=frames[i], color='purple', linestyle='-', alpha=0.3)
                ax.text(frames[i], ax.get_ylim()[1]*0.95, f'{models[i-1]}→{models[i]}', 
                       rotation=90, fontsize=8, ha='right')
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Confidence')
        ax.set_title('Tracking Confidence with Model Switches')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 2. Uncertainty over time
        ax = axes[1]
        ax.plot(frames, uncertainties, 'r-', linewidth=2)
        ax.axhline(y=self.variance_threshold, color='red', linestyle='--', alpha=0.5, 
                  label=f'Threshold ({self.variance_threshold})')
        ax.fill_between(frames, 0, uncertainties, alpha=0.3, color='red')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Uncertainty')
        ax.set_title('Tracking Uncertainty')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Model usage timeline
        ax = axes[2]
        model_to_num = {m: i for i, m in enumerate(self.model_list)}
        model_nums = [model_to_num[m] for m in models]
        
        # Create colored segments for each model
        for i in range(len(frames)):
            color = [c/255 for c in self.model_colors[models[i]]][::-1]  # BGR to RGB
            ax.barh(0, 1, left=frames[i], height=0.5, color=color, alpha=0.8)
        
        ax.set_xlim(frames[0], frames[-1])
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Frame')
        ax.set_yticks([])
        ax.set_title('Model Usage Timeline')
        
        # Add legend for models
        for model, color in self.model_colors.items():
            rgb_color = [c/255 for c in color][::-1]
            ax.barh(-1, 0, color=rgb_color, label=model)
        ax.legend(loc='upper right', ncol=5)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(self.config['output']['results_dir']) / 'adaptive_tracking_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Analysis plot saved to {output_path}")
        
        return fig

def main():
    """Run enhanced adaptive tracking with all features"""
    
    import argparse
    parser = argparse.ArgumentParser(description='Enhanced Adaptive Tracker')
    parser.add_argument('--use-rl', action='store_true', 
                       help='Use RL-based model selection instead of rule-based')
    parser.add_argument('--rl-weights', type=str, default=None,
                       help='Path to pretrained RL weights')
    args = parser.parse_args()
    
    tracker = EnhancedAdaptiveTracker(use_rl=args.use_rl, rl_weights_path=args.rl_weights)
    
    print("="*60)
    print("ENHANCED ADAPTIVE SINGLE OBJECT TRACKING")
    if args.use_rl:
        print("🤖 MODE: RL-based Model Selection (DQN)")
    else:
        print("📏 MODE: Rule-based Model Selection")
    print("="*60)
    
    # Run tracking
    results, final_state = tracker.run_adaptive_tracking(save_video=True)
    
    if results:
        # Generate analysis plots
        tracker.generate_analysis_plots(results, final_state)
        
        # Analyze results
        tracked = [r for r in results if r['status'] == 'tracked']
        model_usage = {}
        model_switches = []
        
        for i, r in enumerate(results):
            model = r['model']
            if model not in model_usage:
                model_usage[model] = 0
            model_usage[model] += 1
            
            if i > 0 and r['model'] != results[i-1]['model']:
                model_switches.append({
                    'frame': r['frame'],
                    'from': results[i-1]['model'],
                    'to': r['model'],
                    'confidence': r['confidence'],
                    'uncertainty': r['uncertainty']
                })
        
        # Calculate statistics
        avg_confidence = np.mean([r['confidence'] for r in tracked if r['confidence'] > 0])
        avg_uncertainty = np.mean([r['uncertainty'] for r in results])
        
        # Model parameters used
        param_usage = []
        for r in results:
            params = tracker.config['models']['parameters'][r['model']]
            param_usage.append(params)
        avg_params = np.mean(param_usage)
        
        # Save results
        output_dir = Path(tracker.config['output']['results_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        with open(output_dir / 'tracking_results.json', 'w') as f:
            json.dump({
                'config': tracker.config,
                'summary': {
                    'total_frames': len(results),
                    'tracked_frames': len(tracked),
                    'tracking_rate': len(tracked) / len(results),
                    'avg_confidence': float(avg_confidence),
                    'avg_uncertainty': float(avg_uncertainty),
                    'avg_model_params': float(avg_params),
                    'model_switches': len(model_switches),
                    'model_usage': model_usage
                },
                'switches': model_switches,
                'frame_results': results[:100]  # First 100 frames
            }, f, indent=2)
        
        # Print summary
        print(f"\n📊 TRACKING SUMMARY:")
        print(f"Tracking rate: {len(tracked)/len(results):.2%}")
        print(f"Avg confidence: {avg_confidence:.3f}")
        print(f"Avg uncertainty: {avg_uncertainty:.3f}")
        print(f"Avg model params: {avg_params:.1f}M")
        print(f"\n🔄 Model Switches: {len(model_switches)}")
        for switch in model_switches[:5]:  # Show first 5 switches
            print(f"  Frame {switch['frame']}: {switch['from']} → {switch['to']} "
                  f"(conf={switch['confidence']:.3f}, unc={switch['uncertainty']:.3f})")
        
        print(f"\n📈 Model Usage:")
        for model, count in sorted(model_usage.items()):
            percentage = count / len(results) * 100
            print(f"  {model}: {count} frames ({percentage:.1f}%)")
    
    return results

if __name__ == "__main__":
    main()