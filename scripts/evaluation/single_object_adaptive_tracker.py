#!/usr/bin/env python3
"""
Single Object Adaptive Tracker
Track a single object and adapt YOLO model based on tracking uncertainty
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class TrackedObject:
    """Single object being tracked"""
    id: int
    bbox: np.ndarray  # [x, y, w, h]
    confidence: float
    last_seen_frame: int
    history: List[dict]  # Track history with confidence scores
    model_used: str
    
class SingleObjectAdaptiveTracker:
    """
    Track a single object and adapt model based on tracking difficulty
    """
    
    def __init__(self, models=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']):
        self.models = {}
        self.model_names = models
        
        # Load all models
        print("Loading models...")
        for name in models:
            self.models[name] = YOLO(f'models/{name}.pt')
            self.models[name].to('cuda')
        
        # Adaptive thresholds
        self.confidence_thresholds = {
            'very_high': 0.85,  # Very confident - use smaller model
            'high': 0.70,        # Confident - use small/medium model
            'medium': 0.50,      # Uncertain - use medium/large model
            'low': 0.35          # Very uncertain - use largest model
        }
        
        # Model selection based on confidence
        self.confidence_to_model = {
            'very_high': 'yolov8n',
            'high': 'yolov8s',
            'medium': 'yolov8m',
            'low': 'yolov8l',
            'very_low': 'yolov8x'
        }
        
        # Tracking parameters
        self.iou_threshold = 0.3
        self.max_lost_frames = 5
        self.smoothing_window = 5  # Frames to average for model selection
        
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
    
    def select_initial_object(self, frame, strategy='medium_confidence'):
        """
        Select initial object to track
        
        Strategies:
        - 'medium_confidence': Select object with confidence closest to 0.6
        - 'high_confidence': Select object with highest confidence
        - 'largest': Select largest bounding box
        """
        # Use medium model for initial detection
        model = self.models['yolov8m']
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
            
        # Select based on strategy
        if strategy == 'medium_confidence':
            # Find object closest to 0.6 confidence
            target_conf = 0.6
            selected = min(candidates, key=lambda x: abs(x['confidence'] - target_conf))
        elif strategy == 'high_confidence':
            selected = max(candidates, key=lambda x: x['confidence'])
        elif strategy == 'largest':
            selected = max(candidates, key=lambda x: x['area'])
        else:
            selected = candidates[0]
            
        return TrackedObject(
            id=1,
            bbox=np.array(selected['bbox']),
            confidence=selected['confidence'],
            last_seen_frame=0,
            history=[],
            model_used='yolov8m'
        )
    
    def get_adaptive_model(self, confidence_history):
        """Select model based on recent confidence history"""
        if len(confidence_history) < 3:
            return 'yolov8m'  # Default to medium
            
        # Get recent average confidence
        recent_conf = np.mean(confidence_history[-self.smoothing_window:])
        
        # Determine confidence level and select model
        if recent_conf >= self.confidence_thresholds['very_high']:
            return self.confidence_to_model['very_high']
        elif recent_conf >= self.confidence_thresholds['high']:
            return self.confidence_to_model['high']
        elif recent_conf >= self.confidence_thresholds['medium']:
            return self.confidence_to_model['medium']
        elif recent_conf >= self.confidence_thresholds['low']:
            return self.confidence_to_model['low']
        else:
            return self.confidence_to_model['very_low']
    
    def track_object_in_frame(self, frame, tracked_obj, model_name):
        """Track object in current frame using specified model"""
        model = self.models[model_name]
        results = model(frame, conf=0.25, device='cuda', verbose=False)
        
        if results[0].boxes is None:
            return None, 0.0
            
        # Find best matching detection
        best_match = None
        best_iou = 0
        best_conf = 0
        
        for box in results[0].boxes:
            if int(box.cls) == 0:  # Person
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                det_bbox = [x1, y1, x2-x1, y2-y1]
                
                iou = self.calculate_iou(tracked_obj.bbox, det_bbox)
                
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_match = det_bbox
                    best_conf = float(box.conf)
        
        return best_match, best_conf
    
    def run_adaptive_tracking(self, video_path, max_frames=600, strategy='medium_confidence'):
        """
        Run adaptive tracking on a video/image sequence
        """
        # Load images
        if Path(video_path).is_dir():
            images = sorted(Path(video_path).glob("*.jpg"))[:max_frames]
        else:
            # Handle video file
            raise NotImplementedError("Video file support not yet implemented")
        
        print(f"Tracking strategy: {strategy}")
        print(f"Processing {len(images)} frames...")
        
        # Initialize with first frame
        first_frame = cv2.imread(str(images[0]))
        tracked_obj = self.select_initial_object(first_frame, strategy)
        
        if tracked_obj is None:
            print("No object found in first frame!")
            return None
            
        print(f"Initial object: confidence={tracked_obj.confidence:.3f}, "
              f"bbox={tracked_obj.bbox.astype(int).tolist()}")
        
        # Tracking results
        tracking_results = []
        confidence_history = [tracked_obj.confidence]
        lost_frames = 0
        
        for frame_idx, img_path in enumerate(images):
            frame = cv2.imread(str(img_path))
            
            # Get adaptive model based on confidence history
            model_name = self.get_adaptive_model(confidence_history)
            
            # Track object
            new_bbox, confidence = self.track_object_in_frame(frame, tracked_obj, model_name)
            
            if new_bbox is not None:
                # Update tracked object
                tracked_obj.bbox = np.array(new_bbox)
                tracked_obj.confidence = confidence
                tracked_obj.last_seen_frame = frame_idx
                tracked_obj.model_used = model_name
                confidence_history.append(confidence)
                lost_frames = 0
                
                status = 'tracked'
            else:
                lost_frames += 1
                confidence = 0.0
                status = 'lost' if lost_frames > self.max_lost_frames else 'searching'
                
                # Try with larger model if lost
                if lost_frames > 2 and model_name != 'yolov8x':
                    model_name = 'yolov8x'
                    new_bbox, confidence = self.track_object_in_frame(frame, tracked_obj, model_name)
                    if new_bbox is not None:
                        tracked_obj.bbox = np.array(new_bbox)
                        tracked_obj.confidence = confidence
                        tracked_obj.model_used = model_name
                        confidence_history.append(confidence)
                        lost_frames = 0
                        status = 'recovered'
            
            # Record frame result
            result = {
                'frame': frame_idx,
                'model_used': model_name,
                'confidence': confidence,
                'avg_confidence': np.mean(confidence_history[-10:]) if confidence_history else 0,
                'bbox': tracked_obj.bbox.tolist() if status != 'lost' else None,
                'status': status,
                'model_params': sum(p.numel() for p in self.models[model_name].model.parameters()) / 1e6
            }
            tracking_results.append(result)
            
            # Print progress
            if frame_idx % 50 == 0:
                avg_conf = np.mean(confidence_history[-10:]) if confidence_history else 0
                print(f"Frame {frame_idx}: model={model_name}, conf={confidence:.3f}, "
                      f"avg_conf={avg_conf:.3f}, status={status}")
            
            # Stop if object is completely lost
            if lost_frames > self.max_lost_frames:
                print(f"Object lost after frame {frame_idx}")
                break
        
        return tracking_results
    
    def analyze_results(self, results):
        """Analyze tracking results"""
        if not results:
            return None
            
        # Model usage statistics
        model_usage = {}
        for r in results:
            model = r['model_used']
            if model not in model_usage:
                model_usage[model] = 0
            model_usage[model] += 1
        
        # Calculate metrics
        tracked_frames = [r for r in results if r['status'] in ['tracked', 'recovered']]
        lost_frames = [r for r in results if r['status'] == 'lost']
        
        avg_confidence = np.mean([r['confidence'] for r in tracked_frames if r['confidence'] > 0])
        avg_params = np.mean([r['model_params'] for r in results])
        
        analysis = {
            'total_frames': len(results),
            'tracked_frames': len(tracked_frames),
            'lost_frames': len(lost_frames),
            'tracking_rate': len(tracked_frames) / len(results) if results else 0,
            'avg_confidence': avg_confidence,
            'avg_model_params': avg_params,
            'model_usage': model_usage,
            'model_switches': sum(1 for i in range(1, len(results)) 
                                 if results[i]['model_used'] != results[i-1]['model_used'])
        }
        
        return analysis

def main():
    """Run adaptive tracking experiments"""
    
    tracker = SingleObjectAdaptiveTracker()
    
    # Test both strategies
    strategies = ['medium_confidence', 'high_confidence']
    
    img_path = "data/MOT17/train/MOT17-02-FRCNN/img1"
    
    all_results = {}
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Running {strategy} strategy")
        print(f"{'='*60}")
        
        results = tracker.run_adaptive_tracking(img_path, max_frames=600, strategy=strategy)
        
        if results:
            analysis = tracker.analyze_results(results)
            
            print(f"\n📊 Analysis for {strategy}:")
            print(f"Tracking rate: {analysis['tracking_rate']:.2%}")
            print(f"Avg confidence: {analysis['avg_confidence']:.3f}")
            print(f"Avg model params: {analysis['avg_model_params']:.1f}M")
            print(f"Model switches: {analysis['model_switches']}")
            print(f"Model usage: {analysis['model_usage']}")
            
            # Save results
            output_file = f"results/adaptive/{strategy}_tracking_results.json"
            Path("results/adaptive").mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump({
                    'strategy': strategy,
                    'analysis': analysis,
                    'frame_results': results[:100]  # Save first 100 frames
                }, f, indent=2)
            
            all_results[strategy] = analysis
    
    return all_results

if __name__ == "__main__":
    main()