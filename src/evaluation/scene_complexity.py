"""
Scene complexity analyzer for adaptive model selection
"""

import numpy as np
from collections import deque
import time

class SceneComplexityAnalyzer:
    """
    Analyzes scene complexity to determine optimal YOLO model selection
    """
    
    def __init__(self, history_size=30, switch_threshold=0.3):
        """
        Args:
            history_size: Number of frames to consider for stability
            switch_threshold: Threshold for triggering model switch
        """
        self.history_size = history_size
        self.switch_threshold = switch_threshold
        
        # Track metrics over time
        self.detection_history = deque(maxlen=history_size)
        self.confidence_history = deque(maxlen=history_size)
        self.fps_history = deque(maxlen=history_size)
        
        # Model selection state
        self.current_model = 'yolov8m'  # Start with medium
        self.frames_since_switch = 0
        self.min_frames_before_switch = 30  # Stability requirement
        
        # Model complexity levels
        self.model_hierarchy = {
            'yolov8n': {'level': 0, 'target_density': 5, 'min_fps': 30},
            'yolov8s': {'level': 1, 'target_density': 10, 'min_fps': 25},
            'yolov8m': {'level': 2, 'target_density': 15, 'min_fps': 15},
            'yolov8l': {'level': 3, 'target_density': 25, 'min_fps': 10},
            'yolov8x': {'level': 4, 'target_density': 50, 'min_fps': 5}
        }
    
    def calculate_instant_complexity(self, detections, confidence_scores=None):
        """
        Calculate instantaneous complexity metrics for current frame
        
        Args:
            detections: Number of detections or detection boxes
            confidence_scores: Optional confidence scores for detections
        
        Returns:
            Dictionary with complexity metrics
        """
        # Handle different input types
        if hasattr(detections, '__len__'):
            num_detections = len(detections)
        else:
            num_detections = detections
        
        # Calculate confidence statistics if provided
        if confidence_scores is not None and len(confidence_scores) > 0:
            avg_confidence = np.mean(confidence_scores)
            confidence_variance = np.var(confidence_scores)
        else:
            avg_confidence = 1.0  # Default if not provided
            confidence_variance = 0.0
        
        # Simple complexity score based on detection density
        # More detections = higher complexity
        if num_detections < 3:
            complexity_level = "very_simple"
            complexity_score = 0.1
        elif num_detections < 7:
            complexity_level = "simple"
            complexity_score = 0.3
        elif num_detections < 12:
            complexity_level = "medium"
            complexity_score = 0.5
        elif num_detections < 20:
            complexity_level = "complex"
            complexity_score = 0.7
        else:
            complexity_level = "very_complex"
            complexity_score = 0.9
        
        # Adjust based on confidence (low confidence = more complex)
        if avg_confidence < 0.5:
            complexity_score += 0.1
        
        # High variance in confidence = complex scene
        if confidence_variance > 0.1:
            complexity_score += 0.05
        
        complexity_score = min(1.0, complexity_score)
        
        return {
            'num_detections': num_detections,
            'avg_confidence': avg_confidence,
            'confidence_variance': confidence_variance,
            'complexity_level': complexity_level,
            'complexity_score': complexity_score
        }
    
    def update_history(self, complexity_metrics, fps=None):
        """Update historical metrics"""
        self.detection_history.append(complexity_metrics['num_detections'])
        self.confidence_history.append(complexity_metrics['avg_confidence'])
        if fps is not None:
            self.fps_history.append(fps)
        
        self.frames_since_switch += 1
    
    def get_recommended_model(self, complexity_metrics, current_fps=None):
        """
        Determine recommended model based on complexity
        
        Args:
            complexity_metrics: Output from calculate_instant_complexity
            current_fps: Current processing FPS
        
        Returns:
            Recommended model name and switch confidence
        """
        complexity_score = complexity_metrics['complexity_score']
        
        # Map complexity score to model
        if complexity_score < 0.2:
            target_model = 'yolov8n'
        elif complexity_score < 0.4:
            target_model = 'yolov8s'
        elif complexity_score < 0.6:
            target_model = 'yolov8m'
        elif complexity_score < 0.8:
            target_model = 'yolov8l'
        else:
            target_model = 'yolov8x'
        
        # Calculate switching confidence
        if target_model == self.current_model:
            switch_confidence = 0.0
        else:
            # Higher confidence if complexity clearly indicates different model
            current_level = self.model_hierarchy[self.current_model]['level']
            target_level = self.model_hierarchy[target_model]['level']
            level_diff = abs(target_level - current_level)
            
            switch_confidence = min(1.0, level_diff * 0.3)
            
            # Reduce confidence if we switched recently (hysteresis)
            if self.frames_since_switch < self.min_frames_before_switch:
                switch_confidence *= 0.5
            
            # Consider FPS constraint
            if current_fps is not None:
                target_min_fps = self.model_hierarchy[target_model]['min_fps']
                if current_fps < target_min_fps * 0.8:  # Below target
                    # Should switch to lighter model
                    if target_level > current_level:
                        switch_confidence *= 0.3  # Reduce confidence for heavier model
                    else:
                        switch_confidence *= 1.5  # Increase confidence for lighter model
        
        return target_model, switch_confidence
    
    def should_switch_model(self, target_model, switch_confidence):
        """
        Determine if model switch should occur
        
        Args:
            target_model: Recommended model
            switch_confidence: Confidence in switch (0-1)
        
        Returns:
            Boolean indicating if switch should occur
        """
        # Don't switch if same model
        if target_model == self.current_model:
            return False
        
        # Don't switch too frequently
        if self.frames_since_switch < self.min_frames_before_switch:
            return False
        
        # Switch if confidence exceeds threshold
        if switch_confidence > self.switch_threshold:
            return True
        
        # Check historical trend
        if len(self.detection_history) >= self.history_size:
            recent_avg = np.mean(list(self.detection_history)[-10:])
            older_avg = np.mean(list(self.detection_history)[-30:-10])
            
            # If trend is significantly different, consider switching
            if abs(recent_avg - older_avg) > 5:
                return switch_confidence > self.switch_threshold * 0.7
        
        return False
    
    def execute_switch(self, new_model):
        """Execute model switch"""
        print(f"[Adaptive] Switching from {self.current_model} to {new_model}")
        old_model = self.current_model
        self.current_model = new_model
        self.frames_since_switch = 0
        
        return old_model, new_model
    
    def get_adaptive_summary(self):
        """Get summary of adaptive behavior"""
        if len(self.detection_history) == 0:
            return {}
        
        return {
            'current_model': self.current_model,
            'frames_since_switch': self.frames_since_switch,
            'avg_detections': np.mean(self.detection_history),
            'std_detections': np.std(self.detection_history),
            'avg_confidence': np.mean(self.confidence_history) if self.confidence_history else 0,
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0,
            'history_size': len(self.detection_history)
        }


class AdaptiveYOLOSelector:
    """
    Manages adaptive YOLO model selection during inference
    """
    
    def __init__(self, models_dict=None):
        """
        Args:
            models_dict: Dictionary of loaded YOLO models
        """
        self.models = models_dict or {}
        self.complexity_analyzer = SceneComplexityAnalyzer()
        
        # Performance tracking
        self.model_usage_count = {m: 0 for m in ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']}
        self.model_total_time = {m: 0.0 for m in self.model_usage_count.keys()}
        self.switch_count = 0
        
    def process_frame(self, frame, force_model=None):
        """
        Process a frame with adaptive model selection
        
        Args:
            frame: Input image
            force_model: Optional - force specific model
        
        Returns:
            Detection results and model used
        """
        start_time = time.time()
        
        # Select model
        if force_model:
            model_name = force_model
        else:
            model_name = self.complexity_analyzer.current_model
        
        # Get model (would be loaded in real implementation)
        if model_name not in self.models:
            print(f"Warning: {model_name} not loaded, using default")
            model_name = 'yolov8m'
        
        # Run inference (placeholder - would use actual model)
        # results = self.models[model_name](frame)
        
        # For testing, simulate results
        import random
        num_detections = random.randint(5, 20)
        confidence_scores = np.random.rand(num_detections) * 0.5 + 0.5
        
        # Calculate complexity
        complexity = self.complexity_analyzer.calculate_instant_complexity(
            num_detections, confidence_scores
        )
        
        # Calculate FPS
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        
        # Update history
        self.complexity_analyzer.update_history(complexity, fps)
        
        # Check if we should switch models
        if not force_model:
            recommended_model, switch_confidence = self.complexity_analyzer.get_recommended_model(
                complexity, fps
            )
            
            if self.complexity_analyzer.should_switch_model(recommended_model, switch_confidence):
                old_model, new_model = self.complexity_analyzer.execute_switch(recommended_model)
                self.switch_count += 1
        
        # Track usage
        self.model_usage_count[model_name] += 1
        self.model_total_time[model_name] += inference_time
        
        return {
            'detections': num_detections,
            'model_used': model_name,
            'complexity': complexity,
            'fps': fps,
            'inference_time': inference_time
        }
    
    def get_statistics(self):
        """Get adaptive selection statistics"""
        total_frames = sum(self.model_usage_count.values())
        
        stats = {
            'total_frames': total_frames,
            'switch_count': self.switch_count,
            'model_usage_percentage': {},
            'model_avg_time': {},
            'current_model': self.complexity_analyzer.current_model
        }
        
        for model in self.model_usage_count:
            if total_frames > 0:
                stats['model_usage_percentage'][model] = (
                    self.model_usage_count[model] / total_frames * 100
                )
            
            if self.model_usage_count[model] > 0:
                stats['model_avg_time'][model] = (
                    self.model_total_time[model] / self.model_usage_count[model]
                )
        
        stats.update(self.complexity_analyzer.get_adaptive_summary())
        
        return stats


# Example usage
if __name__ == "__main__":
    print("Scene Complexity Analyzer Test")
    print("="*50)
    
    analyzer = SceneComplexityAnalyzer()
    
    # Simulate different scenarios
    scenarios = [
        ("Empty hallway", 2, 0.9),
        ("Light crowd", 8, 0.85),
        ("Medium crowd", 15, 0.75),
        ("Dense crowd", 25, 0.65),
        ("Very dense crowd", 40, 0.55)
    ]
    
    for name, num_det, avg_conf in scenarios:
        complexity = analyzer.calculate_instant_complexity(
            num_det,
            [avg_conf] * num_det
        )
        
        recommended, confidence = analyzer.get_recommended_model(complexity)
        
        print(f"\n{name}:")
        print(f"  Detections: {num_det}")
        print(f"  Complexity: {complexity['complexity_level']} ({complexity['complexity_score']:.2f})")
        print(f"  Recommended: {recommended} (confidence: {confidence:.2f})")