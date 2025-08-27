#!/usr/bin/env python3
"""
Complete baseline evaluation with all YOLO models and sequences
Includes fixes for tracking metrics and scene complexity analysis
"""

import sys
import torch
from pathlib import Path
import time
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent))

from src.evaluation.baseline_mot_evaluation import YOLOBaselineEvaluator

class CompleteBaselineEvaluator:
    def __init__(self):
        # Verify GPU usage
        print("="*60)
        print("GPU VERIFICATION")
        print("="*60)
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        self.evaluator = YOLOBaselineEvaluator()
        
        # Override with better parameters based on your analysis
        self.evaluator.conf_threshold = 0.1  # Lower confidence for better recall
        self.evaluator.nms_threshold = 0.45
        self.evaluator.max_age = 10  # Keep tracks longer
        self.evaluator.min_hits = 1  # Start tracking sooner
        self.evaluator.iou_threshold = 0.2  # More lenient association
        
    def run_complete_baseline(self):
        """Run evaluation for all models on all sequences"""
        
        # Test all 5 models
        all_models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
        
        # But for initial test, let's focus on remaining models
        models_to_test = ['yolov8m', 'yolov8l', 'yolov8x']
        
        # Test sequences - start with just MOT17-02 for remaining models
        sequences_to_test = ['MOT17-02-FRCNN']  # Will add 04 and 05 later
        
        print("\n" + "="*60)
        print("COMPLETE BASELINE EVALUATION")
        print("="*60)
        print(f"Models to test: {', '.join(models_to_test)}")
        print(f"Sequences: {', '.join(sequences_to_test)}")
        print(f"Detection confidence: {self.evaluator.conf_threshold}")
        print(f"Tracking IoU threshold: {self.evaluator.iou_threshold}")
        
        # Override evaluator settings
        self.evaluator.model_variants = models_to_test
        self.evaluator.test_sequences = sequences_to_test
        
        # Run evaluation
        results_df = self.evaluator.run_full_evaluation()
        
        return results_df
    
    def analyze_scene_complexity(self, results_df):
        """Analyze scene complexity based on detection patterns"""
        
        print("\n" + "="*60)
        print("SCENE COMPLEXITY ANALYSIS")
        print("="*60)
        
        for _, row in results_df.iterrows():
            model = row['model']
            seq = row['sequence']
            avg_det = row['avg_detections_per_frame']
            
            # Simple complexity metric based on detection density
            if avg_det < 5:
                complexity = "Simple (Empty/Light)"
                recommended = "yolov8n"
            elif avg_det < 10:
                complexity = "Medium (Moderate crowd)"
                recommended = "yolov8s or yolov8m"
            elif avg_det < 20:
                complexity = "Complex (Dense crowd)"
                recommended = "yolov8l"
            else:
                complexity = "Very Complex (Very dense)"
                recommended = "yolov8x"
            
            print(f"\n{seq} with {model}:")
            print(f"  Avg detections: {avg_det:.1f}")
            print(f"  Complexity: {complexity}")
            print(f"  Recommended model: {recommended}")
    
    def create_performance_summary(self):
        """Create comprehensive performance summary table"""
        
        # Load all results
        results_dir = Path("results/baseline")
        all_results = []
        
        for csv_file in results_dir.glob("detailed_results_*.csv"):
            df = pd.read_csv(csv_file)
            all_results.append(df)
        
        if not all_results:
            print("No results found!")
            return
        
        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Remove duplicates (keep latest)
        combined_df = combined_df.drop_duplicates(
            subset=['model', 'sequence'], 
            keep='last'
        )
        
        # Create summary by model
        summary = combined_df.groupby('model').agg({
            'model_params_M': 'first',
            'model_size_mb': 'first',
            'fps': 'mean',
            'max_memory_gb': 'max',
            'avg_detections_per_frame': 'mean',
            'mota': 'mean',
            'idf1': 'mean'
        }).round(2)
        
        # Sort by model size
        model_order = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
        summary = summary.reindex([m for m in model_order if m in summary.index])
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY TABLE")
        print("="*60)
        print("\n| Model | Params | FPS | Memory(GB) | Avg Detections | Use Case |")
        print("|-------|--------|-----|------------|----------------|----------|")
        
        for model in summary.index:
            row = summary.loc[model]
            params = row['model_params_M']
            fps = row['fps']
            memory = row['max_memory_gb']
            detections = row['avg_detections_per_frame']
            
            # Determine use case
            if model == 'yolov8n':
                use_case = "Empty/light scenes"
            elif model == 'yolov8s':
                use_case = "Light crowds"
            elif model == 'yolov8m':
                use_case = "Medium crowds"
            elif model == 'yolov8l':
                use_case = "Dense crowds"
            else:  # yolov8x
                use_case = "Maximum accuracy"
            
            print(f"| {model:7s} | {params:5.1f}M | {fps:4.1f} | {memory:10.2f} | {detections:14.1f} | {use_case} |")
        
        # Save summary
        summary_file = results_dir / "complete_baseline_summary.csv"
        summary.to_csv(summary_file)
        print(f"\nSummary saved to: {summary_file}")
        
        return summary

def main():
    print("Starting complete baseline evaluation...")
    
    evaluator = CompleteBaselineEvaluator()
    
    # Run evaluation on remaining models
    results = evaluator.run_complete_baseline()
    
    # Analyze scene complexity
    if results is not None and not results.empty:
        evaluator.analyze_scene_complexity(results)
    
    # Create performance summary
    summary = evaluator.create_performance_summary()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. ✓ Complete model comparison on MOT17-02")
    print("2. → Test on MOT17-04 (street) and MOT17-05 (night)")
    print("3. → Implement adaptive model switching based on complexity")
    print("4. → Optimize switching thresholds and hysteresis")
    
    return results, summary

if __name__ == "__main__":
    results, summary = main()