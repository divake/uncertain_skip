#!/usr/bin/env python3
"""
Quick test with limited frames to verify the evaluation pipeline works
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.evaluation.baseline_mot_evaluation import YOLOBaselineEvaluator

def main():
    # Create evaluator with modified settings for quick test
    evaluator = YOLOBaselineEvaluator()
    
    # Test with only nano model and limited frames
    evaluator.model_variants = ['yolov8n', 'yolov8s']  # Only test two models
    evaluator.test_sequences = ['MOT17-02-FRCNN']  # Only one sequence
    
    # Run evaluation
    print("Starting quick evaluation test...")
    results_df = evaluator.run_full_evaluation()
    
    if not results_df.empty:
        print("\n" + "="*60)
        print("QUICK TEST RESULTS")
        print("="*60)
        print("\nMetrics Summary:")
        print(results_df[['model', 'mota', 'idf1', 'fps', 'max_memory_gb']].to_string())
        
        print("\n✅ Evaluation pipeline working correctly!")
    else:
        print("❌ No results generated")

if __name__ == "__main__":
    main()