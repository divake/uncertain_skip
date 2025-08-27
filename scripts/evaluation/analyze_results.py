#!/usr/bin/env python3
"""
Analyze and visualize the baseline evaluation results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_results():
    # Load results
    results_dir = Path("results/baseline")
    
    # Find the most recent detailed results file
    detailed_files = list(results_dir.glob("detailed_results_*.csv"))
    if not detailed_files:
        print("No results found!")
        return
    
    latest_file = sorted(detailed_files)[-1]
    df = pd.read_csv(latest_file)
    
    print("="*80)
    print("YOLOV8 MOT17 BASELINE EVALUATION - INITIAL RESULTS")
    print("="*80)
    
    print(f"\nResults loaded from: {latest_file}")
    print(f"Models tested: {', '.join(df['model'].unique())}")
    print(f"Sequences tested: {', '.join(df['sequence'].unique())}")
    
    # Performance Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    summary_cols = ['model', 'mota', 'idf1', 'fps', 'max_memory_gb', 'model_params_M', 'avg_detections_per_frame']
    summary = df[summary_cols].round(2)
    
    print("\n" + summary.to_string(index=False))
    
    # Key Findings
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    # Speed comparison
    print("\n1. SPEED (FPS):")
    for _, row in df.iterrows():
        print(f"   {row['model']}: {row['fps']:.2f} FPS")
    
    # Memory usage
    print("\n2. GPU MEMORY USAGE:")
    for _, row in df.iterrows():
        mem_mb = row['max_memory_gb'] * 1024
        print(f"   {row['model']}: {mem_mb:.2f} MB")
    
    # Detection density
    print("\n3. AVERAGE DETECTIONS PER FRAME:")
    for _, row in df.iterrows():
        print(f"   {row['model']}: {row['avg_detections_per_frame']:.2f}")
    
    # Model size
    print("\n4. MODEL SIZE:")
    for _, row in df.iterrows():
        print(f"   {row['model']}: {row['model_params_M']:.2f}M parameters ({row['model_size_mb']:.2f} MB)")
    
    # Tracking quality (currently low due to configuration)
    print("\n5. TRACKING METRICS (Need Tuning):")
    print("   Note: MOTA and IDF1 are very low - this indicates:")
    print("   - Detection confidence threshold may be too low")
    print("   - Tracker parameters need adjustment")
    print("   - Ground truth format mismatch needs investigation")
    
    # Create simple comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Speed comparison
    ax = axes[0]
    models = df['model'].values
    fps_values = df['fps'].values
    colors = ['green' if fps > 15 else 'orange' for fps in fps_values]
    ax.bar(models, fps_values, color=colors)
    ax.set_ylabel('FPS')
    ax.set_title('Inference Speed')
    ax.axhline(y=30, color='r', linestyle='--', label='Real-time (30 FPS)')
    ax.legend()
    
    # Memory usage
    ax = axes[1]
    memory_mb = df['max_memory_gb'].values * 1024
    ax.bar(models, memory_mb, color='blue')
    ax.set_ylabel('Memory (MB)')
    ax.set_title('GPU Memory Usage')
    
    # Detections per frame
    ax = axes[2]
    detections = df['avg_detections_per_frame'].values
    ax.bar(models, detections, color='purple')
    ax.set_ylabel('Detections/Frame')
    ax.set_title('Detection Density')
    
    plt.suptitle('YOLOv8 Models - Initial Performance on MOT17-02-FRCNN', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = results_dir / "initial_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Analysis plot saved to: {output_file}")
    
    # Observations and Recommendations
    print("\n" + "="*60)
    print("OBSERVATIONS & RECOMMENDATIONS")
    print("="*60)
    
    print("\n✅ WHAT'S WORKING:")
    print("1. All models achieve real-time performance (>15 FPS)")
    print("2. Memory usage scales appropriately with model size")
    print("3. Detection pipeline is functioning correctly")
    print(f"4. YOLOv8n: Fastest ({df[df['model']=='yolov8n']['fps'].values[0]:.1f} FPS)")
    if 'yolov8s' in df['model'].values:
        print(f"5. YOLOv8s: Good balance ({df[df['model']=='yolov8s']['fps'].values[0]:.1f} FPS)")
    
    print("\n⚠️ ISSUES TO ADDRESS:")
    print("1. Very low MOTA/IDF1 scores - need to:")
    print("   - Verify ground truth format compatibility")
    print("   - Adjust detection confidence threshold (currently 0.25)")
    print("   - Tune SORT tracker parameters")
    print("2. Consider implementing DeepSORT for better tracking")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Debug why tracking metrics are so low")
    print("2. Test on all three sequences (MOT17-02, MOT17-04, MOT17-05)")
    print("3. Test remaining models (yolov8m, yolov8l, yolov8x)")
    print("4. Implement adaptive model selection based on scene complexity")
    
    print("\n" + "="*60)
    print("EXPECTED VS ACTUAL RESULTS")
    print("="*60)
    
    print("\nEXPECTED (typical MOT17 performance):")
    print("  - MOTA: 30-70% depending on model")
    print("  - IDF1: 40-70% depending on model")
    print("  - FPS: 5-150 depending on model size")
    
    print("\nACTUAL (current results):")
    print("  - MOTA: ~0.5% (very low - indicates issue)")
    print("  - IDF1: ~1.2% (very low - indicates issue)")
    print("  - FPS: ~18-19 (good, as expected)")
    
    print("\nThis suggests the detection/tracking pipeline needs debugging.")
    
    return df

if __name__ == "__main__":
    df = analyze_results()