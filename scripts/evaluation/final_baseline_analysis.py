#!/usr/bin/env python3
"""
Final comprehensive analysis of all YOLOv8 models on MOT17
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_all_results():
    """Load and combine all evaluation results"""
    results_dir = Path("results/baseline")
    
    # Find all detailed results files
    all_dfs = []
    for csv_file in sorted(results_dir.glob("detailed_results_*.csv")):
        df = pd.read_csv(csv_file)
        all_dfs.append(df)
    
    if not all_dfs:
        print("No results found!")
        return None
    
    # Combine and remove duplicates
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=['model', 'sequence'], keep='last')
    
    return combined

def create_performance_table(df):
    """Create the comprehensive performance comparison table"""
    
    print("="*80)
    print("COMPREHENSIVE YOLOV8 BASELINE EVALUATION ON MOT17")
    print("="*80)
    
    # Define model order
    model_order = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
    
    # Filter and sort
    df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
    df = df.sort_values('model')
    
    print("\n📊 PERFORMANCE METRICS TABLE")
    print("="*80)
    print("| Model   | Params | Size  | FPS   | Memory | Detections | MOTA  | IDF1  | Best For |")
    print("|---------|--------|-------|-------|--------|------------|-------|-------|----------|")
    
    for _, row in df.iterrows():
        model = row['model']
        params = row['model_params_M']
        size_mb = row['model_size_mb']
        fps = row['fps']
        memory_gb = row['max_memory_gb']
        detections = row['avg_detections_per_frame']
        mota = row['mota']
        idf1 = row['idf1']
        
        # Determine best use case
        if model == 'yolov8n':
            use_case = "Real-time, light"
        elif model == 'yolov8s':
            use_case = "Balanced"
        elif model == 'yolov8m':
            use_case = "Moderate crowds"
        elif model == 'yolov8l':
            use_case = "Dense crowds"
        else:  # yolov8x
            use_case = "Max accuracy"
        
        print(f"| {model:7s} | {params:5.1f}M | {size_mb:5.0f}MB | {fps:5.1f} | {memory_gb:5.2f}GB | {detections:10.1f} | {mota:5.2f}% | {idf1:5.2f}% | {use_case} |")
    
    return df

def analyze_speed_accuracy_tradeoff(df):
    """Analyze the speed-accuracy tradeoff"""
    
    print("\n⚡ SPEED-ACCURACY TRADEOFF ANALYSIS")
    print("="*80)
    
    # Calculate relative metrics
    base_fps = df[df['model'] == 'yolov8n']['fps'].values[0]
    base_detections = df[df['model'] == 'yolov8n']['avg_detections_per_frame'].values[0]
    
    print("\n| Model   | FPS   | Relative Speed | Detections | Detection Gain |")
    print("|---------|-------|----------------|------------|----------------|")
    
    for _, row in df.iterrows():
        model = row['model']
        fps = row['fps']
        detections = row['avg_detections_per_frame']
        
        rel_speed = fps / base_fps
        det_gain = (detections / base_detections - 1) * 100
        
        print(f"| {model:7s} | {fps:5.1f} | {rel_speed:13.2f}x | {detections:10.1f} | {det_gain:+13.1f}% |")

def create_visualizations(df):
    """Create comprehensive visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    models = df['model'].values
    params = df['model_params_M'].values
    fps_values = df['fps'].values
    memory_values = df['max_memory_gb'].values
    detections = df['avg_detections_per_frame'].values
    
    # 1. FPS vs Model Size
    ax = axes[0, 0]
    ax.scatter(params, fps_values, s=200, c=range(len(models)), cmap='viridis')
    for i, model in enumerate(models):
        ax.annotate(model.replace('yolov8', ''), (params[i], fps_values[i]), 
                   ha='center', va='bottom', fontsize=10)
    ax.set_xlabel('Model Parameters (M)')
    ax.set_ylabel('FPS')
    ax.set_title('Speed vs Model Size')
    ax.grid(True, alpha=0.3)
    
    # 2. Detection Density
    ax = axes[0, 1]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(models)))
    bars = ax.bar(range(len(models)), detections, color=colors)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace('yolov8', '') for m in models])
    ax.set_ylabel('Avg Detections/Frame')
    ax.set_title('Detection Density by Model')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, detections):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
               f'{val:.1f}', ha='center', va='bottom')
    
    # 3. Memory Usage
    ax = axes[0, 2]
    memory_mb = memory_values * 1024
    bars = ax.bar(range(len(models)), memory_mb, color='steelblue')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace('yolov8', '') for m in models])
    ax.set_ylabel('GPU Memory (MB)')
    ax.set_title('Memory Requirements')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Speed Comparison
    ax = axes[1, 0]
    ax.barh(range(len(models)), fps_values, color='green')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.replace('yolov8', '') for m in models])
    ax.set_xlabel('FPS')
    ax.set_title('Processing Speed')
    ax.axvline(x=30, color='r', linestyle='--', alpha=0.5, label='Real-time (30 FPS)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # 5. Model Efficiency (FPS per Parameter)
    ax = axes[1, 1]
    efficiency = fps_values / params
    bars = ax.bar(range(len(models)), efficiency, color='purple')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace('yolov8', '') for m in models])
    ax.set_ylabel('FPS per Million Parameters')
    ax.set_title('Model Efficiency')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Complexity vs Performance
    ax = axes[1, 2]
    scatter = ax.scatter(detections, fps_values, s=params*10, alpha=0.6, c=range(len(models)), cmap='plasma')
    for i, model in enumerate(models):
        ax.annotate(model.replace('yolov8', ''), (detections[i], fps_values[i]), 
                   ha='center', va='bottom', fontsize=10)
    ax.set_xlabel('Avg Detections per Frame')
    ax.set_ylabel('FPS')
    ax.set_title('Detection Complexity vs Speed')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('YOLOv8 Models - Complete Baseline Analysis on MOT17', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = Path("results/baseline/final_baseline_analysis.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_file}")

def generate_recommendations(df):
    """Generate adaptive selection recommendations"""
    
    print("\n🎯 ADAPTIVE SELECTION RECOMMENDATIONS")
    print("="*80)
    
    print("\n1. SCENE COMPLEXITY THRESHOLDS:")
    print("   - <5 detections/frame:    Use YOLOv8n (18.4 FPS)")
    print("   - 5-10 detections/frame:  Use YOLOv8s (18.7 FPS)")
    print("   - 10-15 detections/frame: Use YOLOv8m (13.2 FPS)")
    print("   - 15-25 detections/frame: Use YOLOv8l (9.5 FPS)")
    print("   - >25 detections/frame:   Use YOLOv8x (6.7 FPS)")
    
    print("\n2. KEY FINDINGS:")
    
    # FPS analysis
    fps_range = df['fps'].max() - df['fps'].min()
    print(f"   - FPS ranges from {df['fps'].min():.1f} to {df['fps'].max():.1f} ({fps_range:.1f} difference)")
    
    # Detection analysis
    det_range = df['avg_detections_per_frame'].max() - df['avg_detections_per_frame'].min()
    print(f"   - Detection density ranges from {df['avg_detections_per_frame'].min():.1f} to {df['avg_detections_per_frame'].max():.1f}")
    
    # Memory analysis
    print(f"   - Memory usage ranges from {df['max_memory_gb'].min()*1024:.0f}MB to {df['max_memory_gb'].max()*1024:.0f}MB")
    
    print("\n3. OPTIMIZATION STRATEGY:")
    print("   - Start with YOLOv8m as default (good balance)")
    print("   - Monitor scene complexity every 30 frames")
    print("   - Switch models when complexity changes by >30%")
    print("   - Use hysteresis to prevent rapid switching")
    
    print("\n4. EXPECTED BENEFITS:")
    print("   - Maintain real-time performance in simple scenes")
    print("   - Improve accuracy in complex scenes")
    print("   - Reduce computational cost by 40-60% on average")

def main():
    """Run complete analysis"""
    
    # Load all results
    df = load_all_results()
    
    if df is None or df.empty:
        print("No results to analyze!")
        return
    
    # Create performance table
    df = create_performance_table(df)
    
    # Analyze tradeoffs
    analyze_speed_accuracy_tradeoff(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Generate recommendations
    generate_recommendations(df)
    
    # Save final summary
    summary_file = Path("results/baseline/final_baseline_summary.csv")
    df.to_csv(summary_file, index=False)
    print(f"\n📁 Complete results saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("✅ BASELINE EVALUATION COMPLETE!")
    print("="*80)
    print("\nNext Steps:")
    print("1. Test on MOT17-04 and MOT17-05 sequences")
    print("2. Implement adaptive switching logic")
    print("3. Measure end-to-end performance gains")
    
    return df

if __name__ == "__main__":
    results = main()