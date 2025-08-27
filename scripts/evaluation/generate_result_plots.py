#!/usr/bin/env python3
"""
Generate comprehensive result plots from evaluation data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_result_plots():
    """Generate all evaluation plots"""
    
    # Read accuracy results
    results_file = Path("results/baseline/quick_accuracy_results.csv")
    df = pd.read_csv(results_file)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('YOLOv8 MOT17 Baseline Evaluation Results', fontsize=16, fontweight='bold')
    
    models = df['model'].str.replace('yolov8', '').values
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#DDA0DD']
    
    # 1. F1 Score Comparison
    ax = axes[0, 0]
    bars = ax.bar(models, df['f1'], color=colors, alpha=0.7)
    ax.set_xlabel('Model')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score by Model Size')
    ax.set_ylim(0, 0.8)
    for bar, val in zip(bars, df['f1']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    ax.grid(True, alpha=0.3)
    
    # 2. Precision vs Recall
    ax = axes[0, 1]
    ax.scatter(df['recall'], df['precision'], s=df['params_M']*3, 
               c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    for i, model in enumerate(models):
        ax.annotate(model, (df.iloc[i]['recall'], df.iloc[i]['precision']),
                   xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Recall Trade-off')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.3, 0.7)
    ax.set_ylim(0.6, 0.8)
    
    # 3. Model Size vs Performance
    ax = axes[0, 2]
    ax2 = ax.twinx()
    line1 = ax.plot(models, df['f1'], 'b-o', linewidth=2, markersize=8, label='F1 Score')
    line2 = ax2.plot(models, df['params_M'], 'r-s', linewidth=2, markersize=8, label='Parameters (M)')
    ax.set_xlabel('Model')
    ax.set_ylabel('F1 Score', color='b')
    ax2.set_ylabel('Parameters (M)', color='r')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.set_title('Model Size vs Performance')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 4. Detection Components
    ax = axes[1, 0]
    width = 0.25
    x = np.arange(len(models))
    ax.bar(x - width, df['tp'], width, label='True Positives', color='green', alpha=0.7)
    ax.bar(x, df['fp'], width, label='False Positives', color='orange', alpha=0.7)
    ax.bar(x + width, df['fn'], width, label='False Negatives', color='red', alpha=0.7)
    ax.set_xlabel('Model')
    ax.set_ylabel('Count')
    ax.set_title('Detection Components (50 frames)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Detections per Frame
    ax = axes[1, 1]
    bars = ax.bar(models, df['detections_per_frame'], color=colors, alpha=0.7)
    ax.set_xlabel('Model')
    ax.set_ylabel('Detections per Frame')
    ax.set_title('Average Detections per Frame')
    for bar, val in zip(bars, df['detections_per_frame']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}', ha='center', va='bottom')
    ax.grid(True, alpha=0.3)
    
    # 6. Performance Improvement
    ax = axes[1, 2]
    baseline_f1 = df.iloc[0]['f1']
    improvements = ((df['f1'] - baseline_f1) / baseline_f1 * 100).values
    bars = ax.bar(models, improvements, color=colors, alpha=0.7)
    ax.set_xlabel('Model')
    ax.set_ylabel('F1 Improvement (%)')
    ax.set_title('Performance Improvement vs Nano')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("results/baseline/comprehensive_evaluation.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Also create a consolidated CSV with all metrics
    consolidated_df = pd.DataFrame({
        'Model': df['model'],
        'Parameters_M': df['params_M'],
        'Precision': df['precision'],
        'Recall': df['recall'],
        'F1_Score': df['f1'],
        'True_Positives': df['tp'],
        'False_Positives': df['fp'],
        'False_Negatives': df['fn'],
        'Avg_Confidence': df['avg_confidence'],
        'Detections_Per_Frame': df['detections_per_frame'],
        'Quality_Score': df['f1'] * np.sqrt(df['precision'] * df['recall']),
        'F1_Improvement_%': improvements
    })
    
    # Round numeric columns
    numeric_cols = consolidated_df.select_dtypes(include=[np.number]).columns
    consolidated_df[numeric_cols] = consolidated_df[numeric_cols].round(3)
    
    # Save consolidated CSV
    consolidated_path = Path("results/baseline/comprehensive_evaluation_results.csv")
    consolidated_df.to_csv(consolidated_path, index=False)
    print(f"Consolidated results saved to {consolidated_path}")
    
    return consolidated_df

if __name__ == "__main__":
    print("Generating comprehensive result plots...")
    results = create_result_plots()
    print("\nFinal Results Summary:")
    print(results.to_string(index=False))