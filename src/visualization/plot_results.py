"""
Visualization utilities for MOT evaluation results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


def create_comparison_plots(results_df, output_dir):
    """
    Create comprehensive comparison plots for model evaluation results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create main comparison figure
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Speed-Accuracy Tradeoff
    ax1 = plt.subplot(2, 3, 1)
    plot_speed_accuracy_tradeoff(results_df, ax1)
    
    # 2. Model Size vs Performance
    ax2 = plt.subplot(2, 3, 2)
    plot_model_size_performance(results_df, ax2)
    
    # 3. Memory Usage
    ax3 = plt.subplot(2, 3, 3)
    plot_memory_usage(results_df, ax3)
    
    # 4. Performance by Sequence
    ax4 = plt.subplot(2, 3, 4)
    plot_sequence_performance(results_df, ax4)
    
    # 5. Tracking Errors
    ax5 = plt.subplot(2, 3, 5)
    plot_tracking_errors(results_df, ax5)
    
    # 6. Radar Chart
    ax6 = plt.subplot(2, 3, 6, projection='polar')
    plot_radar_chart(results_df, ax6)
    
    plt.suptitle('YOLOv8 Models Performance on MOT17', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / "comprehensive_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comprehensive comparison to {output_file}")
    
    # Create additional detailed plots
    create_detailed_plots(results_df, output_dir)


def plot_speed_accuracy_tradeoff(df, ax):
    """Plot FPS vs MOTA to show speed-accuracy tradeoff"""
    
    models = df['model'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        ax.scatter(model_data['fps'], model_data['mota'], 
                  label=model, s=150, alpha=0.7, color=colors[i])
        
        # Add average point
        avg_fps = model_data['fps'].mean()
        avg_mota = model_data['mota'].mean()
        ax.scatter(avg_fps, avg_mota, s=200, color=colors[i], 
                  edgecolors='black', linewidth=2, marker='s')
    
    ax.set_xlabel('FPS (Frames per Second)', fontsize=12)
    ax.set_ylabel('MOTA (%)', fontsize=12)
    ax.set_title('Speed vs Accuracy Tradeoff', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)


def plot_model_size_performance(df, ax):
    """Plot model size vs tracking performance"""
    
    summary = df.groupby('model').agg({
        'model_params_M': 'first',
        'mota': 'mean',
        'idf1': 'mean'
    }).sort_values('model_params_M')
    
    x = summary['model_params_M']
    
    ax2 = ax.twinx()
    
    line1 = ax.plot(x, summary['mota'], 'o-', color='blue', 
                   label='MOTA', markersize=10, linewidth=2)
    line2 = ax2.plot(x, summary['idf1'], 's-', color='red', 
                    label='IDF1', markersize=10, linewidth=2)
    
    ax.set_xlabel('Model Parameters (Millions)', fontsize=12)
    ax.set_ylabel('MOTA (%)', color='blue', fontsize=12)
    ax2.set_ylabel('IDF1 (%)', color='red', fontsize=12)
    ax.set_title('Model Size vs Performance', fontsize=14, fontweight='bold')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='best')
    
    ax.grid(True, alpha=0.3)


def plot_memory_usage(df, ax):
    """Plot GPU memory usage by model"""
    
    memory_data = df.groupby('model')['max_memory_gb'].max().sort_values()
    
    bars = ax.bar(range(len(memory_data)), memory_data.values, color='steelblue')
    
    # Color bars based on memory usage
    for i, (bar, val) in enumerate(zip(bars, memory_data.values)):
        if val < 1:
            bar.set_color('#2ecc71')  # Green
        elif val < 2:
            bar.set_color('#f39c12')  # Orange
        else:
            bar.set_color('#e74c3c')  # Red
    
    ax.set_xticks(range(len(memory_data)))
    ax.set_xticklabels(memory_data.index, rotation=45)
    ax.set_ylabel('GPU Memory (GB)', fontsize=12)
    ax.set_title('Maximum GPU Memory Usage', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, memory_data.values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
               f'{val:.2f}', ha='center', va='bottom', fontsize=10)


def plot_sequence_performance(df, ax):
    """Plot performance across different sequences"""
    
    pivot = df.pivot_table(index='sequence', columns='model', values='mota')
    
    x = np.arange(len(pivot.index))
    width = 0.15
    
    for i, model in enumerate(pivot.columns):
        offset = (i - len(pivot.columns)/2) * width + width/2
        ax.bar(x + offset, pivot[model], width, label=model, alpha=0.8)
    
    ax.set_xlabel('Sequence', fontsize=12)
    ax.set_ylabel('MOTA (%)', fontsize=12)
    ax.set_title('Performance by Sequence', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45)
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')


def plot_tracking_errors(df, ax):
    """Plot tracking errors (ID switches and fragmentations)"""
    
    error_metrics = df.groupby('model').agg({
        'id_switches': 'sum',
        'fp': 'sum',
        'fn': 'sum'
    })
    
    # Normalize to per 1000 frames for better comparison
    total_frames = df.groupby('model')['total_frames'].sum()
    error_metrics = error_metrics.div(total_frames, axis=0) * 1000
    
    x = np.arange(len(error_metrics))
    width = 0.25
    
    ax.bar(x - width, error_metrics['id_switches'], width, 
          label='ID Switches', color='#e74c3c', alpha=0.8)
    ax.bar(x, error_metrics['fp'], width, 
          label='False Positives', color='#f39c12', alpha=0.8)
    ax.bar(x + width, error_metrics['fn'], width, 
          label='False Negatives', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Errors per 1000 frames', fontsize=12)
    ax.set_title('Tracking Errors', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(error_metrics.index, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')


def plot_radar_chart(df, ax):
    """Create radar chart for comprehensive performance comparison"""
    
    # Select top 3 models for clarity
    models_to_plot = ['yolov8n', 'yolov8m', 'yolov8x']
    
    metrics = ['MOTA', 'IDF1', 'MT%', 'Speed', 'Efficiency']
    
    # Prepare data
    radar_data = {}
    for model in models_to_plot:
        if model in df['model'].values:
            model_data = df[df['model'] == model].mean()
            
            # Normalize metrics to 0-100 scale
            radar_data[model] = [
                model_data['mota'],
                model_data['idf1'],
                model_data['mt_percentage'],
                min(100, model_data['fps'] / 1.5),  # Normalize FPS
                100 - min(100, model_data['max_memory_gb'] * 20)  # Memory efficiency
            ]
    
    # Number of variables
    num_vars = len(metrics)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    # Plot
    for model, values in radar_data.items():
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.25)
    
    # Fix axis to go in the right order
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Set y-axis limits and labels
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(['20', '40', '60', '80'], size=8)
    
    ax.set_title('Performance Overview', fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)


def create_detailed_plots(df, output_dir):
    """Create additional detailed analysis plots"""
    
    # 1. Heatmap of all metrics
    create_metrics_heatmap(df, output_dir)
    
    # 2. Time series of FPS stability
    create_fps_stability_plot(df, output_dir)
    
    # 3. Detection quality analysis
    create_detection_quality_plot(df, output_dir)


def create_metrics_heatmap(df, output_dir):
    """Create heatmap of all metrics"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Select key metrics
    metrics = ['mota', 'motp', 'idf1', 'mt_percentage', 'ml_percentage', 
              'fps', 'max_memory_gb']
    
    # Create pivot table
    summary = df.groupby('model')[metrics].mean()
    
    # Normalize metrics for better visualization
    normalized = summary.copy()
    for col in normalized.columns:
        max_val = normalized[col].max()
        if max_val > 0:
            normalized[col] = normalized[col] / max_val * 100
    
    # Create heatmap
    sns.heatmap(normalized.T, annot=True, fmt='.1f', cmap='YlOrRd', 
               cbar_kws={'label': 'Normalized Score (%)'},
               linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)
    ax.set_title('Normalized Performance Metrics Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / "metrics_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved metrics heatmap to {output_file}")


def create_fps_stability_plot(df, output_dir):
    """Analyze FPS stability across sequences"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate FPS variance for each model
    fps_stats = df.groupby('model')['fps'].agg(['mean', 'std', 'min', 'max'])
    
    x = range(len(fps_stats))
    
    # Plot with error bars
    ax.errorbar(x, fps_stats['mean'], yerr=fps_stats['std'], 
               fmt='o', capsize=5, capthick=2, markersize=10)
    
    # Add min-max range
    for i, (idx, row) in enumerate(fps_stats.iterrows()):
        ax.plot([i, i], [row['min'], row['max']], 'k-', alpha=0.3, linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(fps_stats.index, rotation=45)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('FPS', fontsize=12)
    ax.set_title('FPS Stability Analysis (Mean ± Std)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "fps_stability.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved FPS stability plot to {output_file}")


def create_detection_quality_plot(df, output_dir):
    """Analyze detection quality metrics"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Precision vs Recall
    ax1 = axes[0]
    if 'precision' in df.columns and 'recall' in df.columns:
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            ax1.scatter(model_data['recall'], model_data['precision'], 
                       label=model, s=100, alpha=0.7)
        ax1.set_xlabel('Recall', fontsize=12)
        ax1.set_ylabel('Precision', fontsize=12)
        ax1.set_title('Precision-Recall', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Detections per frame
    ax2 = axes[1]
    det_stats = df.groupby('model')['avg_detections_per_frame'].mean()
    bars = ax2.bar(range(len(det_stats)), det_stats.values, color='skyblue')
    ax2.set_xticks(range(len(det_stats)))
    ax2.set_xticklabels(det_stats.index, rotation=45)
    ax2.set_ylabel('Avg Detections/Frame', fontsize=12)
    ax2.set_title('Detection Density', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. MT/ML Distribution
    ax3 = axes[2]
    mt_ml_data = df.groupby('model')[['mt_percentage', 'ml_percentage']].mean()
    x = np.arange(len(mt_ml_data))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, mt_ml_data['mt_percentage'], width, 
                   label='Mostly Tracked', color='#2ecc71', alpha=0.8)
    bars2 = ax3.bar(x + width/2, mt_ml_data['ml_percentage'], width, 
                   label='Mostly Lost', color='#e74c3c', alpha=0.8)
    
    ax3.set_xlabel('Model', fontsize=12)
    ax3.set_ylabel('Percentage (%)', fontsize=12)
    ax3.set_title('Track Coverage', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(mt_ml_data.index, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Detection Quality Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / "detection_quality.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved detection quality plot to {output_file}")


def create_video_visualization(video_path, tracks_file, output_path, gt_file=None, max_frames=100):
    """
    Create video visualization with tracking results
    """
    cap = cv2.VideoCapture(str(video_path))
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Load tracks
    tracks = []
    with open(tracks_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                tracks.append([float(p) for p in parts[:6]])
    
    # Load ground truth if available
    gt_tracks = []
    if gt_file and Path(gt_file).exists():
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    gt_tracks.append([float(p) for p in parts[:6]])
    
    frame_num = 0
    while cap.isOpened() and frame_num < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # Draw tracks for this frame
        frame_tracks = [t for t in tracks if int(t[0]) == frame_num]
        
        for track in frame_tracks:
            track_id = int(track[1])
            x, y, w, h = track[2:6]
            
            # Generate consistent color for track ID
            color = ((track_id * 50) % 255, (track_id * 100) % 255, (track_id * 150) % 255)
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
            
            # Draw track ID
            cv2.putText(frame, f"ID:{track_id}", (int(x), int(y-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw ground truth if available
        if gt_tracks:
            frame_gt = [t for t in gt_tracks if int(t[0]) == frame_num]
            for gt in frame_gt:
                x, y, w, h = gt[2:6]
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 1)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Saved visualization video to {output_path}")