#!/usr/bin/env python3
"""
Simple runner script for adaptive tracking demo
Easy to modify for different configurations
"""

import os
import sys
import yaml
from pathlib import Path

# ==============================================================================
# EASY CONFIGURATION SECTION - MODIFY THESE VALUES AS NEEDED
# ==============================================================================

# Which dataset/sequence to use
DATASET_PATH = "data/MOT17/train/MOT17-04-FRCNN/img1"  # MOT17-04: Busy street crossing with 1050 frames!
MAX_FRAMES = 1050  # Process ALL 1050 frames!

# Model to start with
STARTING_MODEL = "yolov8n"  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x

# Object selection strategy
OBJECT_STRATEGY = "high_confidence"  # Changed to track most confident object for longer tracking
TARGET_CONFIDENCE = 0.8  # For medium_confidence strategy (not used now)

# Switching thresholds (when to switch models)
CONFIDENCE_THRESHOLDS = {
    "very_high": 0.85,  # Switch to smaller model
    "high": 0.70,
    "medium": 0.50,
    "low": 0.35,
    "very_low": 0.25    # Switch to larger model
}

# Video output
GENERATE_VIDEO = True
VIDEO_OUTPUT_PATH = "results/adaptive/MOT17-04_adaptive_tracking_1050frames.mp4"

# ==============================================================================
# END OF CONFIGURATION SECTION
# ==============================================================================

def update_config():
    """Update the YAML configuration with the values above"""
    
    config_path = Path("configs/adaptive_tracking_config.yaml")
    
    # Load existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update with our values
    config['dataset']['path'] = DATASET_PATH
    config['dataset']['max_frames'] = MAX_FRAMES
    config['models']['starting_model'] = STARTING_MODEL
    config['object_selection']['strategy'] = OBJECT_STRATEGY
    config['object_selection']['target_confidence'] = TARGET_CONFIDENCE
    config['adaptive_thresholds']['confidence_levels'] = CONFIDENCE_THRESHOLDS
    config['visualization']['generate_video'] = GENERATE_VIDEO
    config['visualization']['output_video_path'] = VIDEO_OUTPUT_PATH
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("✓ Configuration updated")
    return config

def run_demo():
    """Run the adaptive tracking demo"""
    
    print("="*60)
    print("ADAPTIVE SINGLE OBJECT TRACKING DEMO")
    print("="*60)
    
    # Update configuration
    config = update_config()
    
    print("\n📋 Current Configuration:")
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Max frames: {MAX_FRAMES}")
    print(f"  Starting model: {STARTING_MODEL}")
    print(f"  Object selection: {OBJECT_STRATEGY}")
    print(f"  Generate video: {GENERATE_VIDEO}")
    
    # Run the tracker
    print("\n🚀 Starting adaptive tracking...")
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1
    
    # Import and run the tracker
    from src.core.adaptive_tracker import EnhancedAdaptiveTracker
    
    # Create tracker (use_rl=False for rule-based, use_rl=True for RL-based)
    tracker = EnhancedAdaptiveTracker(use_rl=False)
    results, state = tracker.run_adaptive_tracking(save_video=GENERATE_VIDEO)
    
    if results:
        # Generate plots
        tracker.generate_analysis_plots(results, state)
        
        # Print summary
        tracked = [r for r in results if r['status'] == 'tracked']
        print(f"\n✅ Tracking Complete!")
        print(f"  Frames tracked: {len(tracked)}/{len(results)}")
        print(f"  Success rate: {len(tracked)/len(results)*100:.1f}%")
        
        if GENERATE_VIDEO:
            print(f"  Video saved to: {VIDEO_OUTPUT_PATH}")
        
        # Model usage
        model_usage = {}
        for r in results:
            model = r['model']
            if model not in model_usage:
                model_usage[model] = 0
            model_usage[model] += 1
        
        print("\n📊 Model Usage:")
        for model, count in sorted(model_usage.items()):
            print(f"  {model}: {count} frames ({count/len(results)*100:.1f}%)")
        
        # Count switches
        switches = sum(1 for i in range(1, len(results)) 
                      if results[i]['model'] != results[i-1]['model'])
        print(f"\n🔄 Total model switches: {switches}")
        
        # Show bidirectional switching
        print("\n🔀 Switching Pattern:")
        prev_model = results[0]['model']
        for i, r in enumerate(results[1:], 1):
            if r['model'] != prev_model:
                direction = "↑" if r['model'] > prev_model else "↓"
                print(f"  Frame {r['frame']}: {prev_model} {direction} {r['model']}")
                prev_model = r['model']
    else:
        print("❌ No tracking results generated")

if __name__ == "__main__":
    run_demo()