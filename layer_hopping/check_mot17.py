#!/usr/bin/env python3
"""
Check MOT17 dataset structure and statistics
"""
import os
from pathlib import Path
import pandas as pd

def analyze_mot17():
    """Analyze MOT17 dataset structure"""
    
    mot17_path = Path("../data/MOT17")
    
    print("=" * 60)
    print("MOT17 Dataset Analysis")
    print("=" * 60)
    
    # Check train/test split
    train_path = mot17_path / "train"
    test_path = mot17_path / "test"
    
    print(f"\n📁 Dataset location: {mot17_path.absolute()}")
    print(f"✅ Train folder exists: {train_path.exists()}")
    print(f"✅ Test folder exists: {test_path.exists()}")
    
    # Analyze train sequences
    print("\n📊 Training Sequences:")
    print("-" * 40)
    
    sequences = []
    for seq_dir in sorted(train_path.glob("MOT17-*-FRCNN")):
        seq_name = seq_dir.name
        
        # Count frames
        img_dir = seq_dir / "img1"
        n_frames = len(list(img_dir.glob("*.jpg")))
        
        # Check ground truth
        gt_file = seq_dir / "gt" / "gt.txt"
        has_gt = gt_file.exists()
        
        if has_gt:
            # Count unique objects
            gt_data = pd.read_csv(gt_file, header=None)
            n_objects = gt_data[1].nunique()  # Column 1 is object ID
            n_boxes = len(gt_data)
        else:
            n_objects = 0
            n_boxes = 0
            
        sequences.append({
            'sequence': seq_name.replace('-FRCNN', ''),
            'frames': n_frames,
            'has_gt': has_gt,
            'unique_objects': n_objects,
            'total_boxes': n_boxes
        })
        
        print(f"  {seq_name}: {n_frames} frames, {n_objects} objects, {n_boxes} boxes")
    
    # Summary statistics
    df = pd.DataFrame(sequences)
    print("\n📈 Summary Statistics:")
    print("-" * 40)
    print(f"Total sequences: {len(df)}")
    print(f"Total frames: {df['frames'].sum()}")
    print(f"Total unique objects: {df['unique_objects'].sum()}")
    print(f"Total bounding boxes: {df['total_boxes'].sum()}")
    print(f"Average frames per sequence: {df['frames'].mean():.0f}")
    print(f"Average objects per sequence: {df['unique_objects'].mean():.0f}")
    
    # Train/Val split suggestion
    print("\n🔄 Suggested Train/Validation Split:")
    print("-" * 40)
    
    # Use 80/20 split
    n_train = int(len(df) * 0.8)
    train_seqs = df.iloc[:n_train]
    val_seqs = df.iloc[n_train:]
    
    print("Training sequences (80%):")
    for seq in train_seqs['sequence']:
        print(f"  - {seq}")
    
    print("\nValidation sequences (20%):")
    for seq in val_seqs['sequence']:
        print(f"  - {seq}")
    
    print("\n📝 Ground Truth Format:")
    print("-" * 40)
    
    # Show example GT format
    sample_gt = seq_dir / "gt" / "gt.txt"
    if sample_gt.exists():
        with open(sample_gt, 'r') as f:
            lines = f.readlines()[:3]
        print("Format: frame,id,x,y,w,h,conf,class,visibility")
        print("Examples:")
        for line in lines:
            print(f"  {line.strip()}")
    
    # Check test data
    print("\n🧪 Test Sequences:")
    print("-" * 40)
    
    test_sequences = []
    for seq_dir in sorted(test_path.glob("MOT17-*-FRCNN")):
        seq_name = seq_dir.name
        img_dir = seq_dir / "img1"
        n_frames = len(list(img_dir.glob("*.jpg")))
        
        # Test data has no ground truth
        gt_file = seq_dir / "gt" / "gt.txt"
        has_gt = gt_file.exists()
        
        test_sequences.append({
            'sequence': seq_name.replace('-FRCNN', ''),
            'frames': n_frames,
            'has_gt': has_gt
        })
        
        print(f"  {seq_name}: {n_frames} frames (GT: {has_gt})")
    
    test_df = pd.DataFrame(test_sequences)
    print(f"\nTotal test sequences: {len(test_df)}")
    print(f"Total test frames: {test_df['frames'].sum()}")
    
    return df, test_df

if __name__ == "__main__":
    train_df, test_df = analyze_mot17()
    
    # Save to CSV for reference
    train_df.to_csv("mot17_train_stats.csv", index=False)
    test_df.to_csv("mot17_test_stats.csv", index=False)
    print("\n✅ Saved statistics to CSV files")