#!/usr/bin/env python3
"""
Test script to verify the setup and run a minimal evaluation
"""

import sys
import torch
from pathlib import Path
import numpy as np
import cv2

def test_imports():
    """Test if all required packages are imported correctly"""
    print("Testing imports...")
    
    try:
        from ultralytics import YOLO
        print("✓ ultralytics (YOLOv8)")
    except ImportError as e:
        print(f"✗ ultralytics: {e}")
        return False
    
    try:
        import motmetrics
        print("✓ motmetrics")
    except ImportError as e:
        print(f"✗ motmetrics: {e}")
        return False
    
    try:
        import lap
        print("✓ lap (Hungarian algorithm)")
    except ImportError as e:
        print(f"✗ lap: {e}")
        return False
    
    try:
        from filterpy.kalman import KalmanFilter
        print("✓ filterpy (Kalman filters)")
    except ImportError as e:
        print(f"✗ filterpy: {e}")
        return False
    
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("✓ Data visualization libraries")
    except ImportError as e:
        print(f"✗ Visualization libraries: {e}")
        return False
    
    print("\nAll imports successful!")
    return True


def test_gpu():
    """Test GPU availability"""
    print("\nTesting GPU...")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("✗ CUDA not available, will use CPU")
    
    return True


def test_yolo_model():
    """Test loading a small YOLO model"""
    print("\nTesting YOLO model loading...")
    
    try:
        from ultralytics import YOLO
        
        # Load nano model (smallest)
        print("Loading YOLOv8n...")
        model = YOLO('yolov8n.pt')
        
        # Create dummy image
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Run inference
        print("Running inference on dummy image...")
        results = model(dummy_img, verbose=False)
        
        print(f"✓ Model loaded and inference successful")
        print(f"  Model parameters: {sum(p.numel() for p in model.model.parameters()) / 1e6:.2f}M")
        
        return True
        
    except Exception as e:
        print(f"✗ YOLO test failed: {e}")
        return False


def test_tracking():
    """Test SORT tracking with dummy detections"""
    print("\nTesting SORT tracking...")
    
    try:
        from src.tracking.sort import Sort
        
        # Initialize tracker
        tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)
        
        # Create dummy detections for 3 frames
        detections_per_frame = [
            np.array([[100, 100, 200, 200, 0.9],
                      [300, 300, 400, 400, 0.8]]),  # Frame 1
            np.array([[105, 105, 205, 205, 0.85],
                      [305, 295, 405, 395, 0.75]]),  # Frame 2
            np.array([[110, 110, 210, 210, 0.88],
                      [310, 290, 410, 390, 0.82]])   # Frame 3
        ]
        
        print("Processing 3 frames with 2 objects each...")
        for i, dets in enumerate(detections_per_frame):
            tracks = tracker.update(dets)
            print(f"  Frame {i+1}: {len(tracks)} tracks")
        
        print("✓ SORT tracking working")
        return True
        
    except Exception as e:
        print(f"✗ Tracking test failed: {e}")
        print("  Make sure src/tracking/sort.py exists")
        return False


def test_mot_utils():
    """Test MOT utility functions"""
    print("\nTesting MOT utilities...")
    
    try:
        from src.utils.mot_utils import calculate_iou, convert_bbox_format
        
        # Test bbox conversion
        bbox_xywh = [100, 100, 50, 50]
        bbox_xyxy = convert_bbox_format(bbox_xywh, 'xywh', 'xyxy')
        assert bbox_xyxy == [100, 100, 150, 150], "Bbox conversion failed"
        
        # Test IoU calculation
        bbox1 = [100, 100, 50, 50]
        bbox2 = [125, 125, 50, 50]
        iou = calculate_iou(bbox1, bbox2, format='xywh')
        assert 0 < iou < 1, "IoU calculation failed"
        
        print("✓ MOT utilities working")
        print(f"  Sample IoU: {iou:.3f}")
        return True
        
    except Exception as e:
        print(f"✗ MOT utils test failed: {e}")
        print("  Make sure src/utils/mot_utils.py exists")
        return False


def test_directory_structure():
    """Test if directory structure is created"""
    print("\nTesting directory structure...")
    
    required_dirs = [
        "data/MOT17",
        "src/evaluation",
        "src/tracking",
        "src/utils",
        "src/visualization",
        "configs",
        "models/yolo_weights",
        "results/baseline",
        "logs"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} missing")
            all_exist = False
    
    return all_exist


def run_minimal_evaluation():
    """Run a minimal evaluation to test the full pipeline"""
    print("\n" + "="*60)
    print("Running minimal evaluation test...")
    print("="*60)
    
    try:
        from src.evaluation.baseline_mot_evaluation import YOLOBaselineEvaluator
        from ultralytics import YOLO
        import tempfile
        import shutil
        
        # Create temporary test data
        temp_dir = Path("data/MOT17/train/MOT17-02-FRCNN")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy structure
        (temp_dir / "img1").mkdir(exist_ok=True)
        (temp_dir / "gt").mkdir(exist_ok=True)
        (temp_dir / "det").mkdir(exist_ok=True)
        
        # Create 5 dummy images
        print("\nCreating dummy test data...")
        for i in range(1, 6):
            img = np.ones((480, 640, 3), dtype=np.uint8) * 128
            # Add some rectangles to detect
            cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)
            cv2.rectangle(img, (300, 300), (400, 400), (0, 255, 0), -1)
            
            img_path = temp_dir / "img1" / f"{i:06d}.jpg"
            cv2.imwrite(str(img_path), img)
        
        # Create dummy ground truth
        gt_file = temp_dir / "gt" / "gt.txt"
        with open(gt_file, 'w') as f:
            for i in range(1, 6):
                f.write(f"{i},1,100,100,100,100,1,1,1\n")
                f.write(f"{i},2,300,300,100,100,1,1,1\n")
        
        print("✓ Dummy data created")
        
        # Test with only nano model and one sequence
        print("\nInitializing evaluator...")
        evaluator = YOLOBaselineEvaluator()
        evaluator.model_variants = ['yolov8n']  # Only test nano
        evaluator.test_sequences = ['MOT17-02-FRCNN']  # Only test one sequence
        
        print("\nLoading YOLOv8n model...")
        model, model_info = evaluator.download_and_load_model('yolov8n')
        print(f"✓ Model loaded: {model_info['parameters']/1e6:.2f}M parameters")
        
        print("\nProcessing sequence...")
        metrics = evaluator.process_sequence(model, 'yolov8n', 'MOT17-02-FRCNN')
        
        if metrics:
            print("\n✓ Minimal evaluation successful!")
            print("\nSample metrics:")
            print(f"  FPS: {metrics.get('fps', 0):.2f}")
            print(f"  Total frames: {metrics.get('total_frames', 0)}")
            print(f"  Detections per frame: {metrics.get('avg_detections_per_frame', 0):.2f}")
            return True
        else:
            print("✗ Evaluation returned no metrics")
            return False
            
    except Exception as e:
        print(f"\n✗ Minimal evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("YOLOv8 MOT17 Evaluation Setup Test")
    print("="*60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Package Imports", test_imports),
        ("GPU Availability", test_gpu),
        ("YOLO Model", test_yolo_model),
        ("SORT Tracking", test_tracking),
        ("MOT Utilities", test_mot_utils),
        ("Minimal Evaluation", run_minimal_evaluation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    total_passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 All tests passed! Setup is complete.")
        print("\nNext steps:")
        print("1. Download MOT17 dataset and extract to data/MOT17/")
        print("2. Run: python src/evaluation/baseline_mot_evaluation.py")
    else:
        print("\n⚠️ Some tests failed. Please install missing dependencies:")
        print("Run: pip install -r requirements.txt")


if __name__ == "__main__":
    main()