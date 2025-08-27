# Adaptive YOLO: Dynamic Model Selection for MOT17

Comprehensive baseline evaluation framework for YOLOv8 models on MOT17 dataset to enable adaptive model selection based on scene complexity.

## Project Structure

```
uncertain_skip/
├── scripts/
│   ├── evaluation/         # Main evaluation scripts
│   ├── testing/           # Quick test scripts
│   └── debugging/         # Debug and diagnostic tools
├── src/
│   ├── evaluation/         # Core evaluation modules
│   │   ├── baseline_mot_evaluation.py
│   │   └── scene_complexity.py
│   ├── tracking/           # SORT tracking implementation
│   ├── utils/              # MOT format utilities
│   └── visualization/      # Results plotting
├── results/
│   └── baseline/          # Evaluation outputs and metrics
├── data/
│   └── MOT17/             # Dataset location
└── requirements.txt       # Dependencies
```

## Features

- **Multi-Model Evaluation**: Tests YOLOv8 nano, small, medium, large, and xlarge models
- **MOT17 Integration**: Full support for MOT Challenge format and metrics
- **SORT Tracking**: Simple Online and Realtime Tracking implementation
- **Comprehensive Metrics**: MOTA, MOTP, IDF1, MT/ML, FP/FN, ID switches
- **Performance Monitoring**: FPS, GPU memory usage, inference time tracking
- **Visualization Tools**: Automated generation of comparison plots and analysis

## Quick Start

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Dataset Setup**
```bash
# MOT17 dataset already available at: data/MOT17/
```

3. **Run Quick Accuracy Test**
```bash
python scripts/testing/quick_accuracy_test.py
```

4. **Run Full Evaluation**
```bash
python scripts/evaluation/evaluate_detection_accuracy.py
```

## Latest Results (50 frames, GPU 1)

### Detection Accuracy
| Model | Precision | Recall | F1 Score | Parameters |
|-------|-----------|--------|----------|------------|
| YOLOv8n | 0.654 | 0.381 | 0.481 | 3.2M |
| YOLOv8s | 0.757 | 0.503 | 0.604 | 11.2M |
| YOLOv8m | 0.687 | 0.552 | 0.612 | 25.9M |
| YOLOv8l | 0.691 | 0.549 | 0.612 | 43.7M |
| YOLOv8x | 0.709 | 0.633 | 0.669 | 68.2M |

**Key Finding**: 39% F1 score improvement from nano to xlarge model

## Key Components

### 1. Baseline Evaluation (`src/evaluation/baseline_mot_evaluation.py`)
- Loads and tests multiple YOLO models
- Processes MOT17 sequences frame by frame
- Calculates comprehensive tracking metrics
- Generates performance comparisons

### 2. SORT Tracker (`src/tracking/sort.py`)
- Kalman filter-based state estimation
- Hungarian algorithm for association
- Configurable tracking parameters

### 3. MOT Utilities (`src/utils/mot_utils.py`)
- MOT format I/O operations
- Bounding box format conversions
- IoU calculations
- Track interpolation

### 4. Visualization (`src/visualization/plot_results.py`)
- Speed-accuracy tradeoff plots
- Memory usage comparisons
- Performance radar charts
- Detection quality analysis

## Configuration

Edit `configs/experiment_configs.yaml` to adjust:
- Detection parameters (confidence, NMS thresholds)
- Tracking parameters (max_age, min_hits, IoU threshold)
- Evaluation settings
- Output preferences

## Next Steps

1. **Fix Tracking Metrics**: Tune detection confidence and tracker parameters
2. **Complete Evaluation**: Test all 5 YOLO models on all sequences
3. **Implement Adaptive Selection**: Create scene complexity analyzer
4. **Optimize Switching**: Develop smooth model transition strategy

## License

MIT

## Citation

If you use this code, please cite:
```
@software{adaptive_yolo_2024,
  title = {Adaptive YOLO: Dynamic Model Selection for MOT17},
  year = {2024},
  url = {https://github.com/yourusername/uncertain_skip}
}
```