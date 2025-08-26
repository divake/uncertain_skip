# Adaptive YOLO: Dynamic Model Selection for MOT17

Comprehensive baseline evaluation framework for YOLOv8 models on MOT17 dataset to enable adaptive model selection based on scene complexity.

## Project Structure

```
uncertain_skip/
├── src/
│   ├── evaluation/         # Model evaluation scripts
│   ├── tracking/           # SORT tracking implementation
│   ├── utils/              # MOT format utilities
│   └── visualization/      # Results plotting
├── configs/                # Experiment configurations
├── data/
│   └── MOT17/             # Dataset location
├── models/
│   └── yolo_weights/      # Downloaded YOLO models
├── results/
│   └── baseline/          # Evaluation results
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

2. **Download MOT17 Dataset**
```bash
# Download from https://motchallenge.net/data/MOT17/
# Extract to data/MOT17/
```

3. **Run Evaluation**
```bash
python src/evaluation/baseline_mot_evaluation.py
```

4. **Analyze Results**
```bash
python analyze_results.py
```

## Initial Results

| Model | FPS | Memory | Parameters | MOTA | IDF1 |
|-------|-----|--------|------------|------|------|
| YOLOv8n | 18.42 | 32 MB | 3.16M | 0.50% | 1.18% |
| YOLOv8s | 18.67 | <1 MB | 11.17M | 0.56% | 1.20% |

*Note: MOTA/IDF1 scores need tuning - detection pipeline is working but tracking parameters require adjustment.*

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