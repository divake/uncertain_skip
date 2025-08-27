# YOLOv8 MOT17 Baseline Evaluation Results

## Overview
Comprehensive evaluation of YOLOv8 models (nano, small, medium, large, xlarge) on MOT17 dataset to establish baseline performance metrics for adaptive model selection.

## Test Environment
- **Dataset**: MOT17-02-FRCNN (600 frames, pedestrian tracking)
- **GPU**: NVIDIA GPU 1 (to avoid resource contention on GPU 0)
- **Framework**: Ultralytics YOLOv8, SORT tracking
- **Confidence Threshold**: 0.25

## Detection Accuracy Results

### Quick Accuracy Test (50 frames)
| Model     | Params | Precision | Recall | F1    | TP  | FP  | FN  | Det/Frame |
|-----------|--------|-----------|--------|-------|-----|-----|-----|-----------|
| yolov8n   | 3.2M   | 0.654     | 0.381  | 0.481 | 300 | 159 | 488 | 9.18      |
| yolov8s   | 11.2M  | 0.757     | 0.503  | 0.604 | 396 | 127 | 392 | 10.46     |
| yolov8m   | 25.9M  | 0.687     | 0.552  | 0.612 | 435 | 198 | 353 | 12.66     |
| yolov8l   | 43.7M  | 0.691     | 0.549  | 0.612 | 433 | 194 | 355 | 12.54     |
| yolov8x   | 68.2M  | 0.709     | 0.633  | 0.669 | 499 | 205 | 289 | 14.08     |

**Key Finding**: F1 Score improvement from nano to xlarge: **39.1%**

### Performance Scaling (GPU Inference)
| Model     | Time(ms) | FPS   | Memory(MB) | Relative Speed |
|-----------|----------|-------|------------|----------------|
| yolov8n   | 12.3     | 81.3  | 344.8      | 1.00x          |
| yolov8s   | 13.9     | 71.9  | 457.3      | 1.13x          |
| yolov8m   | 18.0     | 55.6  | 644.9      | 1.46x          |
| yolov8l   | 22.2     | 45.0  | 878.3      | 1.81x          |
| yolov8x   | 29.0     | 34.5  | 1249.0     | 2.36x          |

## Tracking Performance (Initial)
| Model     | MOTA  | IDF1  | MOTP | MT | ML | FP    | FN     | ID_sw | Frag |
|-----------|-------|-------|------|----|----|-------|--------|-------|------|
| yolov8n   | 0.5%  | 1.2%  | 0.29 | 0  | 62 | 3,511 | 25,439 | 13    | 36   |
| yolov8s   | 0.8%  | 1.6%  | 0.30 | 0  | 61 | 3,897 | 24,861 | 16    | 44   |
| yolov8m   | 1.4%  | 2.2%  | 0.29 | 0  | 61 | 5,161 | 23,933 | 24    | 66   |
| yolov8l   | 1.3%  | 2.2%  | 0.29 | 0  | 61 | 5,080 | 24,049 | 25    | 68   |
| yolov8x   | 1.7%  | 2.4%  | 0.30 | 0  | 61 | 5,767 | 23,265 | 29    | 75   |

**Note**: Low tracking metrics indicate issues with track continuity that need debugging.

## Model Comparison Summary

### Detection Quality Progression
- **Best Precision**: yolov8s (0.757)
- **Best Recall**: yolov8x (0.633)
- **Best F1**: yolov8x (0.669)

### Model Size vs Accuracy Trade-off
- Nano (3.2M params): Baseline F1 0.481
- Small (11.2M params): +25% F1 improvement, 3.5x params
- Medium (25.9M params): +27% F1 improvement, 8x params
- Large (43.7M params): +27% F1 improvement, 14x params
- XLarge (68.2M params): +39% F1 improvement, 21x params

## Adaptive Selection Strategy Insights

Based on these results, an effective adaptive selection strategy should:

1. **Use YOLOv8n** for simple scenes with:
   - Low object density (<10 objects)
   - Clear visibility
   - Minimal occlusion

2. **Use YOLOv8s** for moderate scenes requiring:
   - Better precision (0.757 best among all)
   - Balanced accuracy-speed trade-off

3. **Use YOLOv8m** for complex scenes with:
   - Higher object density
   - Moderate occlusion

4. **Use YOLOv8x** for critical scenarios requiring:
   - Maximum recall (0.633)
   - Best overall F1 score (0.669)
   - Complex multi-object tracking

## Scene Complexity Thresholds (Proposed)
- **Simple**: <0.3 complexity score → yolov8n
- **Moderate**: 0.3-0.6 complexity score → yolov8s/m
- **Complex**: >0.6 complexity score → yolov8l/x

## Next Steps
1. Fix tracking metrics (MOTA/IDF1) by debugging track association
2. Test on additional MOT17 sequences (MOT17-04, MOT17-05)
3. Implement adaptive switching based on scene complexity
4. Validate end-to-end performance improvement

## File Organization
- `scripts/evaluation/`: Main evaluation scripts
- `scripts/testing/`: Quick test and validation scripts
- `scripts/debugging/`: Debug and diagnostic tools
- `src/evaluation/`: Core evaluation modules
- `src/tracking/`: SORT implementation
- `src/utils/`: Utility functions
- `results/baseline/`: All evaluation outputs and metrics