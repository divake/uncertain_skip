# AdaDetect: Adaptive Layer-Skipping Object Detection

Efficient object detection through dynamic transformer layer skipping using YOLOS on BDD100K dataset.

## Project Structure

```
uncertain_skip/
├── test_layer_skipping.py    # Main experiment with layer skipping
├── test_full_model.py         # Baseline YOLOS test
├── bdd100k_dataset.py         # BDD100K dataset loader
├── requirements.txt           # Python dependencies
├── layer_skipping_results.csv # Experiment results
├── layer_skipping_results.json # Raw metrics
├── layer_skipping_analysis_simple.png # Visualizations
└── bdd100k/                   # Dataset (images + annotations)
    └── 100k/
        ├── train/
        ├── val/
        └── test/
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test baseline model
python test_full_model.py

# Run layer skipping experiment
python test_layer_skipping.py
```

## Results Summary

| Configuration | Layers | FPS  | Speedup | Detection Quality |
|--------------|--------|------|---------|------------------|
| Full         | 12/12  | 170  | 1.0x    | 100%            |
| Heavy        | 10/12  | 195  | 1.14x   | 17.5%           |
| Standard     | 8/12   | 226  | 1.33x   | 3.1%            |
| **Light**    | **6/12** | **270** | **1.59x** | **56.0%**    |
| Minimal      | 4/12   | 335  | 1.97x   | 46.0%           |

**Optimal Configuration**: 6 layers provides best speed/accuracy tradeoff

## Key Findings

1. **Linear Speedup**: ~2x speedup achievable by skipping 8 layers
2. **Non-linear Accuracy**: Detection quality varies non-monotonically with layer count
3. **Sweet Spot**: 6-layer configuration retains >50% detection quality with 1.59x speedup

## Model Details

- Base Model: YOLOS-Tiny (6.5M parameters)
- Dataset: BDD100K validation set
- Device: CUDA GPU
- Input Size: 512x512

## Next Steps

- [ ] Add uncertainty quantification for adaptive selection
- [ ] Implement RL policy for automatic configuration selection
- [ ] Test on video sequences for temporal consistency
- [ ] Analyze performance vs scene complexity