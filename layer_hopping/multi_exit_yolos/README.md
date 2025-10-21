# Multi-Exit YOLOS for Pedestrian Detection

Implementation of multi-exit YOLOS architecture with detection heads at layers 8, 10, and 12 for adaptive inference on pedestrian tracking tasks.

## Overview

This project extends YOLOS (You Only Look at One Sequence) with multiple exit points for computational efficiency. The model can exit early from layers 8 or 10 when confidence is high, or continue to layer 12 for difficult cases.

**Key Features:**
- Three detection exits at transformer layers 8, 10, and 12
- Transfer learning from COCO-pretrained YOLOS-small
- Fine-tuned on MOT17 pedestrian detection dataset
- DETR-style training with Hungarian matching and auxiliary losses
- Mixed precision training (FP16) for faster convergence

## Architecture

```
YOLOS Backbone (frozen)
├── Transformer Layer 1-7
├── Layer 8  → Detection Head (NEW, trainable)
├── Layer 9
├── Layer 10 → Detection Head (NEW, trainable)
├── Layer 11
└── Layer 12 → Detection Head (pretrained bbox + NEW class head)
```

**Model Statistics:**
- Total parameters: 31.8M
- Trainable parameters: 1.5M (4.7%)
- Frozen parameters: 30.4M (95.3%)

## Directory Structure

```
multi_exit_yolos/
├── configs/
│   └── multi_exit_config.yaml      # Training configuration
├── data/
│   ├── mot_dataset.py              # MOT17 dataset loader
│   └── __init__.py
├── models/
│   ├── multi_exit_yolos.py         # Main model architecture
│   └── __init__.py
├── training/
│   ├── train.py                    # Training script
│   ├── criterion.py                # DETR loss function
│   └── __init__.py
├── evaluation/
│   ├── evaluate.py                 # Evaluation script
│   └── __init__.py
├── utils/
│   ├── box_ops.py                  # Box operations (IoU, etc.)
│   ├── matcher.py                  # Hungarian matcher
│   ├── metrics.py                  # mAP, AP50, precision/recall
│   └── __init__.py
├── results/
│   └── multi_exit_training/        # Training outputs
│       └── 20251020_165209/
│           ├── best_model.pt       # Best checkpoint (epoch 6)
│           ├── checkpoint_epoch_15.pt
│           ├── training_log.json
│           ├── training_analysis.png
│           └── evaluation_results.json
├── README.md
├── TRAINING_REPORT.md              # Detailed training analysis
└── EVALUATION_ANALYSIS.md          # Evaluation results and findings
```

## Training

### Phase 1: Detection Head Training (Completed)

**Configuration:**
- Freeze backbone: Yes
- Freeze Layer 12 bbox head: Yes (use pretrained)
- Train: Layer 8, 10 heads + Layer 12 class head
- Epochs: 15
- Learning rate: 0.0001
- Batch size: 8
- Optimizer: AdamW
- Loss: CE + L1 bbox + GIoU (DETR-style)

**Results:**
- Training time: 12 minutes
- Best validation loss: 7.99 (epoch 6)
- Training loss: 7.49 → 4.86 (35% reduction)
- All auxiliary losses properly computed (fixed validation bug)

### Running Training

```bash
CUDA_VISIBLE_DEVICES=1 python training/train.py
```

The script will:
1. Load pretrained YOLOS-small from HuggingFace
2. Initialize detection heads for layers 8, 10, 12
3. Train for 15 epochs with mixed precision
4. Save checkpoints every 5 epochs + best model
5. Generate training curves and logs

## Evaluation

### Quick Evaluation

```bash
CUDA_VISIBLE_DEVICES=1 python evaluation/evaluate.py \
  --checkpoint results/multi_exit_training/20251020_165209/best_model.pt \
  --config configs/multi_exit_config.yaml
```

### Results (Phase 1)

| Layer | mAP | AP50 | AP75 | Precision | Recall | F1 |
|-------|-----|------|------|-----------|--------|----|
| 8     | 1.34% | 3.06% | 0.83% | 4.16% | 14.68% | 6.49% |
| 10    | 4.24% | 9.81% | 1.59% | 4.93% | 16.56% | 7.60% |
| 12    | 21.04% | 34.87% | 22.57% | 24.22% | 43.86% | 31.21% |

**Key Findings:**
- Layer 12 achieves reasonable performance (21% mAP)
- Layers 8 and 10 underperform due to frozen backbone
- Clear performance hierarchy: Layer 8 < 10 < 12
- See `EVALUATION_ANALYSIS.md` for detailed analysis

### Evaluation Options

```bash
# Lower confidence threshold
--conf_threshold 0.1

# Specific batch size
--batch_size 16

# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate.py ...
```

## Configuration

Edit `configs/multi_exit_config.yaml` to modify:

**Model:**
- Backbone architecture
- Exit layer positions
- Detection head architecture
- Number of detection tokens

**Dataset:**
- Root path
- Training/validation sequences
- Image size
- Normalization parameters

**Training:**
- Learning rate and schedule
- Batch size
- Epochs
- Freeze options
- Loss weights

**Hardware:**
- Device (cuda/cpu)
- Number of workers
- Distributed training

## Key Implementation Details

### 1. Multi-Exit Forward Pass

The model supports three forward modes:
```python
# Training: return all exits
outputs = model(images)  # Returns dict with layer_8, layer_10, layer_12

# Inference: single exit
outputs = model(images, exit_layer=8)  # Returns only layer 8

# Adaptive: exit based on confidence
outputs = model(images, confidence_threshold=0.7)  # Auto-select exit
```

### 2. Loss Computation

Uses DETR-style criterion with:
- Hungarian matching for bipartite assignment
- Cross-entropy for classification
- L1 loss for bbox coordinates
- GIoU loss for bbox quality
- Auxiliary losses for layers 8 and 10

**Loss weights:**
- CE: 1.0
- BBox L1: 5.0
- GIoU: 2.0
- Empty class: 0.1

### 3. Critical Fixes

**Validation Zero-Loss Bug:**
During validation, `model.eval()` caused auxiliary outputs to not be returned. Fixed by explicitly passing `output_all_exits=True` in validation loop.

**mAP Computation Optimization:**
Original O(n³) implementation caused evaluation to hang. Optimized to O(n) by tracking image indices for each prediction.

## Dependencies

```
torch >= 2.0
torchvision
transformers (HuggingFace)
PyYAML
numpy
matplotlib
tqdm
```

Install with:
```bash
pip install torch torchvision transformers pyyaml numpy matplotlib tqdm
```

## Dataset

**MOT17 Format:**
```
data/MOT17/train/
├── MOT17-02-FRCNN/
│   ├── img1/           # Images
│   └── gt/gt.txt       # Ground truth
├── MOT17-04-FRCNN/
├── ...
```

Ground truth format (CSV):
```
frame_id, track_id, x, y, w, h, conf, class, visibility
```

## Known Issues and Future Work

### Current Limitations:
1. **Early exit performance**: Layers 8 and 10 have low mAP (<5%)
2. **Frozen backbone**: Features not optimized for early detection
3. **Simple heads**: 3-layer MLPs may be too shallow

### Planned Improvements:
1. **Phase 2 training**: Unfreeze backbone with small LR (1e-6)
2. **Deeper detection heads**: 5-layer MLPs with residual connections
3. **Multi-scale fusion**: Combine features from adjacent layers
4. **Knowledge distillation**: Use Layer 12 as teacher for Layers 8/10
5. **Confidence calibration**: Temperature scaling for better thresholds

### Next Steps:
- Test with lower confidence thresholds (0.1-0.2)
- Visualize predictions to understand failure modes
- Run Phase 2 training to improve early exits
- Implement adaptive inference strategy
- Benchmark computational savings

## Performance Targets

For practical adaptive inference, we need:
- Layer 8: >15% mAP (easy frames, 3x speedup)
- Layer 10: >20% mAP (medium frames, 2x speedup)
- Layer 12: >30% mAP (hard frames, baseline)

**Expected system performance:** 2x speedup with <10% accuracy loss

## References

- YOLOS: You Only Look at One Sequence (Fang et al., 2021)
- DETR: End-to-End Object Detection with Transformers (Carion et al., 2020)
- MOT17: Multiple Object Tracking Benchmark

## License

Research and educational use only.
