# Multi-Exit YOLOS - Layer Hopping for Adaptive Object Detection

**Status**: Architecture Investigation Phase
**Last Updated**: October 11, 2025

## Quick Links

- **[Status_01.md](Status_01.md)** - Complete status report with all experiments, results, and issues
- **[prompt.md](prompt.md)** - Original problem description for Claude Code Chat

## Project Structure

```
layer_hopping/
├── configs/
│   └── experiment_config.yaml      # Central configuration file
├── src/
│   ├── 01_multi_exit_yolos.py     # Multi-exit YOLOS architecture (archived)
│   ├── 02_evaluate_baseline.py    # Baseline YOLOS evaluation (archived)
│   ├── 03_evaluate_finetuned.py   # Failed fine-tuning evaluation (archived)
│   ├── utils.py                    # Config loading, logging, device setup
│   └── data_loader.py              # MOT17 dataset loader
├── visualization/
│   └── attention_viz.py            # Attention visualization functions
├── models/
│   └── pretrained/                 # Downloaded pretrained models
├── results/                        # Experiment outputs
├── logs/                           # Log files
├── results_archive/                # Previous failed experiments
│   ├── attempt_02_finetuning/     # Fine-tuning attempt (mAP < 1%)
│   └── baseline_evaluation/        # Baseline results (mAP ~90%)
├── archive/                        # Old code versions
├── archive_naive/                  # Very old approaches
├── Status_01.md                    # **READ THIS FIRST**
└── README.md                       # This file
```

## Current Status: STOPPED - Architecture Investigation Required

### What We've Done

1. ✅ **Baseline Evaluation** (October 10)
   - Pretrained YOLOS (layer 12 only): **mAP@0.5 = 90.7%** ✓
   - Confirms model + dataset work well together

2. ❌ **Attempt 1**: Binary classification (wrong approach)
   - Achieved ~80-90% classification accuracy
   - But no bounding boxes → not real detection

3. ❌ **Attempt 2**: Fine-tuning detection heads (3 epochs)
   - Training stable, loss decreased to 2.98
   - **Results catastrophic: mAP@0.5 < 1%** (478× worse than baseline!)
   - All 4 layers (3, 6, 9, 12) essentially random guessing

4. ❌ **Attempt 3**: Train from scratch (50-300 epochs)
   - Abandoned after realizing 4-6 days needed
   - Uncertain if approach will even work

### Critical Unanswered Question

**Can intermediate transformer layers (3, 6, 9) of YOLOS actually support object detection?**

We don't know if:
- Layer 3 features contain semantic object information (or just textures)
- Layer 3 attention focuses on objects (or scattered randomly)
- Early exit is architecturally feasible for YOLOS

**We cannot proceed with training until we answer this.**

---

## Next Steps: 3 Critical Experiments

### Experiment 1: Attention Visualization (READY TO RUN)

**Goal**: See what each layer "looks at"

**Run**:
```bash
cd /ssd_4TB/divake/uncertain_skip/layer_hopping
python 01_experiment_attention.py
```

**Output**: `results/experiment_01_attention/`
- Attention maps for layers 3, 6, 9, 12
- Overlays on original images
- Side-by-side comparisons

**Success Criteria**:
- ✅ If layer 3 attention focuses on person regions → early exit feasible
- ❌ If layer 3 attention is scattered randomly → early exit NOT feasible

---

### Experiment 2: Feature Analysis (TODO)

**Goal**: Check if intermediate features have semantic information

**Method**: PCA/t-SNE on layer 3, 6, 9, 12 features

**Success Criteria**:
- ✅ Features cluster by object type → semantic info present
- ❌ Features are random → no semantic info

---

### Experiment 3: Linear Probe (TODO)

**Goal**: Test detection capability with simplest possible head

**Method**:
- Freeze all layers
- Add ONLY simple linear classifier on top
- Train for 1 epoch
- Evaluate mAP

**Success Criteria**:
- ✅ Layer 3 + linear head achieves mAP > 20% → early exit feasible
- ❌ Layer 3 + linear head achieves mAP < 5% → layer 3 useless for detection

---

## Decision Point (After Experiments)

### If Layer 3 IS Viable (mAP > 20%):
→ Continue multi-exit YOLOS with better training:
- Use pretrained detection head for initialization
- Train for 50+ epochs
- Knowledge distillation (layer 12 teaches earlier layers)

### If Layer 3 is NOT Viable (mAP < 5%):
→ Pivot strategy:
- Use different exit points (layers 8, 10, 12)
- Or switch architecture (DETR, Mask R-CNN)
- Or abandon early exit (use model distillation)

---

## Configuration

All experiments use `configs/experiment_config.yaml`:
- Dataset paths and sequences
- Model architecture parameters
- Experiment-specific settings
- Output directories

**No hardcoded values in code** - everything is configurable.

---

## Key Learnings

### What Worked:
- ✅ Pretrained YOLOS baseline (90% mAP)
- ✅ Data pipeline (correct normalization, bounding boxes)
- ✅ Training infrastructure (stable, no crashes)

### What Failed:
- ❌ Training detection heads from scratch (too slow or too bad)
- ❌ Assuming low loss → good detection (loss ≠ mAP)
- ❌ Starting training without validating architecture first

### Critical Insight:
**We were "firing in the dark"** - training without knowing if the architecture fundamentally supports our goal.

Before investing weeks in training, we must validate that intermediate layers CAN detect objects.

---

## How to Use This Repository

### 1. Read Status Report
```bash
cat Status_01.md
```
Complete history, results, analysis, and next steps.

### 2. Run Experiment 1
```bash
python 01_experiment_attention.py
```
This will take ~5-10 minutes and generate attention visualizations.

### 3. Analyze Results
Manually examine `results/experiment_01_attention/` and decide if Layer 3 is viable.

### 4. Make Decision
Based on experiments, either:
- Continue with multi-exit YOLOS (if viable)
- Pivot to different approach (if not viable)

---

## Requirements

```bash
# Activate environment
conda activate env_cu121

# Install dependencies (if not already installed)
pip install transformers torch torchvision pillow pyyaml opencv-python matplotlib tqdm numpy
```

---

## Important Notes

- **GPU**: Scripts use CUDA device 0 by default
- **Dataset**: Must have MOT17 at `/ssd_4TB/divake/uncertain_skip/data/MOT17/train`
- **Results**: All outputs saved to `results/` with experiment-specific subdirectories
- **Logs**: All logs saved to `logs/` with experiment names
- **Archives**: Old failed experiments in `results_archive/` (for reference only)

---

## File Naming Convention

- `01_xxx.py`, `02_xxx.py` - Main experiment scripts (numbered sequentially)
- `src/utils.py`, `src/data_loader.py` - Utility modules (descriptive names)
- `Status_01.md`, `Status_02.md` - Status reports (numbered by date)

**No more** `improved_`, `fixed_`, `v2_` prefixes! Use git for version control.

---

## Contact & Collaboration

This is active research with no guaranteed solution. Current status: investigating architecture feasibility before proceeding with training.

**Next action**: Run Experiment 1 and analyze attention visualizations.

---

**Last Status Update**: October 11, 2025 - See Status_01.md for details
