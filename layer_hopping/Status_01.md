# Status Report 01 - Multi-Exit YOLOS for Adaptive Object Detection

**Date**: October 11, 2025
**Project**: Layer-wise Early Exit in YOLOS for Computational Efficiency
**Status**: Experiments Failed - Fundamental Architecture Investigation Required

---

## 1. Project Goal: What We're Trying to Achieve

### Objective
Build a YOLOS-based object detection system with multiple early exit points that can dynamically select which transformer layer to exit at based on detection confidence, achieving computational savings without sacrificing accuracy.

### Target Architecture
```
Input Image (512×512)
    ↓
Patch Embedding (16×16 patches → 1024 tokens)
    ↓
Transformer Layer 1-3  → Detection Head 3  (Exit Point 1: ~17.6 GFLOPs, 75% savings)
    ↓
Transformer Layer 4-6  → Detection Head 6  (Exit Point 2: ~35.0 GFLOPs, 50% savings)
    ↓
Transformer Layer 7-9  → Detection Head 9  (Exit Point 3: ~52.4 GFLOPs, 25% savings)
    ↓
Transformer Layer 10-12 → Detection Head 12 (Exit Point 4: ~69.8 GFLOPs, 0% savings)
```

### Expected Behavior
- **Easy detections** (single person, clear background): Exit at layer 3 (75% GFLOPs saved)
- **Medium detections** (2-3 people, some clutter): Exit at layer 6 (50% GFLOPs saved)
- **Hard detections** (crowds, occlusions): Exit at layer 9 or 12 (25% or 0% savings)
- **Adaptive routing**: Confidence-based or RL-based decision making

### Why YOLOS?
- Pure Vision Transformer (no CNN backbone that cannot be skipped)
- 94% of computation in transformer layers (skippable via early exit)
- Global receptive field from layer 1 (architecturally supports early exit)
- Pretrained on COCO (good initialization)

### Dataset
- **MOT17**: Pedestrian tracking dataset
- **Task**: Single-class person detection
- **Training sequences**: MOT17-02, 04, 05, 09, 10 (~25,000 person instances)
- **Validation sequences**: MOT17-11, 13 (~25,873 person instances)

---

## 2. What We've Achieved So Far

### 2.1 Baseline Evaluation: Pretrained YOLOS (Layer 12 Only)

**Model**: `hustvl/yolos-small` (42.6M parameters)
**Date**: October 10, 2025
**File**: `evaluate_baseline_yolos.py`

#### Results (Validation Set):

| Confidence Threshold | mAP@0.5 | mAP@0.75 | Precision | Recall | F1 Score |
|---------------------|---------|----------|-----------|--------|----------|
| 0.1 | 55.8% | 49.2% | 19.1% | 65.8% | 29.6% |
| 0.3 | **90.7%** | **78.0%** | 28.9% | 64.0% | 39.8% |
| 0.5 | **100.0%** | **89.4%** | 38.3% | 61.8% | 47.3% |

**Analysis**:
- ✅ Pretrained YOLOS works excellently (mAP@0.5 = 90-100%)
- ✅ Confirms model and dataset are compatible
- ✅ Establishes baseline for comparison
- ⚠️ Low precision (19-38%) indicates many false positives (expected for MOT17 tracking dataset)

**Key Insight**: The pretrained detection head (layer 12) performs well, confirming that YOLOS can detect people in MOT17.

---

### 2.2 Attempt 1: Binary Classification (Wrong Approach)

**Approach**: Train simple classifiers at intermediate layers to predict "person present: yes/no"
**Result**: Good classification accuracy (~80-90%)
**Conclusion**: ❌ **Wrong task** - binary classification ≠ object detection (no bounding boxes)

**Why it failed**:
- No spatial localization (bounding boxes)
- Not comparable to baseline detection metrics
- Cannot be used for actual object detection

---

### 2.3 Attempt 2: Fine-Tuning Detection Heads (3 Epochs)

**Approach**:
- Freeze YOLOS backbone (30M parameters, all 12 transformer layers)
- Add 4 custom DETR-style detection heads at layers 3, 6, 9, 12
- Train only detection heads (12.5M parameters) from **random initialization**
- Training: 3 epochs, batch_size=4, lr=1e-4

**Architecture**: `multi_exit_yolos.py`
```python
class MultiExitYOLOS:
    - Base: Pretrained YOLOS backbone (frozen)
    - Detection heads: 4 × DetectionHead(hidden_dim=384, num_queries=100, num_classes=92)
    - Loss: Hungarian matching + Classification + L1 bbox + GIoU
```

**Training Progress**:
```
Epoch 1: train_loss=3.13, val_loss=3.13
Epoch 2: train_loss=3.04, val_loss=3.00
Epoch 3: train_loss=2.97, val_loss=2.98
```

**Observations**:
- ✅ No overfitting (validation loss decreased steadily)
- ✅ Training stable (no gradient explosions or NaN)
- ❓ Loss values seemed reasonable (~3.0)

---

#### Results (Validation Set): CATASTROPHIC FAILURE

**Date**: October 10, 2025
**File**: `evaluate_finetuned.py`
**Model**: `results/multi_exit_training/finetuned_best.pt` (259MB)

| Layer | mAP@0.5 | mAP@0.75 | Precision | Recall | F1 Score | Predictions |
|-------|---------|----------|-----------|--------|----------|-------------|
| Layer 3 | **0.076%** | 0.0001% | 3.60% | 1.62% | 2.24% | 25,718 |
| Layer 6 | **0.046%** | 0.0004% | 2.31% | 1.62% | 1.90% | 35,798 |
| Layer 9 | **0.215%** | 0.008% | 3.30% | 2.75% | 3.00% | 29,234 |
| Layer 12 | **0.189%** | 0.001% | 4.54% | 3.79% | 4.13% | 25,705 |

**Comparison to Baseline**:
```
Baseline YOLOS (layer 12, pretrained head): mAP@0.5 = 90.7%
Our Multi-Exit (layer 12, trained head):    mAP@0.5 = 0.19%

Performance degradation: 478× worse! (90.7% → 0.19%)
```

---

#### Analysis: Why It Failed So Badly

**Problem 1: Detection Heads Trained From Scratch**
- Query embeddings randomly initialized → need to learn spatial object locations from zero
- Hungarian matcher randomly initialized → need to learn assignment strategy
- DETR-style models typically need **50-300 epochs** to converge (original YOLOS: 150+ epochs)
- We only trained for **3 epochs** → barely started learning

**Problem 2: Loss vs Metrics Mismatch**
- Validation loss = 2.98 (looks reasonable)
- But mAP@0.5 < 1% (catastrophically bad)
- **Indicates**: Loss function may not be measuring the right things

**Problem 3: No Layer Hierarchy**
- Layer 9 (0.215% mAP) > Layer 12 (0.189% mAP) → makes no sense
- Later layers should be strictly better if architecture works correctly

**Problem 4: Predictions Are Random Noise**
- Making 25,000-36,000 predictions per dataset
- Only 2-4% precision → 96-98% of predictions are false positives
- Only 1-4% recall → missing 96-99% of actual people
- Model is essentially guessing randomly

---

### 2.4 Attempt 3: Training From Scratch (Abandoned)

**Approach**: Continue training for 50-300 epochs to converge detection heads
**Status**: ❌ Abandoned after ~10 epochs

**Reasoning**:
- Training too slow (~2-3 hours per epoch on MOT17)
- 50 epochs = 100-150 hours (4-6 days)
- Uncertain if approach will even work after that time investment
- Need to validate architecture feasibility first

---

## 3. Critical Issues and Open Questions

### 3.1 Fundamental Architecture Question (UNRESOLVED)

**Core Debate**: Can intermediate layers (3, 6, 9) of pretrained YOLOS actually support object detection?

#### Evidence Against (Supports Failure):
1. **Our earlier experiment**: Only layer 12 (pretrained head) worked; layers 3/6/9 gave zero results
2. **Presentation feedback**: Experts argued that transformer layers DO specialize differently
3. **Current results**: Even after training, all layers perform terribly (<1% mAP)

#### Evidence For (Supports Feasibility):
1. **Architecture**: Global receptive field from layer 1 (every layer sees entire image)
2. **Literature**: BERTxit, FastBERT, PABEE show early exit works in language transformers
3. **Theory**: Transformers do iterative refinement (not hierarchical abstraction like CNNs)

#### What We Don't Know:
- ❓ Do layer 3 features contain semantic object information, or just textures?
- ❓ Is the specialization in YOLOS learned or architectural?
- ❓ Can layer 3 attention focus on objects, or is it scattered randomly?
- ❓ Is detection (spatial localization) fundamentally harder than classification for early exit?

**THIS IS THE CRITICAL BOTTLENECK** - we cannot proceed without answering this.

---

### 3.2 Training Strategy Issues

**Problem**: Detection heads trained from random initialization need 50-300 epochs

**Potential Solutions** (untested):
1. **Use pretrained detection head as initialization**: Copy layer 12 head weights to initialize layers 3/6/9 heads
2. **Progressive unfreezing**: Unfreeze last few transformer layers (10-12) for faster adaptation
3. **Knowledge distillation**: Train layer 3 head to mimic layer 12 predictions
4. **Smaller output space**: Reduce num_queries from 100 to 10-20 for faster convergence

**But**: All solutions are pointless if layer 3 features fundamentally cannot support detection.

---

### 3.3 Loss Function Issues

**Problem**: Loss = 2.98 looks fine, but mAP < 1% is terrible

**Possible Causes**:
1. Hungarian matching may be assigning predictions to wrong targets
2. Loss weights (classification vs bbox vs GIoU) may be imbalanced
3. No-object class (class_id=91) may be dominating predictions
4. GIoU loss may not be effective for badly initialized boxes

**Need**: Better loss monitoring (log individual components: cls_loss, bbox_loss, giou_loss)

---

### 3.4 Implementation Quality Issues

**Current State**: Code is messy and disorganized
- ❌ Multiple versions of same file (train_multi_exit.py, train_multi_exit_improved.py, finetune_multi_exit.py)
- ❌ Hardcoded values everywhere (no config files)
- ❌ Results dumped in random locations
- ❌ Log files scattered
- ❌ No clear file naming convention
- ❌ Large monolithic files (hard to reuse code)

**This prevents**:
- Reproducibility
- Debugging
- Iteration speed
- Collaboration

---

## 4. Next Steps: Architecture Investigation Before More Training

### Phase 1: Validate Architecture Feasibility (CRITICAL - DO THIS FIRST)

Before any more training attempts, we MUST answer: **Can YOLOS intermediate layers support detection?**

#### Experiment 1: Attention Visualization
**Goal**: See what each layer "looks at"

**Method**:
```python
# Extract attention maps for layers 3, 6, 9, 12
outputs = model(image, output_attentions=True)
visualize_attention_maps(outputs.attentions)
```

**Success criteria**:
- ✅ Layer 3 attention focuses on person regions (not random)
- ✅ Layer 6 attention focuses on object boundaries
- ✅ Layer 12 attention focuses on full objects
- ❌ If layer 3 attention is scattered randomly → cannot detect objects

---

#### Experiment 2: Feature Analysis (PCA/t-SNE)
**Goal**: Check if intermediate features have semantic information

**Method**:
```python
# Extract layer 3, 6, 9, 12 features for 100 images
# Project to 2D using PCA
# Check if images with people cluster together
```

**Success criteria**:
- ✅ Layer 3 features cluster by object type (semantic info present)
- ❌ Layer 3 features are random (no semantic info)

---

#### Experiment 3: Linear Probe (GOLD STANDARD TEST)
**Goal**: Test detection capability with simplest possible head

**Method**:
```python
# Freeze all layers
# Add ONLY a simple linear classifier on top of layer 3
# Train for 1 epoch
# Evaluate mAP
```

**Success criteria**:
- ✅ If layer 3 + linear head achieves mAP > 20% → early exit is feasible
- ✅ If layer 6 + linear head achieves mAP > 40% → medium exit is feasible
- ❌ If layer 3 + linear head achieves mAP < 5% → layer 3 is useless for detection

**This experiment will definitively tell us if multi-exit YOLOS is possible.**

---

### Phase 2: Literature Review (Parallel to Phase 1)

**Papers to read**:
1. "You Only Look at One Sequence" (YOLOS paper) - understand architecture deeply
2. "What Do Vision Transformers Learn?" - understand layer specialization
3. "BERTxit" - understand early exit in transformers
4. "Multi-Exit Vision Transformer for Dynamic Inference" - directly relevant
5. "DETR: End-to-End Object Detection with Transformers" - understand DETR training

**Focus**:
- Do YOLOS authors mention intermediate layer representations?
- Are there ablation studies on detection at different depths?
- How do other papers handle multi-exit for detection (not just classification)?

---

### Phase 3: Decision Point (After Phase 1 Results)

#### **Decision A: If Layer 3 IS Viable (mAP > 20% with linear probe)**
→ Continue multi-exit YOLOS with improved training:
- Use pretrained detection head for initialization
- Train for 50+ epochs with proper loss monitoring
- Implement knowledge distillation (layer 12 teaches layers 3/6/9)
- Add hierarchy enforcement losses

#### **Decision B: If Layer 3 is NOT Viable (mAP < 5%)**
→ Pivot strategy:
- **Option B1**: Use different exit points (layers 8, 10, 12 instead of 3, 6, 9)
- **Option B2**: Switch to different architecture (DETR with auxiliary losses, Mask R-CNN cascade)
- **Option B3**: Abandon early exit, use model distillation (train smaller YOLOS models)

---

## 5. Key Learnings

### What Worked:
1. ✅ Pretrained YOLOS baseline works well (validates model + dataset compatibility)
2. ✅ Data pipeline is correct (boxes normalized, preprocessing valid)
3. ✅ Training infrastructure is stable (no crashes, NaN, or gradient issues)
4. ✅ Evaluation metrics are implemented correctly

### What Failed:
1. ❌ Training detection heads from scratch (3 epochs = too few, 300 epochs = too slow)
2. ❌ Assuming low loss → good detection (loss doesn't correlate with mAP)
3. ❌ Starting training without validating architecture feasibility first

### Critical Insight:
**We were "firing in the dark"** - training models without understanding if the architecture fundamentally supports our goal. Before investing weeks in training, we must validate that intermediate layers CAN detect objects.

---

## 6. Immediate Action Items

### Priority 1: Clean Up Code (IN PROGRESS)
- [x] Create this Status_01.md document
- [ ] Delete unnecessary files (old versions, duplicate code)
- [ ] Reorganize folder structure:
  ```
  layer_hopping/
  ├── configs/           # Configuration files
  ├── src/               # Core code (models, training, losses)
  ├── visualization/     # Visualization code
  ├── models/            # Downloaded/saved model weights
  ├── results/           # All outputs (plots, metrics)
  ├── logs/              # All log files
  └── Status_01.md       # This document
  ```
- [ ] Use numbered file names (01_xxx.py, 02_xxx.py)
- [ ] Create config files (no hardcoded values)
- [ ] Modularize code (small reusable functions)

### Priority 2: Implement Experiment 1 (NEXT)
- [ ] Create config file for dataset and model paths
- [ ] Implement attention visualization cleanly
- [ ] Generate attention maps for layers 3, 6, 9, 12
- [ ] Analyze results and document findings

### Priority 3: Implement Experiments 2 & 3
- [ ] Feature analysis (PCA/t-SNE)
- [ ] Linear probe test (definitive answer)

### Priority 4: Make Decision
- [ ] Based on experiments, decide: continue or pivot?
- [ ] Update this status document with findings

---

## 7. File Inventory (To Be Cleaned)

### Files to Keep:
- `multi_exit_yolos.py` - Core architecture (will refactor to src/)
- `evaluate_baseline_yolos.py` - Baseline evaluation (will refactor)
- `results/multi_exit_training/evaluation_results.json` - Failed results (for reference)
- `results/multi_exit_training/baseline_yolos_results.json` - Baseline results

### Files to Archive/Delete:
- All `*_improved.py`, `*_fixed.py` versions (duplicates)
- All `.log` files in root directory (move to logs/)
- All markdown fragments (BASELINE_RESULTS.md, DEBUGGING_REPORT.md, etc.)
- Empty result folders
- Model checkpoints from failed experiments (259MB finetuned_best.pt)

---

## 8. Success Metrics (Once Architecture is Validated)

### Minimum Viable Performance:
- Layer 3: mAP@0.5 > 30% (usable for easy cases)
- Layer 6: mAP@0.5 > 50% (usable for medium cases)
- Layer 9: mAP@0.5 > 70% (usable for most cases)
- Layer 12: mAP@0.5 > 85% (match baseline)

### Hierarchy Requirements:
- Layer 3 < Layer 6 < Layer 9 < Layer 12 (strict ordering)

### Computational Savings:
- Average GFLOPs < 50 (25% savings) when using adaptive routing
- Target: 30-40 GFLOPs average (40-60% savings)

---

## 9. Conclusion

We have completed baseline evaluation and two failed training attempts. The core issue is **we don't know if multi-exit YOLOS is architecturally feasible**.

Before any more training, we must validate that intermediate transformer layers contain enough information to support object detection. This requires running 3 focused experiments (attention visualization, feature analysis, linear probe).

Only after validating feasibility should we invest time in long training runs.

**Current Status**: Paused for architecture investigation.
**Next Action**: Clean up codebase, then implement Experiment 1 (attention visualization).

---

**End of Status Report 01**
