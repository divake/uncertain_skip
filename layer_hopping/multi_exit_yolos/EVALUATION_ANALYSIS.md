# Multi-Exit YOLOS Evaluation Analysis

**Date:** October 21, 2025
**Model:** Best checkpoint from Phase 1 training (Epoch 6)
**Dataset:** MOT17 validation set (1,650 images, 2 sequences)
**Confidence Threshold:** 0.5

---

## Executive Summary

The multi-exit YOLOS model shows a clear **performance hierarchy** across the three exit layers, with Layer 12 (final exit) significantly outperforming the early exits. However, the **early exit layers (8 and 10) are underperforming**, suggesting potential issues with training or architectural decisions.

### Key Findings:
- ✅ **Layer 12 performs best**: 21.0% mAP, 34.9% AP50, 31.2% F1
- ⚠️ **Layer 8 and 10 very weak**: <5% mAP, indicating poor detection capability
- 📊 **Clear progression**: Layer 8 < Layer 10 < Layer 12 (as expected)
- 🎯 **Room for improvement**: All layers below target performance

---

## Detailed Results

### Performance Metrics

| Layer | mAP ↑ | AP50 ↑ | AP75 ↑ | Precision ↑ | Recall ↑ | F1 Score ↑ |
|-------|-------|--------|--------|-------------|----------|------------|
| **8 (Early)** | 1.34% | 3.06% | 0.83% | 4.16% | 14.68% | 6.49% |
| **10 (Middle)** | 4.24% | 9.81% | 1.59% | 4.93% | 16.56% | 7.60% |
| **12 (Final)** | **21.04%** | **34.87%** | **22.57%** | **24.22%** | **43.86%** | **31.21%** |

### Performance Gaps

**Layer 8 → Layer 10 Improvements:**
- mAP: +2.90 pp (217% relative improvement)
- AP50: +6.75 pp (220% improvement)
- F1: +1.11 pp (17% improvement)

**Layer 10 → Layer 12 Improvements:**
- mAP: +16.80 pp (396% relative improvement)
- AP50: +24.86 pp (253% improvement)
- F1: +23.61 pp (311% improvement)

**Overall (Layer 8 → Layer 12):**
- mAP: **15.7x better**
- AP50: **11.4x better**
- F1 Score: **4.8x better**

---

## Analysis by Metric

### 1. Mean Average Precision (mAP)

mAP measures detection quality across multiple IoU thresholds (0.5 to 0.95).

```
Layer 8:   1.34%  ████
Layer 10:  4.24%  ████████████
Layer 12: 21.04%  ████████████████████████████████████████████████████████
```

**Interpretation:**
- Layer 12 achieves reasonable performance (21% mAP is decent for pedestrian detection)
- Layers 8 and 10 are essentially **non-functional** for practical use (<5% mAP)
- The 15.7x gap between Layer 8 and Layer 12 is concerning

**Comparison to Targets:**
- Target Layer 8: >25% mAP → **Actual: 1.34% ❌ (94.6% below target)**
- Target Layer 10: >35% mAP → **Actual: 4.24% ❌ (87.9% below target)**
- Target Layer 12: >45% mAP → **Actual: 21.04% ❌ (53.2% below target)**

### 2. AP50 (Average Precision at IoU=0.5)

AP50 measures detection quality at a looser IoU threshold (50% overlap).

```
Layer 8:   3.06%  ████
Layer 10:  9.81%  ████████████
Layer 12: 34.87%  ████████████████████████████████████████████
```

**Interpretation:**
- Layer 12: 34.87% AP50 is acceptable for a challenging dataset
- Layers 8/10: Even with relaxed IoU, performance remains very poor
- Shows that early exits struggle with **localization**, not just classification

### 3. Precision vs Recall Trade-off

| Layer | Precision | Recall | Balance |
|-------|-----------|--------|---------|
| 8 | 4.16% | 14.68% | **3.5x more recall than precision** |
| 10 | 4.93% | 16.56% | **3.4x more recall than precision** |
| 12 | 24.22% | 43.86% | **1.8x more recall than precision** |

**Interpretation:**
- All layers have **low precision, higher recall** (many false positives)
- This suggests the model is **over-predicting** (confidence threshold may be too low)
- Layer 12 has better balance, but still biased toward recall

**Precision Analysis:**
- Layer 8: Only 4.16% of predictions are correct → **95.84% false positives!**
- Layer 10: Only 4.93% correct → **95.07% false positives!**
- Layer 12: 24.22% correct → **75.78% false positives**

**Recall Analysis:**
- Layer 8: Detects 14.68% of ground truth pedestrians → **85.32% missed**
- Layer 10: Detects 16.56% → **83.44% missed**
- Layer 12: Detects 43.86% → **56.14% missed**

### 4. F1 Score (Harmonic Mean of Precision & Recall)

```
Layer 8:   6.49%
Layer 10:  7.60%
Layer 12: 31.21%
```

**Interpretation:**
- F1 score represents overall detection capability
- Layer 12 at 31.21% F1 is reasonable for this task
- Layers 8 and 10 at <8% F1 are essentially **unusable**

---

## Root Cause Analysis

### Why Are Layers 8 and 10 So Weak?

#### 1. **Insufficient Feature Richness**
- Layer 8 is only 67% through the 12-layer backbone
- At this depth, features may not be abstract enough for robust pedestrian detection
- **Evidence**: Loss curves showed Layer 8/10 still decreasing at epoch 15

#### 2. **Training Strategy Issue**
- We froze the backbone and trained only detection heads
- This means Layers 8 and 10 must work with features that were **never optimized for early detection**
- The backbone was pretrained for Layer 12 outputs only

#### 3. **Detection Head Capacity**
- Each head is a simple 3-layer MLP (384→384→384→2 for classification)
- This may be **too shallow** to compensate for weaker intermediate features
- Layer 12 benefits from pretrained bbox head (frozen), which Layers 8/10 lack

#### 4. **Confidence Calibration**
- Confidence threshold of 0.5 may be too high for Layers 8/10
- Early exits might need lower thresholds to be useful
- **Test**: Rerun evaluation with conf_threshold=0.1 or 0.2

#### 5. **Class Imbalance**
- MOT17 has many empty regions and crowded scenes
- Early layers may struggle to distinguish pedestrians from background
- Layer 12 has deeper context to resolve ambiguities

---

## Comparison to Expectations

### Pre-Training Expectations:
We expected a gradual performance degradation from Layer 12 → 8, where:
- Layer 12: Best performance (baseline)
- Layer 10: ~15-20% worse than Layer 12
- Layer 8: ~30-40% worse than Layer 12

### Actual Results:
- Layer 12: 21.04% mAP (baseline)
- Layer 10: 79.8% **worse** than Layer 12 (not 15-20%)
- Layer 8: 93.6% **worse** than Layer 12 (not 30-40%)

**The gap is MUCH larger than expected**, suggesting fundamental issues rather than expected degradation.

---

## Recommendations

### Immediate Actions:

#### 1. **Test with Lower Confidence Threshold** 🔧
Run evaluation again with `--conf_threshold 0.1` to see if Layers 8/10 can produce useful predictions:

```bash
CUDA_VISIBLE_DEVICES=1 python evaluation/evaluate.py \
  --checkpoint results/multi_exit_training/20251020_165209/best_model.pt \
  --config configs/multi_exit_config.yaml \
  --conf_threshold 0.1
```

This will reveal if the issue is:
- **Calibration problem** (fixed by lower threshold) → predictions exist but filtered out
- **Fundamental detection failure** (no change) → heads aren't learning

#### 2. **Visualize Predictions** 📊
Create sample visualizations to see:
- What are Layers 8/10 actually predicting?
- Are boxes wildly off-target?
- Are confidence scores uniformly low?

#### 3. **Check Feature Quality** 🔍
Extract features from Layers 8, 10, 12 and visualize with t-SNE:
- Are Layer 8/10 features discriminative?
- Is there clear separation between pedestrian/background?

### Medium-Term Fixes:

#### 4. **Phase 2 Training with Unfrozen Backbone** 🎯
The main issue is that Layers 8/10 work with features optimized only for Layer 12.

**Solution**: Run Phase 2 training:
- Unfreeze entire model
- Use very small learning rate (1e-6)
- This will adapt intermediate layers to support early detection

**Expected improvement**: 3-5x better mAP for Layers 8/10

#### 5. **Deeper Detection Heads** 🧠
Current heads: 3-layer MLP (384→384→384→classes)

**Proposal**: 5-layer MLP with larger hidden dims:
- 384 → 512 → 512 → 384 → classes
- Add skip connections or residual blocks
- More capacity to compensate for weaker features

#### 6. **Multi-Scale Feature Fusion** 🔀
Instead of using raw Layer 8/10 outputs, fuse with features from adjacent layers:
- Layer 8 head: Fuse features from Layers 7, 8, 9
- Layer 10 head: Fuse features from Layers 9, 10, 11
- Provides richer context for early detection

### Long-Term Improvements:

#### 7. **Progressive Training Strategy**
Train in stages:
1. **Stage 1**: Train Layer 12 only (current approach)
2. **Stage 2**: Add Layer 10, fine-tune with both
3. **Stage 3**: Add Layer 8, fine-tune all three
4. **Stage 4**: Unfreeze backbone, end-to-end fine-tune

This ensures each exit learns on features appropriate for its depth.

#### 8. **Auxiliary Loss Reweighting**
Current loss weights: Equal for all layers

**Proposal**: Use adaptive weights:
- Layer 8: 0.5x weight (acknowledge it's harder)
- Layer 10: 0.75x weight
- Layer 12: 1.0x weight (reference)

Prevents Layer 8/10 from dominating gradient updates with large errors.

#### 9. **Knowledge Distillation**
Use Layer 12 predictions as soft targets for Layers 8/10:
- Layer 12 generates "teacher" predictions
- Layers 8/10 learn to mimic Layer 12
- Helps transfer knowledge from deeper to shallower layers

---

## Computational Trade-offs

### Current Performance:

| Layer | mAP | Relative Speed | Practical Use |
|-------|-----|----------------|---------------|
| 8 | 1.34% | ~3x faster | ❌ Too inaccurate |
| 10 | 4.24% | ~2x faster | ❌ Too inaccurate |
| 12 | 21.04% | 1x (baseline) | ✅ Usable |

**Conclusion**: Currently, there's **no computational benefit** because Layers 8/10 aren't accurate enough for adaptive inference. You'd always need to fall back to Layer 12.

### Target Performance for Adaptive Inference:

For adaptive inference to be worthwhile:
- **Layer 8**: Need >15% mAP (10x improvement) for "easy" frames
- **Layer 10**: Need >20% mAP (5x improvement) for "medium" frames
- **Layer 12**: Current 21% mAP is acceptable for "hard" frames

**Estimated computational savings** if targets met:
- 40% of frames exit at Layer 8 → 3x speedup
- 30% of frames exit at Layer 10 → 2x speedup
- 30% of frames use Layer 12 → 1x (baseline)
- **Overall**: ~2.1x faster with minimal accuracy loss

---

## Positive Findings

Despite the disappointing absolute numbers, there are encouraging signs:

### ✅ 1. Clear Layer Hierarchy
The performance progression (8 < 10 < 12) is consistent across ALL metrics:
- Shows the architecture is fundamentally sound
- Confirms deeper layers have access to better features
- Validates the multi-exit design principle

### ✅ 2. Fast Evaluation
After optimization, evaluation completes in <1 minute:
- Enables rapid iteration and experimentation
- Can quickly test different confidence thresholds
- Facilitates hyperparameter tuning

### ✅ 3. Layer 12 Reasonable Performance
21% mAP is decent for:
- Single-class pedestrian detection
- Challenging MOT17 dataset (crowded scenes, occlusions)
- Limited training (15 epochs, frozen backbone)

**Comparable models:**
- YOLOS-small on COCO: ~29% mAP (91 classes)
- Ours on MOT17: ~21% mAP (1 class)
- Considering class simplification, performance is in the ballpark

### ✅ 4. Training Infrastructure Solid
- Training completes without errors
- Losses decrease smoothly
- Validation metrics properly computed
- Checkpointing and logging work correctly

---

## Next Steps Prioritization

### 🔥 Priority 1: Quick Wins (Do Now)

1. **Rerun evaluation with conf_threshold=0.1**
   - Takes <1 minute
   - Will reveal if calibration is the issue
   - May unlock hidden performance in Layers 8/10

2. **Visualize predictions on sample images**
   - Understand what the model is actually doing wrong
   - Identify patterns in failures
   - Inform next optimization steps

### 🎯 Priority 2: Training Improvements (This Week)

3. **Run Phase 2 training (unfreeze backbone)**
   - Most likely to significantly improve Layers 8/10
   - Adapts intermediate features for early detection
   - Expected: 3-5x mAP improvement

4. **Train for more epochs (30-50 instead of 15)**
   - Loss curves hadn't plateaued at epoch 15
   - May need longer training for convergence
   - Particularly important for Layers 8/10

### 📊 Priority 3: Architecture Changes (Next Sprint)

5. **Implement deeper detection heads**
6. **Add multi-scale feature fusion**
7. **Try knowledge distillation**

---

## Conclusion

The multi-exit YOLOS model shows **proof of concept** but requires significant improvement before practical deployment:

**Current State:**
- ✅ Layer 12 works reasonably well (21% mAP)
- ❌ Layers 8 and 10 are too weak (<5% mAP)
- ❌ No computational benefit yet (can't use early exits)

**Path Forward:**
- Test lower confidence thresholds (quick win)
- Unfreeze backbone for Phase 2 training (main fix)
- Consider architectural improvements (long-term)

**Expected Outcome After Fixes:**
- Layer 8: 15-20% mAP (usable for easy frames)
- Layer 10: 25-30% mAP (usable for medium frames)
- Layer 12: 30-35% mAP (improved final fallback)
- **Overall system: 2x speedup with <10% accuracy loss**

The foundation is solid, but the model needs targeted improvements to realize the adaptive inference benefits.

---

**Generated:** October 21, 2025
**Evaluation Results:** `results/multi_exit_training/20251020_165209/evaluation_results.json`
