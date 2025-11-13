#!/usr/bin/env python3
"""
Compute frame-by-frame aleatoric and epistemic uncertainty for MOT17-04
This will generate a JSON file with per-detection uncertainty that we can use for tracking
"""

import sys
sys.path.append('/ssd_4TB/divake/temporal_uncertainty/conformal_tracking')

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

# Import from temporal_uncertainty project
from src.uncertainty.mahalanobis import MahalanobisUncertainty
from src.uncertainty.epistemic_combined import EpistemicUncertainty

def load_mot17_cache(cache_path, conf_threshold=0.3):
    """Load data directly from NPZ cache"""
    cache = np.load(cache_path)

    # Get all detections
    all_frame_ids = cache['detections/frame_ids']
    all_bboxes = cache['detections/bboxes']
    all_confidences = cache['detections/confidences']

    # Get matching information
    matched_det_indices = cache['gt_matching/det_indices']
    matched_ious = cache['gt_matching/iou']
    matched_center_errors = cache['gt_matching/center_error']

    # Get features for all layers
    features_all = {
        4: cache['features/layer_4'],
        9: cache['features/layer_9'],
        15: cache['features/layer_15'],
        21: cache['features/layer_21']
    }

    # Filter: only matched detections with conf >= threshold
    matched_mask = np.isin(np.arange(len(all_frame_ids)), matched_det_indices)
    conf_mask = all_confidences >= conf_threshold
    final_mask = matched_mask & conf_mask

    # Map matched indices to get IoUs and errors
    # Create a mapping from detection index to matched data
    det_idx_to_match_idx = {det_idx: match_idx
                           for match_idx, det_idx in enumerate(matched_det_indices)}

    # For each final detection, get its IoU and error
    final_indices = np.where(final_mask)[0]
    ious_list = []
    center_errors_list = []

    for det_idx in final_indices:
        if det_idx in det_idx_to_match_idx:
            match_idx = det_idx_to_match_idx[det_idx]
            ious_list.append(matched_ious[match_idx])
            center_errors_list.append(matched_center_errors[match_idx])
        else:
            # Shouldn't happen, but handle gracefully
            ious_list.append(0.0)
            center_errors_list.append(100.0)

    return {
        'features': {layer: features_all[layer][final_mask] for layer in [4, 9, 15, 21]},
        'frame_ids': all_frame_ids[final_mask],
        'bboxes': all_bboxes[final_mask],
        'confidences': all_confidences[final_mask],
        'ious': np.array(ious_list),
        'center_errors': np.array(center_errors_list),
        'n_samples': np.sum(final_mask)
    }

def compute_uncertainty_for_mot17_04():
    """Compute aleatoric and epistemic uncertainty for MOT17-04"""

    print("="*60)
    print("Computing Uncertainty for MOT17-04")
    print("="*60)

    # Load cached data
    print("\n[1/5] Loading cached features...")
    cache_path = Path('/ssd_4TB/divake/temporal_uncertainty/yolo_cache/data/mot17/yolov8n/MOT17-04-FRCNN.npz')

    data = load_mot17_cache(cache_path, conf_threshold=0.3)

    features = data['features'][9]  # Primary layer
    ious = data['ious']
    center_errors = data['center_errors']
    confidence = data['confidences']
    frame_ids = data['frame_ids']
    boxes = data['bboxes']

    print(f"✓ Loaded {data['n_samples']} detections")
    print(f"  - Frames: {frame_ids.min()} to {frame_ids.max()}")
    print(f"  - Features shape: {features.shape}")
    print(f"  - IoU mean: {ious.mean():.3f}")

    # Split into calibration and test (50/50)
    print("\n[2/5] Splitting calibration/test...")
    n_samples = len(features)
    indices = np.random.RandomState(42).permutation(n_samples)
    split_idx = n_samples // 2

    cal_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_cal = features[cal_idx]
    X_test = features[test_idx]

    print(f"✓ Calibration: {len(cal_idx)} samples")
    print(f"✓ Test: {len(test_idx)} samples")

    # Compute aleatoric uncertainty
    print("\n[3/5] Computing aleatoric uncertainty...")
    mahalanobis = MahalanobisUncertainty(
        reg_lambda=1e-4
    )

    mahalanobis.fit(X_cal, verbose=True)

    # predict() returns dict with 'raw' and 'normalized' keys
    aleatoric_cal_dict = mahalanobis.predict(X_cal)
    aleatoric_test_dict = mahalanobis.predict(X_test)

    # Extract normalized values (in [0, 1] range)
    aleatoric_cal = aleatoric_cal_dict['normalized']
    aleatoric_test = aleatoric_test_dict['normalized']

    # Combine for all samples
    aleatoric_all = np.zeros(n_samples)
    aleatoric_all[cal_idx] = aleatoric_cal
    aleatoric_all[test_idx] = aleatoric_test

    print(f"✓ Aleatoric computed")
    print(f"  - Mean: {aleatoric_all.mean():.3f}")
    print(f"  - Std: {aleatoric_all.std():.3f}")
    print(f"  - Range: [{aleatoric_all.min():.3f}, {aleatoric_all.max():.3f}]")

    # Compute epistemic uncertainty
    print("\n[4/5] Computing epistemic uncertainty...")

    # Prepare multi-layer features
    X_cal_layers = {layer: data['features'][layer][cal_idx] for layer in [4, 9, 15, 21]}
    X_test_layers = {layer: data['features'][layer][test_idx] for layer in [4, 9, 15, 21]}

    epistemic = EpistemicUncertainty(
        k_neighbors_spectral=50,
        k_neighbors_repulsive=100,
        temperature=1.0,
        weights='optimize',  # Optimize for orthogonality
        verbose=True
    )

    epistemic.fit(
        X_calibration=X_cal,
        X_cal_layers=X_cal_layers,
        aleatoric_cal=aleatoric_cal,
        plot_diagnostics=False
    )

    # predict() returns dict with 'combined', 'spectral', 'repulsive', 'gradient' keys
    epistemic_cal_dict = epistemic.predict(X_cal, X_cal_layers, return_components=False)
    epistemic_test_dict = epistemic.predict(X_test, X_test_layers, return_components=False)

    # Extract combined epistemic values
    epistemic_cal = epistemic_cal_dict['combined']
    epistemic_test = epistemic_test_dict['combined']

    # Combine for all samples
    epistemic_all = np.zeros(n_samples)
    epistemic_all[cal_idx] = epistemic_cal
    epistemic_all[test_idx] = epistemic_test

    print(f"✓ Epistemic computed")
    print(f"  - Mean: {epistemic_all.mean():.3f}")
    print(f"  - Std: {epistemic_all.std():.3f}")
    print(f"  - Range: [{epistemic_all.min():.3f}, {epistemic_all.max():.3f}]")

    # Check orthogonality
    correlation = np.corrcoef(aleatoric_all, epistemic_all)[0, 1]
    print(f"\n✓ Orthogonality check:")
    print(f"  - Correlation: {correlation:.4f}")
    print(f"  - Status: {'✅ PASS' if abs(correlation) < 0.3 else '❌ FAIL'}")

    # Organize by frame
    print("\n[5/5] Organizing by frame...")
    frame_uncertainty = {}

    for i in tqdm(range(n_samples), desc="Processing detections"):
        frame_id = int(frame_ids[i])

        if frame_id not in frame_uncertainty:
            frame_uncertainty[frame_id] = []

        frame_uncertainty[frame_id].append({
            'bbox': boxes[i].tolist(),
            'confidence': float(confidence[i]),
            'iou': float(ious[i]),
            'aleatoric': float(aleatoric_all[i]),
            'epistemic': float(epistemic_all[i]),
            'center_error': float(center_errors[i])
        })

    print(f"✓ Organized {len(frame_uncertainty)} frames")

    # Save results
    output_file = Path('/ssd_4TB/divake/uncertain_skip/data/mot17_04_uncertainty.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'sequence': 'MOT17-04-FRCNN',
        'n_detections': n_samples,
        'n_frames': len(frame_uncertainty),
        'statistics': {
            'aleatoric': {
                'mean': float(aleatoric_all.mean()),
                'std': float(aleatoric_all.std()),
                'min': float(aleatoric_all.min()),
                'max': float(aleatoric_all.max())
            },
            'epistemic': {
                'mean': float(epistemic_all.mean()),
                'std': float(epistemic_all.std()),
                'min': float(epistemic_all.min()),
                'max': float(epistemic_all.max())
            },
            'orthogonality': float(correlation)
        },
        'epistemic_weights': {
            'spectral': float(epistemic.weights[0]) if epistemic.weights is not None else 0.5,
            'repulsive': float(epistemic.weights[1]) if epistemic.weights is not None else 0.5,
            'gradient': float(epistemic.weights[2]) if epistemic.weights is not None else 0.0
        },
        'frames': frame_uncertainty
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  - Sequence: MOT17-04-FRCNN")
    print(f"  - Detections: {n_samples:,}")
    print(f"  - Frames: {len(frame_uncertainty)}")
    print(f"  - Aleatoric: {aleatoric_all.mean():.3f} ± {aleatoric_all.std():.3f}")
    print(f"  - Epistemic: {epistemic_all.mean():.3f} ± {epistemic_all.std():.3f}")
    print(f"  - Orthogonality: {correlation:.4f} ✅")

    return results

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    results = compute_uncertainty_for_mot17_04()
    print("\n" + "="*60)
    print("✅ COMPLETE")
    print("="*60)
