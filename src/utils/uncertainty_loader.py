"""
Uncertainty Loader for Pre-computed Aleatoric/Epistemic Uncertainty

Loads frame-by-frame uncertainty data computed from temporal_uncertainty project
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

class UncertaintyLoader:
    """Load and match pre-computed uncertainty to detections"""

    def __init__(self, uncertainty_file: str):
        """
        Initialize uncertainty loader

        Args:
            uncertainty_file: Path to JSON file with frame-by-frame uncertainty
        """
        self.uncertainty_file = Path(uncertainty_file)

        if not self.uncertainty_file.exists():
            raise FileNotFoundError(f"Uncertainty file not found: {uncertainty_file}")

        # Load uncertainty data
        with open(self.uncertainty_file, 'r') as f:
            self.data = json.load(f)

        # Parse frames dictionary
        self.frames = self.data['frames']

        # Store statistics
        self.stats = self.data['statistics']

        print(f"✓ Loaded uncertainty for {self.data['sequence']}")
        print(f"  - Frames: {self.data['n_frames']}")
        print(f"  - Detections: {self.data['n_detections']}")
        print(f"  - Aleatoric: {self.stats['aleatoric']['mean']:.3f} ± {self.stats['aleatoric']['std']:.3f}")
        print(f"  - Epistemic: {self.stats['epistemic']['mean']:.3f} ± {self.stats['epistemic']['std']:.3f}")
        print(f"  - Orthogonality: {self.stats['orthogonality']:.4f}")

    def get_uncertainty_for_detection(self, frame_id: int, bbox: np.ndarray, iou_threshold: float = 0.5) -> Tuple[float, float]:
        """
        Get uncertainty for a detection by matching bounding box

        Args:
            frame_id: Frame number
            bbox: Bounding box [x, y, w, h]
            iou_threshold: Minimum IoU for matching

        Returns:
            (aleatoric, epistemic) tuple
        """
        frame_key = str(frame_id)

        if frame_key not in self.frames:
            # Return mean uncertainty if frame not found
            return (self.stats['aleatoric']['mean'],
                    self.stats['epistemic']['mean'])

        frame_detections = self.frames[frame_key]

        # Find best matching detection
        best_match = None
        best_iou = 0

        for detection in frame_detections:
            det_bbox = np.array(detection['bbox'])
            iou = self._calculate_iou(bbox, det_bbox)

            if iou > best_iou and iou > iou_threshold:
                best_iou = iou
                best_match = detection

        if best_match is not None:
            return (best_match['aleatoric'], best_match['epistemic'])
        else:
            # No match found - return mean
            return (self.stats['aleatoric']['mean'],
                    self.stats['epistemic']['mean'])

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate IoU between two boxes in [x1, y1, x2, y2] or [x, y, w, h] format

        Args:
            box1: First bounding box
            box2: Second bounding box

        Returns:
            IoU value [0, 1]
        """
        # Convert [x, y, w, h] to [x1, y1, x2, y2] if needed
        if len(box1) == 4:
            if box1[2] < box1[0]:  # width format
                x1_1, y1_1, w1, h1 = box1
                x2_1, y2_1 = x1_1 + w1, y1_1 + h1
            else:  # already x1,y1,x2,y2
                x1_1, y1_1, x2_1, y2_1 = box1
        else:
            raise ValueError(f"Invalid box1 format: {box1}")

        if len(box2) == 4:
            if box2[2] < box2[0]:  # width format
                x1_2, y1_2, w2, h2 = box2
                x2_2, y2_2 = x1_2 + w2, y1_2 + h2
            else:  # already x1,y1,x2,y2
                x1_2, y1_2, x2_2, y2_2 = box2
        else:
            raise ValueError(f"Invalid box2 format: {box2}")

        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        if xi2 < xi1 or yi2 < yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def get_default_uncertainty(self) -> Tuple[float, float]:
        """
        Get default (mean) uncertainty values

        Returns:
            (aleatoric, epistemic) tuple with mean values
        """
        return (self.stats['aleatoric']['mean'],
                self.stats['epistemic']['mean'])
