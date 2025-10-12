"""
MOT17 Data Loader - FIXED VERSION - WILL NEVER CHANGE

This is the final, production-ready data loader for MOT17 dataset.
It has been tested and verified to work correctly.

DO NOT MODIFY this file. If you need different functionality,
create a new file.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from .architecture_constants import (
        INPUT_SIZE,
        IMAGE_MEAN,
        IMAGE_STD,
        PERSON_CLASS_ID,
        DATASET_ROOT,
        TRAIN_SEQUENCES,
        VAL_SEQUENCES
    )
except ImportError:
    # For standalone execution
    from architecture_constants import (
        INPUT_SIZE,
        IMAGE_MEAN,
        IMAGE_STD,
        PERSON_CLASS_ID,
        DATASET_ROOT,
        TRAIN_SEQUENCES,
        VAL_SEQUENCES
    )


class MOT17DetectionDataset(Dataset):
    """
    MOT17 Dataset for Object Detection

    This dataset loader is FIXED and VERIFIED. It handles:
    - Loading images from MOT17 sequences
    - Parsing ground truth annotations
    - Normalizing bounding boxes to [0, 1]
    - Filtering person class only
    - Proper image transformations

    The output format is compatible with YOLOS model.
    """

    def __init__(
        self,
        root_path: str = DATASET_ROOT,
        sequences: List[str] = None,
        transform: Optional[transforms.Compose] = None,
        image_size: int = INPUT_SIZE
    ):
        """
        Args:
            root_path: Path to MOT17/train directory
            sequences: List of sequence names (e.g., ['MOT17-02-FRCNN'])
                      If None, uses TRAIN_SEQUENCES
            transform: Optional transform (if None, uses default)
            image_size: Target image size (default: INPUT_SIZE = 512)
        """
        self.root_path = Path(root_path)
        self.sequences = sequences if sequences is not None else TRAIN_SEQUENCES
        self.image_size = image_size

        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform

        # Load all samples
        self.samples = self._load_all_samples()

        print(f"[MOT17Dataset] Loaded {len(self.samples)} samples from {len(self.sequences)} sequences")

    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transform (resize + normalize)"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ])

    def _load_all_samples(self) -> List[Dict]:
        """Load all frame paths and annotations from all sequences"""
        all_samples = []

        for seq_name in self.sequences:
            seq_samples = self._load_sequence(seq_name)
            all_samples.extend(seq_samples)

        return all_samples

    def _load_sequence(self, seq_name: str) -> List[Dict]:
        """Load samples from a single sequence"""
        seq_path = self.root_path / seq_name
        img_dir = seq_path / "img1"
        gt_file = seq_path / "gt" / "gt.txt"

        if not gt_file.exists():
            print(f"[WARNING] Ground truth not found: {gt_file}")
            return []

        if not img_dir.exists():
            print(f"[WARNING] Image directory not found: {img_dir}")
            return []

        # Load ground truth file
        # Format: frame, id, x, y, w, h, conf, class, visibility
        gt_data = np.loadtxt(gt_file, delimiter=',')

        # Group annotations by frame
        frame_annotations = {}
        for row in gt_data:
            frame_id = int(row[0])
            if frame_id not in frame_annotations:
                frame_annotations[frame_id] = []

            # Extract bounding box
            bbox = {
                'x': float(row[2]),      # Top-left x
                'y': float(row[3]),      # Top-left y
                'w': float(row[4]),      # Width
                'h': float(row[5]),      # Height
                'conf': float(row[6]) if len(row) > 6 else 1.0,
                'class': int(row[7]) if len(row) > 7 else 1,  # 1 = person in MOT
                'visibility': float(row[8]) if len(row) > 8 else 1.0
            }

            # Only keep person class (class == 1 in MOT format)
            if bbox['class'] == 1:
                frame_annotations[frame_id].append(bbox)

        # Create samples for each frame
        samples = []
        for frame_id, boxes in frame_annotations.items():
            img_path = img_dir / f"{frame_id:06d}.jpg"

            if img_path.exists():
                samples.append({
                    'image_path': str(img_path),
                    'sequence': seq_name,
                    'frame_id': frame_id,
                    'boxes': boxes
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get a single sample

        Returns:
            image: Tensor [3, H, W], normalized
            target: Dictionary containing:
                - boxes: Tensor [N, 4] in format [cx, cy, w, h], normalized to [0, 1]
                - labels: Tensor [N] with class IDs (all 0 for person)
                - image_id: Tensor [1] with sample index
                - orig_size: Tensor [2] with [height, width] of original image
                - sequence: str, sequence name
                - frame_id: int, frame number
        """
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        orig_width, orig_height = image.size

        # Convert boxes to YOLOS format: [cx, cy, w, h] normalized to [0, 1]
        boxes = []
        for box in sample['boxes']:
            # Convert from [x, y, w, h] (top-left) to [cx, cy, w, h] (center)
            cx = (box['x'] + box['w'] / 2) / orig_width
            cy = (box['y'] + box['h'] / 2) / orig_height
            w = box['w'] / orig_width
            h = box['h'] / orig_height

            # Clip to [0, 1] range
            cx = np.clip(cx, 0.0, 1.0)
            cy = np.clip(cy, 0.0, 1.0)
            w = np.clip(w, 0.0, 1.0)
            h = np.clip(h, 0.0, 1.0)

            # Only keep valid boxes (non-zero area)
            if w > 0.001 and h > 0.001:
                boxes.append([cx, cy, w, h])

        # Create target dictionary
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros(len(boxes), dtype=torch.long),  # All person (class 0)
            'image_id': torch.tensor([idx], dtype=torch.long),
            'orig_size': torch.tensor([orig_height, orig_width], dtype=torch.long),
            'sequence': sample['sequence'],
            'frame_id': sample['frame_id']
        }

        # Apply transform to image
        if self.transform:
            image = self.transform(image)

        return image, target


def get_train_dataset() -> MOT17DetectionDataset:
    """Get training dataset with default settings"""
    return MOT17DetectionDataset(
        root_path=DATASET_ROOT,
        sequences=TRAIN_SEQUENCES,
        image_size=INPUT_SIZE
    )


def get_val_dataset() -> MOT17DetectionDataset:
    """Get validation dataset with default settings"""
    return MOT17DetectionDataset(
        root_path=DATASET_ROOT,
        sequences=VAL_SEQUENCES,
        image_size=INPUT_SIZE
    )


def get_sample_dataset(sequences: List[str] = None, num_samples: int = 10) -> Dataset:
    """
    Get a small subset of data for experiments

    Args:
        sequences: List of sequences to sample from (default: first 2 val sequences)
        num_samples: Number of samples to return

    Returns:
        Subset of MOT17DetectionDataset
    """
    if sequences is None:
        sequences = VAL_SEQUENCES[:2]  # Use first 2 validation sequences

    dataset = MOT17DetectionDataset(
        root_path=DATASET_ROOT,
        sequences=sequences,
        image_size=INPUT_SIZE
    )

    # Return evenly spaced subset
    if num_samples >= len(dataset):
        return dataset

    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    return torch.utils.data.Subset(dataset, indices)


# ==============================================================================
# VERIFICATION FUNCTION
# ==============================================================================
def verify_data_loader():
    """
    Verify that data loader works correctly.
    Run this once to ensure everything is correct.
    """
    print("Verifying data loader...")

    # Test loading a few samples
    dataset = get_sample_dataset(num_samples=5)

    print(f"✓ Dataset created: {len(dataset)} samples")

    # Test getitem
    image, target = dataset[0]

    print(f"✓ Sample loaded:")
    print(f"  - Image shape: {image.shape}")
    print(f"  - Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  - Boxes shape: {target['boxes'].shape}")
    print(f"  - Labels shape: {target['labels'].shape}")
    print(f"  - Sequence: {target['sequence']}")
    print(f"  - Frame: {target['frame_id']}")

    # Verify image is normalized
    assert image.shape == (3, INPUT_SIZE, INPUT_SIZE), f"Wrong image shape: {image.shape}"
    assert image.min() >= -3 and image.max() <= 3, f"Image not normalized: [{image.min()}, {image.max()}]"

    # Verify boxes are in [0, 1]
    if target['boxes'].shape[0] > 0:
        assert target['boxes'].min() >= 0 and target['boxes'].max() <= 1, "Boxes not normalized"
        print(f"  - Box range: [{target['boxes'].min():.3f}, {target['boxes'].max():.3f}] ✓")

    print("\n✓ Data loader verification passed!")

    return True


if __name__ == "__main__":
    verify_data_loader()
