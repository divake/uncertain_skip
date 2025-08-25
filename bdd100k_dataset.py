"""
BDD100K Dataset Loader for Object Detection
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from transformers import YolosImageProcessor


class BDD100KDataset(Dataset):
    """BDD100K dataset for object detection"""
    
    # BDD100K class names
    BDD_CLASSES = [
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle', 'traffic light', 'traffic sign'
    ]
    
    # Mapping from BDD100K classes to COCO classes (YOLOS uses COCO)
    BDD_TO_COCO = {
        'person': 0,  # person in COCO
        'rider': 0,   # Map rider to person
        'car': 2,     # car in COCO
        'truck': 7,   # truck in COCO
        'bus': 5,     # bus in COCO
        'train': 6,   # train in COCO
        'motorcycle': 3,  # motorcycle in COCO
        'bicycle': 1,     # bicycle in COCO
        'traffic light': 9,  # traffic light in COCO
        'traffic sign': -1,  # No direct mapping, skip
    }
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'val',
        image_processor: Optional[YolosImageProcessor] = None,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            root_dir: Path to BDD100K root directory
            split: 'train', 'val', or 'test'
            image_processor: YOLOS image processor for preprocessing
            max_samples: Maximum number of samples to load (for testing)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_processor = image_processor
        
        # Setup paths
        self.images_dir = self.root_dir / '100k' / split
        self.labels_dir = self.root_dir / '100k' / split
        
        # Load annotations
        self.annotations = self._load_annotations(max_samples)
        
    def _load_annotations(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Load and parse BDD100K annotations"""
        annotations = []
        
        # Get all JSON files in the labels directory
        json_files = sorted(list(self.labels_dir.glob('*.json')))
        
        if max_samples:
            json_files = json_files[:max_samples]
            
        for json_file in json_files:
            # Read annotation file
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # Extract image info and objects
            image_name = data['name'] + '.jpg'  # Add .jpg extension
            image_path = self.images_dir / image_name
            
            # Skip if image doesn't exist
            if not image_path.exists():
                continue
                
            # Parse objects from frames
            objects = []
            if 'frames' in data and len(data['frames']) > 0:
                frame = data['frames'][0]  # Use first frame
                if 'objects' in frame:
                    for obj in frame['objects']:
                        if 'category' in obj and 'box2d' in obj:
                            category = obj['category']
                            if category in self.BDD_TO_COCO:
                                coco_id = self.BDD_TO_COCO[category]
                                if coco_id >= 0:  # Skip unmapped classes
                                    box = obj['box2d']
                                    objects.append({
                                        'category': category,
                                        'coco_id': coco_id,
                                        'bbox': [box['x1'], box['y1'], box['x2'], box['y2']]
                                    })
                                
            if objects:  # Only include images with valid objects
                annotations.append({
                    'image_path': str(image_path),
                    'image_name': image_name,
                    'objects': objects
                })
                
        print(f"Loaded {len(annotations)} images with annotations from {self.split} split")
        return annotations
        
    def __len__(self) -> int:
        return len(self.annotations)
        
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample"""
        ann = self.annotations[idx]
        
        # Load image
        image = Image.open(ann['image_path']).convert('RGB')
        
        # Prepare targets for YOLOS
        boxes = []
        labels = []
        for obj in ann['objects']:
            boxes.append(obj['bbox'])
            labels.append(obj['coco_id'])
            
        # Convert to tensors
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'image_id': torch.tensor([idx])
        }
        
        # Process image if processor is provided
        if self.image_processor:
            # YOLOS expects normalized boxes in COCO format
            width, height = image.size
            
            # Convert to normalized coordinates
            boxes_norm = []
            for box in boxes:
                x1, y1, x2, y2 = box
                boxes_norm.append([
                    x1 / width,
                    y1 / height,
                    x2 / width,
                    y2 / height
                ])
                
            encoding = self.image_processor(
                images=image,
                return_tensors="pt"
            )
            
            # Remove batch dimension
            pixel_values = encoding['pixel_values'].squeeze(0)
            
            return {
                'pixel_values': pixel_values,
                'target': target,
                'original_image': np.array(image),
                'image_name': ann['image_name']
            }
        else:
            return {
                'image': image,
                'target': target,
                'image_name': ann['image_name']
            }


def create_bdd100k_dataloader(
    root_dir: str,
    split: str = 'val',
    batch_size: int = 1,
    image_processor: Optional[YolosImageProcessor] = None,
    max_samples: Optional[int] = None,
    num_workers: int = 4
) -> DataLoader:
    """Create a DataLoader for BDD100K dataset"""
    
    dataset = BDD100KDataset(
        root_dir=root_dir,
        split=split,
        image_processor=image_processor,
        max_samples=max_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x[0] if batch_size == 1 else x
    )
    
    return dataloader