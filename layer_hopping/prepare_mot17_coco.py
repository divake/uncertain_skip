"""
Convert MOT17 dataset to COCO format for official YOLOS training

MOT17 Format:
  - Images: train/MOT17-XX-FRCNN/img1/*.jpg
  - GT: train/MOT17-XX-FRCNN/gt/gt.txt
    Format: frame,id,x,y,w,h,conf,class,visibility

COCO Format:
  {
    "images": [{"id": int, "file_name": str, "height": int, "width": int}, ...],
    "annotations": [{"id": int, "image_id": int, "category_id": int, "bbox": [x,y,w,h], "area": float, "iscrowd": 0}, ...],
    "categories": [{"id": 1, "name": "person"}]
  }
"""

import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse


def convert_mot17_to_coco(mot17_root, sequences, output_file, split_name):
    """
    Convert MOT17 sequences to COCO format

    Args:
        mot17_root: Path to MOT17/train directory
        sequences: List of sequence names (e.g., ['MOT17-02-FRCNN', ...])
        output_file: Output JSON file path
        split_name: 'train' or 'val'
    """

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}]
    }

    image_id = 1
    annotation_id = 1

    print(f"\nConverting {split_name} sequences to COCO format...")
    print(f"Sequences: {sequences}")
    print()

    for seq_name in sequences:
        seq_path = Path(mot17_root) / seq_name
        img_dir = seq_path / "img1"
        gt_file = seq_path / "gt" / "gt.txt"
        seqinfo_file = seq_path / "seqinfo.ini"

        print(f"Processing {seq_name}...")

        # Read sequence info
        seqinfo = {}
        with open(seqinfo_file, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=')
                    seqinfo[key] = value

        img_width = int(seqinfo['imWidth'])
        img_height = int(seqinfo['imHeight'])
        seq_length = int(seqinfo['seqLength'])

        # Read ground truth annotations
        # Format: frame,id,x,y,w,h,conf,class,visibility
        gt_data = {}
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame = int(parts[0])
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6])
                cls = int(parts[7])
                visibility = float(parts[8])

                # Filter: only person class (cls=1), confidence > 0, visibility > 0
                if cls == 1 and conf > 0 and visibility > 0:
                    if frame not in gt_data:
                        gt_data[frame] = []
                    gt_data[frame].append({
                        'bbox': [x, y, w, h],
                        'area': w * h,
                    })

        # Process each frame
        for frame_num in range(1, seq_length + 1):
            # Image info
            img_filename = f"{frame_num:06d}.jpg"
            img_path = img_dir / img_filename

            if not img_path.exists():
                continue

            # Add image entry
            coco_data["images"].append({
                "id": image_id,
                "file_name": f"{seq_name}/{img_filename}",  # Include sequence name
                "height": img_height,
                "width": img_width,
                "seq_name": seq_name,
                "frame_num": frame_num,
            })

            # Add annotations for this frame
            if frame_num in gt_data:
                for bbox_data in gt_data[frame_num]:
                    x, y, w, h = bbox_data['bbox']

                    # COCO format: bbox is [x_min, y_min, width, height]
                    # MOT17 already uses this format (top-left corner + w,h)
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,  # Person
                        "bbox": [x, y, w, h],
                        "area": bbox_data['area'],
                        "iscrowd": 0,
                    })
                    annotation_id += 1

            image_id += 1

        print(f"  ✓ {seq_name}: {len([img for img in coco_data['images'] if img.get('seq_name') == seq_name])} images")

    # Save to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(coco_data, f)

    print(f"\n✓ Saved to {output_file}")
    print(f"  Total images: {len(coco_data['images'])}")
    print(f"  Total annotations: {len(coco_data['annotations'])}")
    print(f"  Avg annotations per image: {len(coco_data['annotations']) / len(coco_data['images']):.2f}")
    print()

    return coco_data


def main():
    parser = argparse.ArgumentParser('Convert MOT17 to COCO format')
    parser.add_argument('--mot17_root', default='/ssd_4TB/divake/uncertain_skip/data/MOT17/train',
                        help='Path to MOT17/train directory')
    parser.add_argument('--output_dir', default='data/mot17_coco',
                        help='Output directory for COCO JSON files')
    args = parser.parse_args()

    # Define train/val splits
    train_sequences = [
        "MOT17-02-FRCNN",
        "MOT17-04-FRCNN",
        "MOT17-05-FRCNN",
        "MOT17-09-FRCNN",
        "MOT17-10-FRCNN"
    ]

    val_sequences = [
        "MOT17-11-FRCNN",
        "MOT17-13-FRCNN"
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert train set
    train_file = output_dir / "train_coco.json"
    train_data = convert_mot17_to_coco(
        args.mot17_root,
        train_sequences,
        train_file,
        'train'
    )

    # Convert val set
    val_file = output_dir / "val_coco.json"
    val_data = convert_mot17_to_coco(
        args.mot17_root,
        val_sequences,
        val_file,
        'val'
    )

    print("="*80)
    print("CONVERSION COMPLETE")
    print("="*80)
    print(f"Train: {len(train_data['images'])} images, {len(train_data['annotations'])} boxes")
    print(f"Val:   {len(val_data['images'])} images, {len(val_data['annotations'])} boxes")
    print()
    print(f"Files saved to: {output_dir}")
    print()


if __name__ == '__main__':
    main()
