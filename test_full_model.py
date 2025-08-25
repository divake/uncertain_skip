"""
Simple test script for full YOLOS model without layer skipping
"""

import torch
import numpy as np
from transformers import YolosForObjectDetection, YolosImageProcessor
from PIL import Image
import time
import warnings
warnings.filterwarnings('ignore')

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model and processor
print("\nLoading YOLOS model...")
model_name = "hustvl/yolos-tiny"
model = YolosForObjectDetection.from_pretrained(model_name)
image_processor = YolosImageProcessor.from_pretrained(model_name)
model.to(device)
model.eval()

print(f"Model loaded: {model_name}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# Load a sample image from BDD100K
import os
from pathlib import Path

bdd_path = Path("/mnt/ssd1/divake/uncertain_skip/bdd100k/100k/val")
image_files = list(bdd_path.glob("*.jpg"))[:5]  # Get first 5 images

if not image_files:
    print("No images found!")
    exit(1)

print(f"\nFound {len(image_files)} test images")

# Test inference on each image
print("\n" + "="*60)
print("Testing Full Model (All 12 Layers)")
print("="*60)

inference_times = []

for idx, image_path in enumerate(image_files):
    print(f"\nImage {idx+1}: {image_path.name}")
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    # Warmup
    if idx == 0:
        print("  Warming up...")
        for _ in range(3):
            with torch.no_grad():
                _ = model(pixel_values=pixel_values)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
    
    # Measure inference time
    times = []
    num_runs = 10
    
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            outputs = model(pixel_values=pixel_values)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    inference_times.append(avg_time)
    
    # Post-process to get detections
    target_sizes = torch.tensor([[512, 512]]).to(device)
    results = image_processor.post_process_object_detection(
        outputs, 
        threshold=0.3,
        target_sizes=target_sizes
    )[0]
    
    num_detections = len(results['boxes'])
    
    print(f"  Inference time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  FPS: {1000/avg_time:.2f}")
    print(f"  Detections: {num_detections}")
    
    # Show some detections
    if num_detections > 0:
        print(f"  Sample scores: {results['scores'][:3].cpu().numpy()}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Average inference time: {np.mean(inference_times):.2f} ± {np.std(inference_times):.2f} ms")
print(f"Average FPS: {1000/np.mean(inference_times):.2f}")
print(f"Model: {model_name}")
print(f"Device: {device}")
print(f"Number of test images: {len(image_files)}")

print("\nTest completed successfully!")