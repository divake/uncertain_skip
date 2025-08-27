#!/usr/bin/env python3
"""
Quick debugging test to identify performance bottlenecks
"""

import torch
import time
import numpy as np
from ultralytics import YOLO

print("="*60)
print("QUICK PERFORMANCE BOTTLENECK TEST")
print("="*60)

# 1. GPU Check
print("\n1. GPU STATUS:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    
    # Set optimization
    torch.backends.cudnn.benchmark = True

# 2. Model Loading and Device Check
print("\n2. MODEL DEVICE CHECK:")
models = ['yolov8n', 'yolov8x']  # Test smallest and largest

for model_name in models:
    print(f"\n   Testing {model_name}:")
    model = YOLO(f'{model_name}.pt')
    
    # Check if model is on GPU
    device = next(model.model.parameters()).device
    print(f"   Model device: {device}")
    
    # Create test image
    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Warm up
    for _ in range(5):
        _ = model(test_img, verbose=False)
    
    # Time 20 inferences
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    times = []
    for _ in range(20):
        start = time.perf_counter()
        _ = model(test_img, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    fps = 1000 / avg_time
    
    print(f"   Avg time: {avg_time:.1f} ms")
    print(f"   FPS: {fps:.1f}")
    
    if torch.cuda.is_available():
        memory = torch.cuda.max_memory_allocated() / 1e6
        print(f"   GPU Memory: {memory:.1f} MB")
        torch.cuda.reset_peak_memory_stats()

# 3. Expected vs Actual
print("\n3. PERFORMANCE ANALYSIS:")
print("   Expected: nano ~100+ FPS, xlarge ~10-20 FPS")
print("   If both are similar, likely CPU bottleneck or sync issue")

# 4. Check if models are actually different sizes
print("\n4. MODEL SIZES:")
for model_name in models:
    model = YOLO(f'{model_name}.pt')
    params = sum(p.numel() for p in model.model.parameters())
    print(f"   {model_name}: {params/1e6:.1f}M parameters")