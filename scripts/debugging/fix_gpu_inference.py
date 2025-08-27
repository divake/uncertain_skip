#!/usr/bin/env python3
"""
Fix GPU inference issue and properly benchmark YOLOv8 models
"""

import torch
import time
import numpy as np
from ultralytics import YOLO

print("="*60)
print("FIXED GPU INFERENCE TEST")
print("="*60)

# Verify GPU
if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}")
torch.backends.cudnn.benchmark = True

# Test both models WITH proper GPU placement
models_to_test = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
results = []

for model_name in models_to_test:
    print(f"\nTesting {model_name}:")
    
    # Load model and EXPLICITLY move to GPU
    model = YOLO(f'{model_name}.pt')
    model.to('cuda')  # Force GPU
    
    # Verify it's on GPU
    device = next(model.model.parameters()).device
    print(f"  Model device: {device}")
    
    if str(device) == 'cpu':
        print("  ⚠️ WARNING: Still on CPU, trying alternative method...")
        model.model = model.model.cuda()
        device = next(model.model.parameters()).device
        print(f"  Model device after .cuda(): {device}")
    
    # Create test batch
    test_imgs = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(10)]
    
    # Warm up GPU
    print("  Warming up...")
    for _ in range(10):
        _ = model(test_imgs[0], device='cuda', verbose=False)
    
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    # Benchmark
    times = []
    for img in test_imgs:
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # Run inference explicitly on GPU
        _ = model(img, device='cuda', verbose=False)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    
    # Calculate metrics
    avg_time = np.mean(times) * 1000  # ms
    std_time = np.std(times) * 1000
    fps = 1000 / avg_time
    
    # Memory usage
    memory_mb = torch.cuda.max_memory_allocated() / 1e6
    torch.cuda.reset_peak_memory_stats()
    
    # Model size
    params = sum(p.numel() for p in model.model.parameters())
    
    results.append({
        'model': model_name,
        'params_M': params/1e6,
        'avg_time_ms': avg_time,
        'std_ms': std_time,
        'fps': fps,
        'memory_mb': memory_mb
    })
    
    print(f"  Parameters: {params/1e6:.1f}M")
    print(f"  Inference: {avg_time:.1f} ± {std_time:.1f} ms")
    print(f"  FPS: {fps:.1f}")
    print(f"  GPU Memory: {memory_mb:.1f} MB")

# Summary table
print("\n" + "="*60)
print("PERFORMANCE SUMMARY (GPU)")
print("="*60)
print("| Model | Params | Time(ms) | FPS   | Memory(MB) | Speedup |")
print("|-------|--------|----------|-------|------------|---------|")

baseline_fps = results[0]['fps']
for r in results:
    speedup = baseline_fps / r['fps'] if r['model'] != 'yolov8n' else 1.0
    print(f"| {r['model']:5s} | {r['params_M']:5.1f}M | {r['avg_time_ms']:8.1f} | "
          f"{r['fps']:5.1f} | {r['memory_mb']:10.1f} | {speedup:7.2f}x |")

# Check if performance scaling is correct
fps_range = max(r['fps'] for r in results) / min(r['fps'] for r in results)
print(f"\nFPS Range: {fps_range:.1f}x (expect 5-10x)")

if fps_range < 3:
    print("⚠️ Performance scaling still seems low. Possible issues:")
    print("  - Models may still be partially on CPU")
    print("  - Input preprocessing might be bottleneck")
    print("  - Need to check CUDA synchronization")
else:
    print("✅ Performance scaling looks correct!")