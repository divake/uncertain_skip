#!/usr/bin/env python3
"""
Profile GFLOPs for YOLOS model at different exit points
This is crucial to understand computational savings potential
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import numpy as np
from transformers import YolosImageProcessor, YolosForObjectDetection, YolosModel
from PIL import Image
from pathlib import Path
import time
from thop import profile, clever_format
import json
from datetime import datetime

class GFLOPProfiler:
    def __init__(self, model_size='small'):
        """Initialize GFLOP profiler for YOLOS"""
        
        print("=" * 60)
        print("GFLOP Profiling for Layer-wise Early Exit")
        print(f"Model: YOLOS-{model_size}")
        print("=" * 60)
        
        self.model_size = model_size
        self.device = 'cuda'
        
        # Load model
        model_name = f"hustvl/yolos-{model_size}"
        self.processor = YolosImageProcessor.from_pretrained(model_name)
        self.full_model = YolosForObjectDetection.from_pretrained(model_name)
        self.full_model.to(self.device)
        self.full_model.eval()
        
        # Get model architecture info
        self.analyze_architecture()
        
        # Results directory
        self.results_dir = Path('results/gflop_analysis')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.profile_file = self.results_dir / f'gflop_profile_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
    def analyze_architecture(self):
        """Analyze YOLOS architecture to understand layer structure"""
        
        print("\n📊 Model Architecture Analysis:")
        print("-" * 40)
        
        # Get the vision model (ViT backbone)
        vit_model = self.full_model.vit
        
        # Count encoder layers
        n_layers = len(vit_model.encoder.layer)
        print(f"Number of ViT encoder layers: {n_layers}")
        
        # Analyze each layer
        for i, layer in enumerate(vit_model.encoder.layer):
            n_params = sum(p.numel() for p in layer.parameters())
            print(f"  Layer {i+1}: {n_params/1e6:.2f}M parameters")
        
        # Detection head parameters
        detect_params = sum(p.numel() for p in self.full_model.class_labels_classifier.parameters())
        detect_params += sum(p.numel() for p in self.full_model.bbox_predictor.parameters())
        print(f"\nDetection head: {detect_params/1e6:.2f}M parameters")
        
        self.n_layers = n_layers
        
    def create_early_exit_model(self, exit_layer):
        """Create a model that exits at specified layer"""
        
        class EarlyExitYOLOS(torch.nn.Module):
            def __init__(self, base_model, exit_layer):
                super().__init__()
                self.base_model = base_model
                self.exit_layer = exit_layer
                
            def forward(self, pixel_values):
                # Get the vision model
                vit = self.base_model.vit
                
                # Patch embedding
                embeddings = vit.embeddings(pixel_values)
                encoder_outputs = embeddings
                
                # Process through encoder layers up to exit point
                for i in range(min(self.exit_layer, len(vit.encoder.layer))):
                    layer_module = vit.encoder.layer[i]
                    encoder_outputs = layer_module(encoder_outputs)[0]
                
                # Simple detection head (simplified for profiling)
                # In real implementation, we'd have separate heads per exit
                batch_size = encoder_outputs.shape[0]
                sequence_length = encoder_outputs.shape[1]
                
                # Dummy output matching YOLOS format
                logits = torch.randn(batch_size, sequence_length, 92).to(encoder_outputs.device)  # 91 classes + 1 no-object
                pred_boxes = torch.randn(batch_size, sequence_length, 4).to(encoder_outputs.device)
                
                return {'logits': logits, 'pred_boxes': pred_boxes}
        
        return EarlyExitYOLOS(self.full_model, exit_layer)
    
    def profile_single_exit(self, exit_layer, input_size=(3, 224, 224)):
        """Profile GFLOPs for a specific exit layer"""
        
        print(f"\n🔍 Profiling exit at layer {exit_layer}...")
        
        # Create model with early exit
        model = self.create_early_exit_model(exit_layer)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_size).to(self.device)
        
        # Profile using THOP
        try:
            macs, params = profile(model, inputs=(dummy_input,), verbose=False)
            gflops = macs / 1e9  # Convert MACs to GFLOPs
            
            # Also measure actual inference time
            # Warm up
            for _ in range(10):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            # Measure
            torch.cuda.synchronize()
            start = time.time()
            n_iterations = 100
            for _ in range(n_iterations):
                with torch.no_grad():
                    _ = model(dummy_input)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            avg_time = elapsed / n_iterations
            
            result = {
                'exit_layer': exit_layer,
                'gflops': gflops,
                'params_M': params / 1e6,
                'inference_time_ms': avg_time * 1000,
                'theoretical_fps': 1.0 / avg_time
            }
            
            print(f"  GFLOPs: {gflops:.2f}")
            print(f"  Params: {params/1e6:.2f}M")
            print(f"  Inference: {avg_time*1000:.2f}ms")
            print(f"  FPS: {1.0/avg_time:.1f}")
            
            return result
            
        except Exception as e:
            print(f"  Error profiling: {e}")
            # Fallback to manual estimation
            return self.estimate_gflops(exit_layer)
    
    def estimate_gflops(self, exit_layer):
        """Estimate GFLOPs based on layer count"""
        
        # Rough estimation based on ViT structure
        # Each ViT layer has similar computational cost
        
        # Base costs (patch embedding, position encoding, etc.)
        base_gflops = 0.5
        
        # Per-layer cost (for ViT-Small, roughly 0.9 GFLOPs per layer)
        per_layer_gflops = 0.9
        
        # Detection head cost
        head_gflops = 0.2
        
        total_gflops = base_gflops + (exit_layer * per_layer_gflops) + head_gflops
        
        return {
            'exit_layer': exit_layer,
            'gflops': total_gflops,
            'params_M': 'estimated',
            'inference_time_ms': 'estimated',
            'theoretical_fps': 'estimated'
        }
    
    def profile_all_exits(self):
        """Profile GFLOPs for all possible exit points"""
        
        print("\n" + "="*60)
        print("Profiling All Exit Points")
        print("="*60)
        
        results = []
        
        # Test early exits at layers 3, 6, 9, 12
        exit_points = [3, 6, 9, 12]
        
        for exit_layer in exit_points:
            result = self.profile_single_exit(exit_layer)
            results.append(result)
        
        # Calculate relative savings
        full_gflops = results[-1]['gflops']  # Layer 12 is full model
        
        print("\n📊 GFLOP Savings Analysis:")
        print("-" * 40)
        print(f"{'Exit Layer':<12} {'GFLOPs':<10} {'Savings':<10} {'Speed-up':<10}")
        print("-" * 40)
        
        for result in results:
            savings = (1 - result['gflops'] / full_gflops) * 100
            speedup = full_gflops / result['gflops']
            result['savings_percent'] = savings
            result['theoretical_speedup'] = speedup
            
            print(f"Layer {result['exit_layer']:<6} {result['gflops']:<10.2f} "
                  f"{savings:<9.1f}% {speedup:<9.2f}x")
        
        # Save results
        self.save_results(results)
        
        return results
    
    def profile_with_real_image(self):
        """Profile using actual MOT17 image"""
        
        print("\n🖼️ Profiling with Real MOT17 Image...")
        
        # Load a real image
        img_path = Path("../data/MOT17/train/MOT17-11-FRCNN/img1/000001.jpg")
        if not img_path.exists():
            print("  Warning: MOT17 image not found, using dummy input")
            return
        
        image = Image.open(img_path)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        print(f"  Image shape: {pixel_values.shape}")
        
        results = []
        for exit_layer in [3, 6, 9, 12]:
            print(f"\n  Testing layer {exit_layer}...")
            
            model = self.create_early_exit_model(exit_layer)
            model.eval()
            
            # Warm up
            for _ in range(5):
                with torch.no_grad():
                    _ = model(pixel_values)
            
            # Measure
            torch.cuda.synchronize()
            times = []
            for _ in range(50):
                start = time.time()
                with torch.no_grad():
                    _ = model(pixel_values)
                torch.cuda.synchronize()
                times.append(time.time() - start)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            print(f"    Avg time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
            print(f"    FPS: {1.0/avg_time:.1f}")
            
            results.append({
                'exit_layer': exit_layer,
                'avg_time_ms': avg_time * 1000,
                'std_time_ms': std_time * 1000,
                'fps': 1.0 / avg_time
            })
        
        return results
    
    def save_results(self, results):
        """Save profiling results"""
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'model': f'yolos-{self.model_size}',
            'n_layers': self.n_layers,
            'profile_results': results,
            'summary': {
                'max_savings_percent': max(r.get('savings_percent', 0) for r in results),
                'max_theoretical_speedup': max(r.get('theoretical_speedup', 1) for r in results)
            }
        }
        
        with open(self.profile_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Results saved to: {self.profile_file}")


def main():
    """Run GFLOP profiling"""
    
    profiler = GFLOPProfiler(model_size='small')
    
    # Profile all exit points
    profiler.profile_all_exits()
    
    # Profile with real image
    profiler.profile_with_real_image()
    
    print("\n" + "="*60)
    print("✅ GFLOP Profiling Complete!")
    print("="*60)


if __name__ == "__main__":
    main()