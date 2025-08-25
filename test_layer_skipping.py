"""
Test layer skipping with manual implementation
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import YolosForObjectDetection, YolosImageProcessor
from PIL import Image
import time
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class LayerSkippingYOLOS(nn.Module):
    """YOLOS with configurable layer skipping"""
    
    def __init__(self, model_name="hustvl/yolos-tiny"):
        super().__init__()
        self.base_model = YolosForObjectDetection.from_pretrained(model_name)
        self.num_layers = len(self.base_model.vit.encoder.layer)
        self.active_layers = [1] * self.num_layers  # All layers active by default
        
    def set_active_layers(self, pattern):
        """Set which layers are active (1) or skipped (0)"""
        self.active_layers = pattern[:self.num_layers]
        # Pad if needed
        while len(self.active_layers) < self.num_layers:
            self.active_layers.append(1)
            
    def forward(self, pixel_values):
        """Forward pass with layer skipping"""
        # Store original forward method
        original_forward = self.base_model.vit.encoder.forward
        
        # Create custom forward for encoder
        def custom_encoder_forward(hidden_states, height=None, width=None, 
                                  head_mask=None, output_attentions=False, 
                                  output_hidden_states=False, return_dict=True):
            for idx, layer in enumerate(self.base_model.vit.encoder.layer):
                if idx < len(self.active_layers) and self.active_layers[idx] == 1:
                    layer_head_mask = head_mask[idx] if head_mask is not None else None
                    layer_outputs = layer(hidden_states, layer_head_mask, output_attentions)
                    hidden_states = layer_outputs[0]
            
            from transformers.modeling_outputs import BaseModelOutput
            return BaseModelOutput(last_hidden_state=hidden_states)
        
        # Temporarily replace encoder forward
        self.base_model.vit.encoder.forward = custom_encoder_forward
        
        # Run normal model forward
        outputs = self.base_model(pixel_values=pixel_values)
        
        # Restore original forward
        self.base_model.vit.encoder.forward = original_forward
        
        return outputs


def test_configuration(model, image_processor, images, config_name, pattern, device):
    """Test a specific layer configuration"""
    model.set_active_layers(pattern)
    
    num_active = sum(pattern)
    print(f"\n{config_name}: {num_active}/{len(pattern)} layers active")
    print(f"Pattern: {pattern}")
    
    inference_times = []
    all_detections = []
    
    for idx, image in enumerate(images):
        # Preprocess
        inputs = image_processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        
        # Measure inference time
        times = []
        with torch.no_grad():
            # Warmup on first image
            if idx == 0:
                for _ in range(3):
                    _ = model(pixel_values)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
            
            # Time measurements
            for _ in range(10):
                start = time.perf_counter()
                outputs = model(pixel_values)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)
        
        avg_time = np.mean(times) * 1000  # ms
        inference_times.append(avg_time)
        
        # Get detections
        target_sizes = torch.tensor([[512, 512]]).to(device)
        try:
            results = image_processor.post_process_object_detection(
                outputs, 
                threshold=0.3,
                target_sizes=target_sizes
            )[0]
            num_detections = len(results['boxes'])
            all_detections.append(num_detections)
        except:
            num_detections = 0
            all_detections.append(0)
    
    avg_inference_time = np.mean(inference_times)
    avg_detections = np.mean(all_detections)
    fps = 1000 / avg_inference_time
    
    print(f"  Avg inference time: {avg_inference_time:.2f} ms")
    print(f"  FPS: {fps:.1f}")
    print(f"  Avg detections: {avg_detections:.1f}")
    
    return {
        'config': config_name,
        'num_layers': num_active,
        'inference_time_ms': avg_inference_time,
        'fps': fps,
        'avg_detections': avg_detections
    }


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = LayerSkippingYOLOS("hustvl/yolos-tiny")
    model.to(device)
    model.eval()
    
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
    
    # Load test images
    bdd_path = Path("/mnt/ssd1/divake/uncertain_skip/bdd100k/100k/val")
    image_paths = list(bdd_path.glob("*.jpg"))[:20]  # Use 20 images for testing
    
    print(f"Loading {len(image_paths)} test images...")
    images = [Image.open(p).convert('RGB') for p in image_paths]
    
    # Define configurations
    configs = {
        'full_12': [1,1,1,1,1,1,1,1,1,1,1,1],
        'heavy_10': [1,1,1,1,1,1,1,1,1,1,0,0],
        'standard_8': [1,1,1,1,1,1,1,1,0,0,0,0],
        'light_6': [1,1,1,1,1,1,0,0,0,0,0,0],
        'minimal_4': [1,1,1,1,0,0,0,0,0,0,0,0],
    }
    
    # Test each configuration
    print("\n" + "="*60)
    print("LAYER SKIPPING EXPERIMENT")
    print("="*60)
    
    results = []
    for config_name, pattern in configs.items():
        result = test_configuration(
            model, image_processor, images, 
            config_name, pattern, device
        )
        results.append(result)
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Calculate relative metrics
    baseline_time = df[df['config'] == 'full_12']['inference_time_ms'].values[0]
    baseline_detections = df[df['config'] == 'full_12']['avg_detections'].values[0]
    
    df['speedup'] = baseline_time / df['inference_time_ms']
    df['detection_ratio'] = df['avg_detections'] / baseline_detections
    
    # Print results table
    print("\n" + "="*60)
    print("RESULTS TABLE")
    print("="*60)
    print(df.to_string(index=False))
    
    # Save results
    df.to_csv('layer_skipping_results.csv', index=False)
    with open('layer_skipping_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Inference time vs layers
    ax1 = axes[0, 0]
    ax1.plot(df['num_layers'], df['inference_time_ms'], 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Active Layers')
    ax1.set_ylabel('Inference Time (ms)')
    ax1.set_title('Inference Time vs Model Complexity')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(df['num_layers'])
    
    # Plot 2: FPS vs layers
    ax2 = axes[0, 1]
    ax2.plot(df['num_layers'], df['fps'], 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Active Layers')
    ax2.set_ylabel('FPS')
    ax2.set_title('Inference Speed vs Model Complexity')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(df['num_layers'])
    
    # Plot 3: Detections vs layers
    ax3 = axes[1, 0]
    ax3.plot(df['num_layers'], df['avg_detections'], 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Active Layers')
    ax3.set_ylabel('Average Detections')
    ax3.set_title('Detection Performance vs Model Complexity')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(df['num_layers'])
    
    # Plot 4: Speedup and detection ratio
    ax4 = axes[1, 1]
    x = np.arange(len(df))
    width = 0.35
    bars1 = ax4.bar(x - width/2, df['speedup'], width, label='Speedup', color='skyblue')
    bars2 = ax4.bar(x + width/2, df['detection_ratio'], width, label='Detection Ratio', color='lightcoral')
    ax4.set_xlabel('Configuration')
    ax4.set_ylabel('Relative Performance')
    ax4.set_title('Performance Tradeoffs')
    ax4.set_xticks(x)
    ax4.set_xticklabels(df['config'], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Layer Skipping Analysis - YOLOS on BDD100K', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('layer_skipping_analysis_simple.png', dpi=150, bbox_inches='tight')
    
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    for _, row in df.iterrows():
        print(f"\n{row['config']}:")
        print(f"  - Layers: {row['num_layers']}/12")
        print(f"  - Speedup: {row['speedup']:.2f}x")
        print(f"  - Detection ratio: {row['detection_ratio']:.1%}")
        print(f"  - FPS: {row['fps']:.1f}")
    
    print("\nAnalysis complete! Results saved to:")
    print("  - layer_skipping_results.csv")
    print("  - layer_skipping_results.json")
    print("  - layer_skipping_analysis_simple.png")


if __name__ == "__main__":
    main()