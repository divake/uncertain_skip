#!/usr/bin/env python3
"""
Quick test of multi-exit training - 2 epochs to verify everything works
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from multi_exit_trainer import *

def quick_test():
    """Quick training test with minimal data"""
    
    print("="*60)
    print("Quick Multi-Exit Training Test (2 epochs)")
    print("="*60)
    
    # Minimal config for testing
    config = {
        'model_name': 'hustvl/yolos-small',
        'train_sequences': ['MOT17-02'],  # Just one sequence
        'val_sequences': ['MOT17-11'],
        'batch_size': 2,
        'epochs': 2,  # Just 2 epochs for testing
        'learning_rate': 1e-4,
        'log_dir': 'results/test_training'
    }
    
    # Initialize
    processor = YolosImageProcessor.from_pretrained(config['model_name'])
    
    # Small datasets for testing
    train_dataset = MOT17Dataset(
        config['train_sequences'], 
        processor=processor
    )
    # Limit to 20 frames for quick test
    train_dataset.frames = train_dataset.frames[:20]
    
    val_dataset = MOT17Dataset(
        config['val_sequences'],
        processor=processor
    )
    val_dataset.frames = val_dataset.frames[:10]
    
    print(f"\n📁 Test datasets:")
    print(f"  Train: {len(train_dataset)} frames")
    print(f"  Val: {len(val_dataset)} frames")
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    # Model
    print("\n🔨 Building model...")
    model = MultiExitYOLOS(config['model_name'])
    model.cuda()
    
    # Test forward passes
    print("\n🧪 Testing forward passes...")
    
    # Get one batch
    batch = next(iter(train_loader))
    pixel_values = batch['pixel_values'].cuda()
    
    # Test single exit
    print("  Testing single exit (layer 3)...")
    out3 = model.forward_single_exit(pixel_values, exit_layer=3)
    print(f"    Output shapes: logits={out3['pred_logits'].shape}, boxes={out3['pred_boxes'].shape}")
    
    # Test all exits
    print("  Testing all exits...")
    all_outputs = model.forward_all_exits(pixel_values)
    for layer_name, outputs in all_outputs.items():
        print(f"    {layer_name}: logits={outputs['pred_logits'].shape}, boxes={outputs['pred_boxes'].shape}")
    
    # Test loss
    print("\n🧪 Testing loss computation...")
    criterion = MultiExitLoss()
    targets = {
        'boxes': batch['boxes'].cuda(),
        'labels': batch['labels'].cuda(),
        'n_objects': batch['n_objects']
    }
    
    loss, loss_dict = criterion(all_outputs, targets)
    print(f"  Total loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"    {k}: {v:.4f}")
    
    # Quick training
    print("\n🚀 Running quick training test...")
    trainer = MultiExitTrainer(model, config)
    trainer.train(
        train_loader, 
        val_loader,
        epochs=config['epochs'],
        lr=config['learning_rate']
    )
    
    print("\n✅ Test complete! Check results/test_training/ for outputs")

if __name__ == "__main__":
    quick_test()