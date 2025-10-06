#!/usr/bin/env python3
"""
Test early exit detection WITHOUT fine-tuning
This answers: Can early layers detect anything meaningful without training?
Research question: How much detection capability exists at each layer naturally?
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import numpy as np
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import pandas as pd

class NaiveDetectionHead(nn.Module):
    """Simple detection head for testing - no training"""
    def __init__(self, hidden_dim=384, num_classes=1):
        super().__init__()
        # Random initialization - no training!
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.bbox_head = nn.Linear(hidden_dim, 4)  # bbox coordinates
        
        # Use YOLOS-style object queries (random)
        self.num_queries = 100
        self.query_embed = nn.Parameter(torch.randn(self.num_queries, hidden_dim))
        
    def forward(self, hidden_states):
        """
        hidden_states: [batch, seq_len, hidden_dim]
        Returns: dict with 'logits' and 'pred_boxes'
        """
        batch_size = hidden_states.shape[0]
        
        # Simple approach: Use object queries to attend to features
        # This is extremely naive - just for testing!
        queries = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Average pooling over spatial positions (except CLS token)
        if hidden_states.shape[1] > self.num_queries:
            # Take first num_queries positions (very naive!)
            features = hidden_states[:, :self.num_queries, :]
        else:
            features = hidden_states
            
        # Get predictions
        logits = self.class_head(features)
        pred_boxes = self.bbox_head(features).sigmoid()  # Normalize to [0,1]
        
        return {
            'logits': logits,
            'pred_boxes': pred_boxes
        }

class EarlyExitTester:
    def __init__(self):
        """Initialize tester for early exit without fine-tuning"""
        
        print("=" * 60)
        print("Testing Early Exit Detection WITHOUT Fine-tuning")
        print("Research Question: What can each layer detect naturally?")
        print("=" * 60)
        
        # Load base model
        print("\n🔄 Loading YOLOS-small...")
        self.model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')
        self.model.cuda()
        self.model.eval()
        
        self.processor = YolosImageProcessor.from_pretrained('hustvl/yolos-small')
        
        # Create naive detection heads for early exits
        print("\n📦 Creating naive detection heads (random init)...")
        self.heads = {
            3: NaiveDetectionHead().cuda(),
            6: NaiveDetectionHead().cuda(),
            9: NaiveDetectionHead().cuda(),
            12: 'original'  # Use original model head
        }
        
        # Put all in eval mode (no training!)
        for head in self.heads.values():
            if head != 'original':
                head.eval()
        
        self.results_dir = Path('results/early_exit_baseline')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_features_at_layer(self, images, layer_num):
        """Extract features at specified layer"""
        
        with torch.no_grad():
            # Get ViT model
            vit = self.model.vit
            
            # Process through embeddings
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs['pixel_values'].cuda()
            
            hidden_states = vit.embeddings(pixel_values)
            
            # Process through encoder layers up to layer_num
            for i in range(min(layer_num, 12)):
                layer_outputs = vit.encoder.layer[i](hidden_states)
                hidden_states = layer_outputs[0]
            
            # Apply layer norm (important!)
            if layer_num == 12:
                hidden_states = vit.layernorm(hidden_states)
            
            return hidden_states
    
    def detect_at_layer(self, images, layer_num):
        """Get detections from specified layer"""
        
        if layer_num == 12:
            # Use original model
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process
            target_sizes = torch.tensor([images.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes.cuda(),
                threshold=0.3  # Lower threshold for testing
            )[0]
            
            return results
        else:
            # Use naive head at early layer
            features = self.extract_features_at_layer(images, layer_num)
            head = self.heads[layer_num]
            
            with torch.no_grad():
                outputs = head(features)
            
            # Convert to detection format
            # This is very naive - just for testing!
            logits = outputs['logits'][0]  # Remove batch dim
            boxes = outputs['pred_boxes'][0]
            
            # Get class predictions (person = class 0 in our single-class setup)
            scores = torch.softmax(logits, dim=-1)
            person_scores = scores[:, 0]  # Person class
            
            # Filter by score
            keep = person_scores > 0.3
            
            # Scale boxes to image size
            img_h, img_w = images.size[::-1]
            boxes_scaled = boxes[keep] * torch.tensor([img_w, img_h, img_w, img_h]).cuda()
            
            # Convert from cxcywh to xyxy (approximate)
            boxes_xyxy = torch.zeros_like(boxes_scaled)
            boxes_xyxy[:, 0] = boxes_scaled[:, 0] - boxes_scaled[:, 2]/2  # x1
            boxes_xyxy[:, 1] = boxes_scaled[:, 1] - boxes_scaled[:, 3]/2  # y1
            boxes_xyxy[:, 2] = boxes_scaled[:, 0] + boxes_scaled[:, 2]/2  # x2
            boxes_xyxy[:, 3] = boxes_scaled[:, 1] + boxes_scaled[:, 3]/2  # y2
            
            return {
                'boxes': boxes_xyxy,
                'scores': person_scores[keep],
                'labels': torch.ones_like(person_scores[keep])
            }
    
    def evaluate_layer_on_sequence(self, layer_num, sequence='MOT17-11', max_frames=50):
        """Evaluate detection at specific layer on sequence"""
        
        print(f"\n🔬 Testing Layer {layer_num} on {sequence}...")
        
        # Load ground truth
        seq_path = Path(f"../data/MOT17/train/{sequence}-FRCNN")
        img_dir = seq_path / "img1"
        gt_file = seq_path / "gt" / "gt.txt"
        
        gt_data = pd.read_csv(gt_file, header=None)
        gt_data.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'visibility']
        
        # Process frames
        img_files = sorted(img_dir.glob("*.jpg"))[:max_frames]
        
        tp_total = 0
        fp_total = 0
        fn_total = 0
        
        for img_path in tqdm(img_files, desc=f"Layer {layer_num}"):
            frame_num = int(img_path.stem)
            
            # Load image
            image = Image.open(img_path)
            
            # Get detections
            try:
                detections = self.detect_at_layer(image, layer_num)
                det_boxes = detections['boxes'].cpu().numpy() if torch.is_tensor(detections['boxes']) else detections['boxes']
            except Exception as e:
                print(f"  Error at frame {frame_num}: {e}")
                det_boxes = np.array([])
            
            # Get GT for this frame
            frame_gt = gt_data[gt_data['frame'] == frame_num]
            gt_boxes = frame_gt[['x', 'y', 'w', 'h']].values
            gt_boxes_xyxy = np.column_stack([
                gt_boxes[:, 0],
                gt_boxes[:, 1],
                gt_boxes[:, 0] + gt_boxes[:, 2],
                gt_boxes[:, 1] + gt_boxes[:, 3]
            ])
            
            # Calculate TP/FP/FN
            if len(det_boxes) == 0:
                fn_total += len(gt_boxes_xyxy)
                continue
                
            if len(gt_boxes_xyxy) == 0:
                fp_total += len(det_boxes)
                continue
            
            # Simple IoU matching
            tp, fp, fn = self.match_boxes(det_boxes, gt_boxes_xyxy)
            tp_total += tp
            fp_total += fp
            fn_total += fn
        
        # Calculate metrics
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'layer': layer_num,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp_total,
            'fp': fp_total,
            'fn': fn_total
        }
    
    def match_boxes(self, pred_boxes, gt_boxes, iou_threshold=0.5):
        """Simple IoU matching for TP/FP/FN calculation"""
        
        tp = 0
        matched_gt = set()
        
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                    
                iou = self.calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
        
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        
        return tp, fp, fn
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def run_complete_test(self):
        """Test all layers without fine-tuning"""
        
        print("\n" + "="*60)
        print("Starting Complete Early Exit Test (No Fine-tuning)")
        print("="*60)
        
        results = []
        
        for layer in [3, 6, 9, 12]:
            metrics = self.evaluate_layer_on_sequence(layer, max_frames=50)
            results.append(metrics)
            
            print(f"\n📊 Layer {layer} Results:")
            print(f"  Precision: {metrics['precision']:.1%}")
            print(f"  Recall: {metrics['recall']:.1%}")
            print(f"  F1 Score: {metrics['f1']:.1%}")
        
        # Save results
        results_df = pd.DataFrame(results)
        results_file = self.results_dir / f'early_exit_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(results_file, index=False)
        
        print("\n" + "="*60)
        print("SUMMARY: Detection Capability Without Fine-tuning")
        print("="*60)
        print(f"{'Layer':<10} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
        print("-"*46)
        for r in results:
            print(f"Layer {r['layer']:<4} {r['precision']:<11.1%} {r['recall']:<11.1%} {r['f1']:<11.1%}")
        
        print(f"\n✅ Results saved to: {results_file}")
        
        return results


if __name__ == "__main__":
    tester = EarlyExitTester()
    results = tester.run_complete_test()
    
    print("\n🔍 Analysis:")
    if results[0]['f1'] < 0.1:  # Layer 3
        print("  ✓ Layer 3 cannot detect without fine-tuning (expected)")
    else:
        print("  ⚠️ Layer 3 has some detection capability - interesting!")
    
    print("\nConclusion: Fine-tuning is essential for early exits!")