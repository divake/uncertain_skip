"""
Multi-Exit YOLOS Model

Extends YOLOS with detection heads at intermediate layers (8, 10, 12)
for adaptive inference with computational savings.

Architecture:
- Layers 1-12: Pretrained YOLOS backbone (frozen initially)
- Layer 8:  NEW detection head (class + bbox MLPs)
- Layer 10: NEW detection head (class + bbox MLPs)
- Layer 12: PRETRAINED detection head (from YOLOS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import YolosModel, YolosForObjectDetection
from typing import Dict, List, Optional


class MLP(nn.Module):
    """
    3-layer MLP (Multi-Layer Perceptron) for detection head

    Same architecture as YOLOS/DETR:
    - Layer 1: Linear(input_dim → hidden_dim) + ReLU
    - Layer 2: Linear(hidden_dim → hidden_dim) + ReLU
    - Layer 3: Linear(hidden_dim → output_dim) [no activation]
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiExitYOLOS(nn.Module):
    """
    Multi-Exit YOLOS for adaptive inference

    Features:
    - Detection heads at layers 8, 10, 12
    - Auxiliary losses during training (all exits)
    - Single exit selection during inference
    - Pretrained backbone and Layer 12 head
    """

    def __init__(
        self,
        pretrained_model_name: str = "hustvl/yolos-small",
        exit_layers: List[int] = [8, 10, 12],
        hidden_dim: int = 384,
        num_classes: int = 1,  # Person only for MOT17
        num_detection_tokens: int = 100,
        freeze_backbone: bool = True,
        freeze_layer12_head: bool = True
    ):
        super().__init__()

        self.exit_layers = sorted(exit_layers)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_detection_tokens = num_detection_tokens

        # Load pretrained YOLOS model
        print(f"Loading pretrained YOLOS from {pretrained_model_name}...")
        pretrained_model = YolosForObjectDetection.from_pretrained(pretrained_model_name)

        # Extract backbone (transformer encoder)
        self.backbone = pretrained_model.vit

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✓ Backbone frozen")

        # Initialize detection heads for each exit layer
        self.detection_heads = nn.ModuleDict()

        for layer_idx in self.exit_layers:
            if layer_idx == 12:
                # Use pretrained Layer 12 bbox head, but create new class head for num_classes
                # Note: YOLOS-small pretrained has 91 classes (COCO), but we need num_classes
                # So we initialize new class head but keep bbox head
                self.detection_heads[f'layer_{layer_idx}'] = nn.ModuleDict({
                    # Classification head: NEW (to match num_classes)
                    'class_head': MLP(hidden_dim, hidden_dim, num_classes + 1, num_layers=3),
                    # Bounding box head: Use pretrained (already knows how to predict boxes)
                    'bbox_head': pretrained_model.bbox_predictor
                })

                # Freeze Layer 12 bbox head if requested (keep class head trainable)
                if freeze_layer12_head:
                    for param in self.detection_heads[f'layer_{layer_idx}']['bbox_head'].parameters():
                        param.requires_grad = False
                    print(f"✓ Layer {layer_idx} head: NEW class head (trainable), pretrained bbox head (frozen)")
                else:
                    print(f"✓ Layer {layer_idx} head: NEW class head (trainable), pretrained bbox head (trainable)")
            else:
                # Create new detection head for early exits
                self.detection_heads[f'layer_{layer_idx}'] = nn.ModuleDict({
                    # Classification head: hidden_dim → num_classes + 1 (including no-object)
                    'class_head': MLP(hidden_dim, hidden_dim, num_classes + 1, num_layers=3),
                    # Bounding box head: hidden_dim → 4 (cx, cy, w, h)
                    'bbox_head': MLP(hidden_dim, hidden_dim, 4, num_layers=3)
                })
                print(f"✓ Layer {layer_idx} head initialized (NEW, trainable)")

        print(f"\nMulti-Exit YOLOS created with exits at layers: {self.exit_layers}")
        self._print_trainable_params()

    def _print_trainable_params(self):
        """Print number of trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\nParameter count:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"  Frozen: {total_params - trainable_params:,}")

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_all_exits: bool = None,
        exit_layer: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model

        Args:
            pixel_values: [batch, 3, H, W] input images
            output_all_exits: If True, return outputs from all exits (for training)
                             If None, use self.training (automatic)
            exit_layer: If specified, only compute this exit (for inference)

        Returns:
            Dictionary containing:
            - pred_logits: [batch, num_queries, num_classes+1] classification logits
            - pred_boxes: [batch, num_queries, 4] bounding box predictions
            - aux_outputs: List of dicts for auxiliary exits (only during training)
        """

        # Determine if we need all exits
        if output_all_exits is None:
            output_all_exits = self.training

        # Get hidden states from all transformer layers
        outputs = self.backbone(
            pixel_values,
            output_hidden_states=True,
            return_dict=True
        )

        # Collect outputs from specified exits
        exit_outputs = {}

        for layer_idx in self.exit_layers:
            # Skip if specific exit requested and this isn't it
            if exit_layer is not None and layer_idx != exit_layer:
                continue

            # Get hidden states for this layer (0-indexed)
            if layer_idx == 12:
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs.hidden_states[layer_idx - 1]

            # Extract detection token features (last 100 tokens)
            # Sequence: [CLS (1)] + [Image patches (1024)] + [Detection tokens (100)]
            detection_features = hidden_states[:, -self.num_detection_tokens:, :]

            # Apply detection heads
            head = self.detection_heads[f'layer_{layer_idx}']
            pred_logits = head['class_head'](detection_features)
            pred_boxes = head['bbox_head'](detection_features).sigmoid()

            exit_outputs[layer_idx] = {
                'pred_logits': pred_logits,
                'pred_boxes': pred_boxes
            }

        # Format output based on mode
        if exit_layer is not None:
            # Single exit mode (inference)
            return exit_outputs[exit_layer]

        elif output_all_exits:
            # Multi-exit mode (training with auxiliary losses)
            # Primary output: highest layer (Layer 12)
            main_output = exit_outputs[self.exit_layers[-1]]

            # Auxiliary outputs: earlier layers
            aux_outputs = [
                exit_outputs[layer_idx]
                for layer_idx in self.exit_layers[:-1]
            ]

            return {
                'pred_logits': main_output['pred_logits'],
                'pred_boxes': main_output['pred_boxes'],
                'aux_outputs': aux_outputs
            }

        else:
            # Default: return only primary output
            main_output = exit_outputs[self.exit_layers[-1]]
            return {
                'pred_logits': main_output['pred_logits'],
                'pred_boxes': main_output['pred_boxes']
            }

    def get_exit_layer_params(self, layer_idx: int) -> List:
        """Get parameters for a specific exit layer"""
        return list(self.detection_heads[f'layer_{layer_idx}'].parameters())

    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")

    def freeze_exit_head(self, layer_idx: int):
        """Freeze detection head for specific layer"""
        for param in self.detection_heads[f'layer_{layer_idx}'].parameters():
            param.requires_grad = False
        print(f"Layer {layer_idx} head frozen")

    def unfreeze_exit_head(self, layer_idx: int):
        """Unfreeze detection head for specific layer"""
        for param in self.detection_heads[f'layer_{layer_idx}'].parameters():
            param.requires_grad = True
        print(f"Layer {layer_idx} head unfrozen")


def build_multi_exit_yolos(config: dict) -> MultiExitYOLOS:
    """
    Build Multi-Exit YOLOS model from configuration

    Args:
        config: Dictionary with model configuration

    Returns:
        MultiExitYOLOS model
    """
    # Handle both Phase 1 and Phase 2 config formats
    if 'phase1' in config['training']:
        # Phase 1 config format
        freeze_backbone = config['training']['phase1']['freeze_backbone']
        freeze_layer12_head = config['training']['phase1']['freeze_layer12_head']
    elif 'freeze' in config['training']:
        # Phase 2 config format
        freeze_backbone = config['training']['freeze']['backbone']
        freeze_layer12_head = config['training']['freeze']['layer_12_bbox']
    else:
        # Fallback defaults
        freeze_backbone = True
        freeze_layer12_head = True

    model = MultiExitYOLOS(
        pretrained_model_name=config['model']['backbone'],
        exit_layers=config['model']['exit_layers'],
        hidden_dim=config['model']['hidden_dim'],
        num_classes=config['model']['num_classes'],
        num_detection_tokens=config['model']['num_detection_tokens'],
        freeze_backbone=freeze_backbone,
        freeze_layer12_head=freeze_layer12_head
    )

    return model


if __name__ == "__main__":
    # Test the model
    print("Testing Multi-Exit YOLOS model...")

    model = MultiExitYOLOS(
        pretrained_model_name="hustvl/yolos-small",
        exit_layers=[8, 10, 12],
        freeze_backbone=True,
        freeze_layer12_head=True
    )

    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 512, 512)

    print("\n" + "="*80)
    print("Testing forward pass...")
    print("="*80)

    # Training mode (all exits)
    model.train()
    output_train = model(dummy_input)
    print(f"\nTraining mode output:")
    print(f"  pred_logits shape: {output_train['pred_logits'].shape}")
    print(f"  pred_boxes shape: {output_train['pred_boxes'].shape}")
    print(f"  aux_outputs: {len(output_train['aux_outputs'])} auxiliary exits")

    # Inference mode (single exit)
    model.eval()
    output_eval_8 = model(dummy_input, exit_layer=8)
    output_eval_10 = model(dummy_input, exit_layer=10)
    output_eval_12 = model(dummy_input, exit_layer=12)

    print(f"\nInference mode (Layer 8):")
    print(f"  pred_logits shape: {output_eval_8['pred_logits'].shape}")
    print(f"  pred_boxes shape: {output_eval_8['pred_boxes'].shape}")

    print("\n✓ Model test passed!")
