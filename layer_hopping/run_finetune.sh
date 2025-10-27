#!/bin/bash

# Fine-tune YOLOS Layer 12 Detection Head on MOT17
# Uses official YOLOS code with comprehensive logging

echo "================================"
echo "YOLOS Layer 12 Fine-tuning"
echo "================================"
echo ""

# GPU selection
export CUDA_VISIBLE_DEVICES=1

# Run fine-tuning
python3 finetune_yolos_layer12.py \
    --backbone_name small \
    --batch_size 4 \
    --epochs 50 \
    --lr 5e-5 \
    --lr_backbone 0.0 \
    --weight_decay 1e-4 \
    --warmup-epochs 5 \
    --lr_drop 40 \
    --num_workers 8 \
    --freeze_backbone \
    --output_dir models/yolos_finetuned_layer12 \
    2>&1 | tee models/yolos_finetuned_layer12/training_output.log

echo ""
echo "================================"
echo "Fine-tuning complete!"
echo "Check models/yolos_finetuned_layer12/ for results"
echo "================================"
