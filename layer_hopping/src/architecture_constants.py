"""
YOLOS Architecture Constants - FIXED, WILL NEVER CHANGE

These are the fundamental parameters of YOLOS-small architecture
determined through inspection and testing. Once verified, these
values are PERMANENT for this project.

DO NOT MODIFY unless switching to a completely different model.
"""

# ==============================================================================
# MODEL IDENTIFICATION
# ==============================================================================
MODEL_NAME = "hustvl/yolos-small"
MODEL_TYPE = "yolos"

# ==============================================================================
# IMAGE PROCESSING - FIXED
# ==============================================================================
INPUT_SIZE = 512  # We use 512×512 square images
PATCH_SIZE = 16   # 16×16 patches (from YOLOS architecture)
NUM_PATCHES_PER_DIM = INPUT_SIZE // PATCH_SIZE  # 32
NUM_IMAGE_PATCHES = NUM_PATCHES_PER_DIM ** 2    # 1024

# Image normalization (ImageNet statistics)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# ==============================================================================
# TRANSFORMER ARCHITECTURE - FIXED
# ==============================================================================
HIDDEN_DIM = 384          # Hidden dimension
NUM_LAYERS = 12           # Transformer layers
NUM_ATTENTION_HEADS = 6   # Attention heads per layer
FFN_INTERMEDIATE_SIZE = 1536  # Feed-forward network size

# ==============================================================================
# DETECTION - FIXED
# ==============================================================================
NUM_DETECTION_TOKENS = 100  # Object query tokens
NUM_CLASSES = 91           # COCO classes (0-90)
PERSON_CLASS_ID = 0        # Class ID for person
NO_OBJECT_CLASS_ID = 91    # Class ID for no object/background

# ==============================================================================
# SEQUENCE STRUCTURE - FIXED
# ==============================================================================
# Sequence = [CLS, image_patch_1, ..., image_patch_1024, det_token_1, ..., det_token_100]
HAS_CLS_TOKEN = True
NUM_CLS_TOKENS = 1 if HAS_CLS_TOKEN else 0

TOTAL_SEQ_LENGTH = NUM_CLS_TOKENS + NUM_IMAGE_PATCHES + NUM_DETECTION_TOKENS  # 1125

# Token positions in sequence
CLS_TOKEN_POS = 0
IMAGE_PATCH_START = NUM_CLS_TOKENS  # Position 1
IMAGE_PATCH_END = IMAGE_PATCH_START + NUM_IMAGE_PATCHES  # Position 1025
DETECTION_TOKEN_START = IMAGE_PATCH_END  # Position 1025
DETECTION_TOKEN_END = DETECTION_TOKEN_START + NUM_DETECTION_TOKENS  # Position 1125

# ==============================================================================
# ATTENTION STRUCTURE - FIXED
# ==============================================================================
# Attention shape: [batch, num_heads, seq_len, seq_len]
ATTENTION_SHAPE = (None, NUM_ATTENTION_HEADS, TOTAL_SEQ_LENGTH, TOTAL_SEQ_LENGTH)

# ==============================================================================
# GFLOPS COMPUTATION (for 512×512 input) - FIXED
# ==============================================================================
GFLOPS_PATCH_EMBEDDING = 0.15
GFLOPS_PER_TRANSFORMER_LAYER = 5.8
GFLOPS_DETECTION_HEAD = 0.04
GFLOPS_TOTAL_12_LAYERS = 69.79

# Early exit GFLOPs
GFLOPS_EXIT_LAYER_3 = 17.6   # 75% savings
GFLOPS_EXIT_LAYER_6 = 35.0   # 50% savings
GFLOPS_EXIT_LAYER_9 = 52.4   # 25% savings
GFLOPS_EXIT_LAYER_12 = 69.8  # 0% savings (full model)

# ==============================================================================
# EXIT LAYERS - CONFIGURABLE (but fixed for experiments)
# ==============================================================================
EXIT_LAYERS = [3, 6, 9, 12]  # Layer indices for early exit

# ==============================================================================
# DATASET - MOT17 - FIXED
# ==============================================================================
DATASET_NAME = "MOT17"
DATASET_ROOT = "/ssd_4TB/divake/uncertain_skip/data/MOT17/train"

# Training sequences
TRAIN_SEQUENCES = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN"
]

# Validation sequences
VAL_SEQUENCES = [
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN"
]

# ==============================================================================
# VERIFICATION FUNCTION
# ==============================================================================
def verify_architecture():
    """
    Verify that the constants match actual YOLOS architecture.
    Run this once to ensure everything is correct.
    """
    from transformers import YolosForObjectDetection
    import torch

    print("Verifying architecture constants...")

    model = YolosForObjectDetection.from_pretrained(MODEL_NAME)
    config = model.config

    # Verify architecture parameters
    assert config.hidden_size == HIDDEN_DIM, f"Hidden size mismatch: {config.hidden_size} vs {HIDDEN_DIM}"
    assert config.num_hidden_layers == NUM_LAYERS, f"Num layers mismatch: {config.num_hidden_layers} vs {NUM_LAYERS}"
    assert config.num_attention_heads == NUM_ATTENTION_HEADS, f"Num heads mismatch"
    assert config.patch_size == PATCH_SIZE, f"Patch size mismatch"
    assert config.num_detection_tokens == NUM_DETECTION_TOKENS, f"Num detection tokens mismatch"
    assert config.num_labels == NUM_CLASSES, f"Num classes mismatch"

    # Verify sequence length
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        outputs = model(dummy_input, output_hidden_states=True)

    actual_seq_len = outputs.hidden_states[0].shape[1]
    assert actual_seq_len == TOTAL_SEQ_LENGTH, f"Sequence length mismatch: {actual_seq_len} vs {TOTAL_SEQ_LENGTH}"

    print("✓ All architecture constants verified!")
    print(f"  - Model: {MODEL_NAME}")
    print(f"  - Input size: {INPUT_SIZE}×{INPUT_SIZE}")
    print(f"  - Sequence length: {TOTAL_SEQ_LENGTH}")
    print(f"  - Hidden dim: {HIDDEN_DIM}")
    print(f"  - Num layers: {NUM_LAYERS}")

    return True


if __name__ == "__main__":
    verify_architecture()
