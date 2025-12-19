"""
TensorFlow training configuration for PicoGPT
Matches PyTorch config: config/train_pico_shakespeare_char.py

Target: ~200-300K parameters for efficient INT8 quantization
CIMv3-compatible architecture
"""

# I/O
out_dir = 'out-tf-sub-pico-shakespeare-char'
eval_interval = 500
log_interval = 10
eval_iters = 200
always_save_checkpoint = True

# Data
dataset = 'shakespeare_char'
batch_size = 64  # Smaller batch size for pico model
block_size = 128  # Reduced context length

# CIMv3-Compatible Pico GPT architecture
# CIMv3 constraints: n_embd must be 128/256/512, d_head should be 64
n_layer = 1      # 1 transformer layers
n_head = 2       # 2 attention heads (128 / 2 = 64 d_head - CIMv3 optimized)
n_embd = 128     # 128 embedding dimensions (CIMv3 compatible)
dropout = 0.1    # Light dropout for pico model
bias = False     # CIMv3 prefers bias-free INT8 GEMM operations

# AdamW optimizer
learning_rate = 3e-3  # Higher LR for small model
max_iters = 8000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# Learning rate decay
warmup_iters = 200
lr_decay_iters = 8000
min_lr = 3e-4

# Mixed precision (TensorFlow equivalent of PyTorch AMP)
# NOTE: Set to False for CPU training - mixed precision is slower on CPU
mixed_precision = False  # Use 'mixed_float16' policy (GPU only)

# CIMv3-Compatible Model size estimation:
# Input embedding: vocab_size * n_embd = 65 * 128 = 8,320
# Position embedding: block_size * n_embd = 128 * 128 = 16,384
#
# Per transformer layer (CIMv3 sequence: MHA → FFN → LN2, no LN1):
#   - Attention QKV: n_embd * (3 * n_embd) = 128 * 384 = 49,152
#   - Attention proj: n_embd * n_embd = 128 * 128 = 16,384
#   - MLP fc: n_embd * (2 * n_embd) = 128 * 256 = 32,768
#   - MLP proj: (2 * n_embd) * n_embd = 256 * 128 = 32,768
#   - LayerNorm 2: n_embd = 128
#   - Total per layer: 49,152 + 16,384 + 32,768 + 32,768 + 128 = 131,200
#
# 3 transformer layers: 3 * 131,200 = 393,600
# Output projection: tied with input embedding (0 additional params)
# Final LayerNorm: n_embd = 128
#
# Total: 8,320 + 16,384 + 393,600 + 128 = 418,432 parameters
# CIMv3 Features: ReLU activation, no bias, d_head=64 optimized
