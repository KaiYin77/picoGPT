"""
TensorFlow training configuration for PicoGPT on TinyStories dataset
Optimized for clean English text with simple vocabulary

Target: ~200-300K parameters for efficient INT8 quantization
CIMv3-compatible architecture
Dataset: TinyStories (simple English stories, no Unicode issues)
"""

# I/O
out_dir = 'out-tf-sub-pico-tinystories-tiny-bpe'
eval_interval = 500
log_interval = 10
eval_iters = 200
always_save_checkpoint = True

# Data
dataset = 'tinystories_tiny_bpe'
# TinyStories with BPE produces clean tokens; larger batch for stable training
batch_size = 128
block_size = 128  # Context length (can increase to 256 if memory allows)

# CIMv3-Compatible Pico GPT architecture
# CIMv3 constraints: n_embd must be 128/256/512, d_head should be 64
n_layer = 1      # 1 transformer layer (increase to 2 for better quality)
n_head = 2       # 2 attention heads (128 / 2 = 64 d_head - CIMv3 optimized)
n_embd = 128     # 128 embedding dimensions (CIMv3 compatible)
dropout = 0.1    # Light dropout for pico model
bias = False     # CIMv3 prefers bias-free INT8 GEMM operations

# AdamW optimizer
learning_rate = 2e-3  # Start with 2e-3, can lower to 1e-3 for more stable convergence
max_iters = 20000     # Increased for larger TinyStories dataset (vs 12000 for Shakespeare)
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# Learning rate decay
warmup_iters = 800      # Longer warmup for larger dataset
lr_decay_iters = 20000  # Match max_iters
min_lr = 2e-4

# Mixed precision (TensorFlow equivalent of PyTorch AMP)
# NOTE: Set to False for CPU training - mixed precision is slower on CPU
mixed_precision = False  # Use 'mixed_float16' policy (GPU only)

# Dataset-specific notes:
# TinyStories advantages:
#   - Clean ASCII text (no Unicode encoding issues)
#   - Simple vocabulary (~1,500 words vs WikiText's 30,000+)
#   - Coherent narrative structure
#   - Perfect for 256-vocab BPE tokenizer
#
# Expected performance:
#   - Better convergence than Shakespeare (more data)
#   - More coherent generations (simple grammar)
#   - Faster training (clean tokenization)

# CIMv3-Compatible Model size estimation:
# Input embedding: vocab_size * n_embd = 256 * 128 = 32,768
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
# 1 transformer layer: 1 * 131,200 = 131,200
# Output projection: tied with input embedding (0 additional params)
# Final LayerNorm: n_embd = 128
#
# Total: 32,768 + 16,384 + 131,200 + 128 = 180,480 parameters
# CIMv3 Features: ReLU activation, no bias, d_head=64 optimized
#
# For better quality on TinyStories, consider:
#   - Increasing n_layer to 2: ~311K parameters (still embeddable)
#   - Increasing n_embd to 256: ~700K parameters (requires more RAM)
#   - Increasing max_iters to 30000-50000 (better convergence)
