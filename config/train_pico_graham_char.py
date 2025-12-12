# Pico GPT for character-level Graham Essays with int8 quantization
# Target: ~200-300K parameters for efficient quantization

import torch

# I/O
out_dir = 'out-pico-graham-char'
eval_interval = 250
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# Wandb logging (set wandb_log=True to enable)
wandb_log = False
wandb_project = 'pico-graham-char'
wandb_run_name = 'pico-gpt-graham-int8'

# Data
dataset = 'graham_char'
gradient_accumulation_steps = 1
batch_size = 32  # Smaller batch size for pico model
block_size = 128  # Reduced context length

# Pico GPT architecture targeting ~200K parameters
# Formula: params ≈ vocab_size*n_embd + n_layer*(4*n_embd^2 + 3*n_embd) + n_embd*vocab_size
# Graham essays will have larger vocab than Shakespeare, so adjusting accordingly
n_layer = 3      # 3 transformer layers
n_head = 4       # 4 attention heads
n_embd = 192     # 192 embedding dimensions (divisible by n_head)
dropout = 0.1    # Light dropout for pico model
bias = False     # Remove bias terms to reduce parameter count slightly

# AdamW optimizer
learning_rate = 3e-3  # Higher LR for small model
max_iters = 8000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 200
lr_decay_iters = 8000
min_lr = 3e-4

# DDP settings
backend = 'nccl'

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# Quantization settings (QAT only)
enable_quantization = True
quantization_backend = 'fbgemm'

# Model size estimation (depends on Graham essays vocab size):
# Assuming vocab_size ~100 characters (larger than Shakespeare's 65):
#
# Input embedding: vocab_size * n_embd = ~100 * 192 = ~19,200
# Position embedding: block_size * n_embd = 128 * 192 = 24,576
#
# Per transformer layer:
#   - LayerNorm 1: n_embd = 192
#   - Attention QKV: n_embd * (3 * n_embd) = 192 * 576 = 110,592
#   - Attention proj: n_embd * n_embd = 192 * 192 = 36,864
#   - LayerNorm 2: n_embd = 192
#   - MLP fc: n_embd * (2 * n_embd) = 192 * 384 = 73,728
#   - MLP proj: (2 * n_embd) * n_embd = 384 * 192 = 73,728
#   - Total per layer: 110,592 + 36,864 + 73,728 + 73,728 + 384 = 295,296
#
# 3 transformer layers: 3 * 295,296 = 885,888
# Output projection: n_embd * vocab_size = 192 * ~100 = ~19,200 (tied with input embedding)
# Final LayerNorm: n_embd = 192
#
# Estimated total: ~19,200 + 24,576 + 885,888 + 192 = ~929,856 parameters
# (Actual size will depend on final vocab_size after data preparation)