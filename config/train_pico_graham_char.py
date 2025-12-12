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

# Wandb logging
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

# Quantization settings
enable_quantization = True
quantization_mode = 'qat'  # Quantization Aware Training
quantization_backend = 'fbgemm'
save_quantized_model_path = None

# Model size estimation will depend on Graham essays vocab size
# Expected to be larger than Shakespeare's 65 characters
# Will be determined after data preparation