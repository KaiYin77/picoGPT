# Pico GPT for character-level Shakespeare with int8 quantization
# Target: ~200-300K parameters for efficient quantization

out_dir = 'out-pico-shakespeare-char'
eval_interval = 250
eval_iters = 200
log_interval = 10

# Save checkpoints for quantization experiments
always_save_checkpoint = True

wandb_log = False
wandb_project = 'pico-shakespeare-char'
wandb_run_name = 'pico-gpt-int8'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 32  # Smaller batch size for pico model
block_size = 128  # Reduced context length

# Pico GPT architecture targeting ~200K parameters
# Formula: params ≈ vocab_size*n_embd + n_layer*(4*n_embd^2 + 3*n_embd) + n_embd*vocab_size
# With vocab_size=65 (Shakespeare chars), this gives us ~200K params
n_layer = 3      # 3 transformer layers
n_head = 4       # 4 attention heads
n_embd = 192     # 192 embedding dimensions (divisible by n_head)
dropout = 0.1    # Light dropout for pico model

learning_rate = 3e-3  # Higher LR for small model
max_iters = 8000
lr_decay_iters = 8000
min_lr = 3e-4
beta2 = 0.99

warmup_iters = 200

# Quantization settings
enable_quantization = True
quantization_mode = 'qat'  # Quantization Aware Training
quantization_backend = 'fbgemm'

# Model size estimation:
# Embedding: 65 * 192 = 12,480
# Each layer: 4 * 192^2 + 3 * 192 = 147,456 + 576 = 148,032
# Total per layer: ~148K
# 3 layers: 3 * 148K = 444K
# Output projection: 192 * 65 = 12,480
# Total: ~469K parameters (within range for quantization experiments)

bias = False  # Remove bias terms to reduce parameter count slightly