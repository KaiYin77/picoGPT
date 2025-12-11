"""
Training script for tiny GPT with int8 quantization support
Supports both standard training and quantization-aware training (QAT)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model.pico_model import PicoGPTConfig, PicoGPT
from quantization import (
    setup_quantization_aware_training,
    finalize_qat_model,
    post_training_quantization,
    compare_model_sizes,
    save_quantized_model,
    benchmark_model_inference
)

# -----------------------------------------------------------------------------
# Default config values designed for training on tiny Shakespeare
# I/O
out_dir = 'out-tiny'
eval_interval = 250
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'

# Data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 128

# Model
n_layer = 3
n_head = 4
n_embd = 192
dropout = 0.1
bias = False

# AdamW optimizer
learning_rate = 3e-3
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
enable_quantization = False
quantization_mode = 'qat'  # 'qat' for quantization-aware training, 'ptq' for post-training quantization
quantization_backend = 'fbgemm'
save_quantized_model_path = None

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Setup DDP if available
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data loading
data_dir = os.path.join('data', dataset)

def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Initialize model
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = PicoGPTConfig(**model_args)
    model = PicoGPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = PicoGPTConfig(**model_args)
    model = PicoGPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

model.to(device)

# Initialize a GradScaler
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Setup for quantization if enabled
if enable_quantization and quantization_mode == 'qat':
    print("Setting up quantization-aware training...")
    model = setup_quantization_aware_training(model, quantization_backend)

# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# Compile model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# DDP wrapper
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Logging
if master_process:
    print(f"Number of parameters: {model.get_num_params() if hasattr(model, 'get_num_params') else sum(p.numel() for p in model.parameters()):,}")

# Training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

if init_from == 'scratch':
    iter_num = 0
    best_val_loss = 1e9

while True:
    # Determine learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if iter_num == 0 and eval_only:
        break

    # Forward backward update, with optional gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps

        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    # Clip gradients
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt) if hasattr(raw_model, 'estimate_mfu') else -1.0
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    # Termination condition
    if iter_num > max_iters:
        break

# Post-training quantization or QAT finalization
if enable_quantization and master_process:
    print("Applying quantization...")

    # Get final model
    final_model = raw_model

    if quantization_mode == 'qat':
        # Finalize QAT model
        quantized_model = finalize_qat_model(final_model)
    elif quantization_mode == 'ptq':
        # Create calibration dataset
        def create_calibration_dataloader():
            calibration_data = []
            for _ in range(100):  # 100 batches for calibration
                X, Y = get_batch('train')
                calibration_data.append((X, Y))
            return calibration_data

        calibration_dataloader = create_calibration_dataloader()
        quantized_model = post_training_quantization(final_model, calibration_dataloader, quantization_backend, device)

    # Compare model sizes
    size_comparison = compare_model_sizes(final_model, quantized_model)
    print(f"Model size comparison:")
    print(f"  Original: {size_comparison['original_size_mb']:.2f} MB")
    print(f"  Quantized: {size_comparison['quantized_size_mb']:.2f} MB")
    print(f"  Compression ratio: {size_comparison['compression_ratio']:.2f}x")
    print(f"  Size reduction: {size_comparison['size_reduction_percent']:.1f}%")

    # Benchmark inference speed
    test_input = torch.randint(0, model_args['vocab_size'], (1, block_size), device=device)

    original_time = benchmark_model_inference(final_model, test_input, device)
    quantized_time = benchmark_model_inference(quantized_model, test_input, device)

    print(f"Inference speed comparison:")
    print(f"  Original model: {original_time:.2f} ms")
    print(f"  Quantized model: {quantized_time:.2f} ms")
    print(f"  Speedup: {original_time/quantized_time:.2f}x")

    # Save quantized model
    if save_quantized_model_path:
        save_quantized_model(quantized_model, save_quantized_model_path)
    else:
        quantized_path = os.path.join(out_dir, 'quantized_model.pt')
        save_quantized_model(quantized_model, quantized_path)

if ddp:
    destroy_process_group()