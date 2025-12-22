# PicoGPT TensorFlow/LiteRT Implementation

This directory contains the TensorFlow implementation of PicoGPT, migrated from the original PyTorch version. The implementation is designed for deployment on CIMv3 hardware using TensorFlow Lite with INT8 quantization.

## Overview

The TensorFlow implementation provides:
- Identical model architecture to PyTorch version
- CIMv3 hardware compatibility (ReLU, no bias, optimized dimensions)
- TFLite INT8 Post-Training Quantization (PTQ)
- Weight tying between embeddings and output projection
- Full training, sampling, and conversion pipeline

## Directory Structure

```
PicoGPT-SDK/
├── tf_model/                       # TensorFlow model implementation
│   ├── __init__.py
│   ├── config.py                   # Model configuration
│   ├── layers.py                   # Custom layers (LayerNorm, Attention, MLP, Block)
│   └── model.py                    # PicoGPT model
├── tf_config/                      # Training configurations
│   └── train_pico_shakespeare_char.py
├── tf_data.py                      # Data pipeline (compatible with PyTorch data)
├── tf_train.py                     # Training script
├── tf_sample.py                    # Text generation script
├── tf_convert_to_tflite.py        # TFLite conversion with INT8 PTQ
├── data/                           # Shared data directory
│   └── shakespeare_char/
│       ├── train.bin
│       ├── val.bin
│       └── meta.pkl
└── README_TENSORFLOW.md            # This file
```

## Requirements

### Prerequisites

```bash
# Install UV for modern Python package management
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure TensorFlow dependencies are installed
uv sync  # Install all dependencies from pyproject.toml
```

**Note**: PyTorch is NOT required for the TensorFlow implementation. The two implementations are completely independent.

## Quick Start

### 1. Prepare Data (Reuse PyTorch Data)

If you already have PyTorch data prepared:

```bash
# Data is already prepared at data/shakespeare_char/
# Contains: train.bin, val.bin, meta.pkl
```

If you need to prepare data from scratch:

```bash
uv run python data/shakespeare_char/prepare.py
```

### 2. Train the Model

```bash
uv run python tf_train.py --config tf_config/train_pico_shakespeare_char.py
```

Training configuration:
- **Batch size**: 32
- **Iterations**: 8000
- **Learning rate**: 3e-3 (cosine decay with warmup)
- **Model size**: ~418K parameters
- **Mixed precision**: FP16 (if GPU available)

Expected training time:
- GPU (RTX 3090): ~10-15 minutes
- CPU: ~2-3 hours

Output:
- Model saved to: `out-tf-pico-shakespeare-char/model.keras`
- Config saved to: `out-tf-pico-shakespeare-char/model_config.pkl`

### 3. Generate Text

```bash
uv run python tf_sample.py --out_dir out-tf-pico-shakespeare-char --start "O God, O God!"
```

Options:
- `--start`: Start prompt for generation
- `--num_samples`: Number of samples to generate
- `--max_new_tokens`: Tokens per sample
- `--temperature`: Sampling temperature (0.1-2.0, higher = more random)
- `--top_k`: Top-k filtering (limits vocabulary to top-k tokens)

### 4. Convert to TFLite with INT8 Quantization + KV-Cache

```bash
# KV-cache enabled by default for optimized inference
uv run python tf_convert_to_tflite.py --model_dir out-tf-pico-shakespeare-char

# To disable KV-cache (export standard model)
uv run python tf_convert_to_tflite.py --model_dir out-tf-pico-shakespeare-char --no_kv_cache
```

This performs:
1. **KV-Cache Export (Default)**: Enables efficient autoregressive generation with O(N) complexity
2. **Post-Training Quantization (PTQ)** using representative dataset
3. **Hybrid INT8 quantization**: Weights INT8, cache tensors float32 (optimal for KV-cache)
4. **Model compression**: ~4x size reduction (1.6MB → 400KB)
5. **Automatic C Array Generation**: Generates `model_data.h` and `model_data.cc` with model config

**Note:** KV-cache models use hybrid quantization (weights INT8, caches float32) because cache tensors must maintain precision. Non-KV-cache models use full INT8 quantization for all layers.

#### KV-Cache Benefits

✅ **2-10x faster inference** - Linear O(N) complexity instead of quadratic O(N²)
✅ **Enabled by default** - No configuration needed
✅ **AOT optimization** - Cache built into TFLite model for maximum efficiency

**How it works:**
- Standard mode: Recomputes attention for all previous tokens at each step
- KV-cache mode: Stores Key/Value tensors, only processes new tokens

**Performance example (128-token generation):**
- Standard: ~16,384 attention operations for last token
- KV-cache: ~128 attention operations for last token
- Result: ~128x faster for final tokens, 2-10x overall speedup

Output:
- `out-tf-pico-shakespeare-char/model_data_int8.tflite`: Quantized TFLite model with KV-cache
- `../include/pico_gpt/model_data.h`: C header with model config defines
- `../src/pico_gpt/model_data.cc`: C source for model array

### 5. Deploy to CIMv3 Hardware

The `tf_convert_to_tflite.py` script automatically generates the necessary C header and source files for embedding the TFLite model into your C/C++ project.

**KV-Cache is automatically enabled in CMakeLists.txt** for the PicoGPT_Runtime project - no manual configuration needed!

Next steps:
1. Verify the generated files:
   - `../include/pico_gpt/model_data.h` (includes model config defines)
   - `../src/pico_gpt/model_data.cc`
2. Build your project:
   ```bash
   cmake --preset=riscv-debug
   cmake --build --preset=riscv-debug-build
   ```
3. The runtime automatically uses KV-cache for efficient generation - no code changes required!

## Model Architecture

### Configuration

```python
PicoGPTConfig(
    block_size=128,      # Maximum sequence length
    vocab_size=65,       # Character vocabulary size
    n_layer=3,           # Transformer layers
    n_head=2,            # Attention heads (d_head=64, CIMv3 optimized)
    n_embd=128,          # Embedding dimension (CIMv3 compatible)
    dropout=0.1,         # Dropout rate
    bias=False           # No bias (CIMv3 preference)
)
```

### CIMv3 Hardware Constraints

✅ **Satisfied**:
- n_embd ∈ {128, 256, 512}
- d_head = 64 (optimized)
- ReLU activation only (no GELU)
- No bias in Linear layers
- Post-norm architecture (no pre-norm LN1)
- block_size ≤ 128

### Layer Details

**LayerNorm** (no bias):
- Custom implementation matching PyTorch
- gamma parameter only (no beta/bias)

**CausalSelfAttention**:
- Combined QKV projection (efficiency)
- Manual causal masking (TFLite compatible)
- Multi-head attention with d_head=64
- No flash attention (TFLite limitation)

**MLP**:
- 2x expansion ratio (compact model)
- ReLU activation (CIMv3 constraint)
- No bias

**Block**:
- Post-norm: MHA → FFN → LN2
- Residual connections

**Weight Tying**:
- Input embeddings shared with output projection
- `logits = matmul(x, wte.embeddings, transpose_b=True)`
- Zero additional parameters

## Troubleshooting

### KV-Cache Issues

**Issue: Compilation error "PICOGPT_N_LAYER not defined"**

Solution: Make sure you exported the model with KV-cache enabled (default) which adds config defines to `model_data.h`.

**Issue: Out of memory during runtime**

Solution: KV-cache requires additional memory (~512 KB for 4-layer model). Increase tensor arena size in `include/pico_gpt/runtime.h`:
```c
#define PICOGPT_TENSOR_ARENA_SIZE (512 * 1024)  // Increase from 256 KB
```

**Issue: Need to disable KV-cache**

Solution:
1. Export without KV-cache: `python tf_convert_to_tflite.py --model_dir <dir> --no_kv_cache`
2. Comment out `"PICOGPT_USE_KV_CACHE"` in `CMakeLists.txt` line 476
3. Rebuild your project

### General Issues

**Issue: Different model output between runs**

Solution: This was a bug in config path lookup - fixed. Both commands now produce identical output:
```bash
python tf_convert_to_tflite.py --model_dir <dir>
python tf_convert_to_tflite.py --model_dir <dir> --output <path>
```

**Issue: Training loss not decreasing**

Check:
- Learning rate (try 3e-3 to 1e-3)
- Batch size (32 recommended for character-level)
- Data loading (verify train.bin exists)
- Model capacity (n_layer=3, n_embd=128 minimum)

## Summary

This TensorFlow implementation provides a complete pipeline from training to deployment:

✅ **Training**: Full TensorFlow training with mixed precision
✅ **Quantization**: INT8 PTQ for 4x compression
✅ **KV-Cache**: 2-10x faster inference (enabled by default)
✅ **CIMv3 Ready**: Hardware constraints satisfied
✅ **Easy Deployment**: Automatic C array generation

The model is fully compatible with the PyTorch version and provides identical results while being optimized for embedded deployment on CIMv3 hardware.
