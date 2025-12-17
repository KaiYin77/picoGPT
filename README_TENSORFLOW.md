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

### 4. Convert to TFLite with INT8 Quantization

```bash
uv run python tf_convert_to_tflite.py \
    --model_dir out-tf-pico-shakespeare-char
    # Output paths for .tflite, .h, and .cc now default to:
    #   - TFLite: out-tf-pico-shakespeare-char/model_data_int8.tflite
    #   - C Header: ../include/pico_gpt/model_data.h
    #   - C Source: ../src/pico_gpt/model_data.cc
    # You can customize these paths with --output, --c_header_path, --c_source_path
```

This performs:
1. **Post-Training Quantization (PTQ)** using representative dataset
2. **INT8 quantization** for weights, activations, inputs, outputs
3. **Model compression**: ~4x size reduction (1.6MB → 400KB)
4. **Automatic C Array Generation**: Generates `model_data.h` and `model_data.cc` for embedding.

Output:
- `out-tf-pico-shakespeare-char/model_data_int8.tflite`: Quantized TFLite model
- `../include/pico_gpt/model_data.h`: C header for model array
- `../src/pico_gpt/model_data.cc`: C source for model array

### 5. Deploy to CIMv3 Hardware

The `tf_convert_to_tflite.py` script now automatically generates the necessary C header and source files for embedding the TFLite model into your C/C++ project.

Next steps:
1.  Verify the generated `model_data.h` in `../include/pico_gpt/` and `model_data.cc` in `../src/pico_gpt/`.
2.  Integrate these files into your CIMv3 firmware project.
3.  Rebuild and test your application on CIMv3 hardware.

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
