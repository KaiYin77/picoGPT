"""
Quantization utilities for Pico GPT models
- QAT (Quantization Aware Training) setup for training
- Post-training int8 quantization for deployment
- Model conversion and analysis utilities
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.ao.quantization import QConfig
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver
from torch.ao.quantization.fake_quantize import FakeQuantize
import copy
import os
import argparse
from model.pico_model import PicoGPTConfig, PicoGPT
from model.pico_model_int8 import quantize_linear_layers_in_model

def setup_quantization_aware_training(model, backend='fbgemm'):
    """
    Setup model for quantization-aware training

    Args:
        model: TinyGPT model
        backend: Quantization backend

    Returns:
        Model prepared for QAT
    """
    print(f"Setting up quantization-aware training with {backend} backend...")

    # Set backend
    torch.backends.quantized.engine = backend

    # Clone model
    model_copy = copy.deepcopy(model)
    model_copy.train()

    # Custom QConfig that works with embeddings
    qconfig = QConfig(
        activation=FakeQuantize.with_args(
            observer=MinMaxObserver.with_args(
                quant_min=0,
                quant_max=127  # Reduced range for better numerical stability
            ),
            quant_min=0,
            quant_max=127,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine
        ),
        weight=FakeQuantize.with_args(
            observer=PerChannelMinMaxObserver.with_args(
                ch_axis=0,
                quant_min=-128,
                quant_max=127
            ),
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric
        )
    )

    # Set the qconfig for the model
    model_copy.qconfig = qconfig

    # Skip quantization for embedding layers to avoid compatibility issues
    embedding_modules = []
    for name, module in model_copy.named_modules():
        if isinstance(module, nn.Embedding):
            embedding_modules.append(name)
            module.qconfig = None

    print(f"Embedding modules (will skip quantization): {embedding_modules}")

    # Prepare for QAT
    prepared_model = quant.prepare_qat(model_copy, inplace=False)

    print(f"QAT prepared model parameters: {sum(p.numel() for p in prepared_model.parameters()):,}")
    print("Model ready for quantization-aware training!")
    return prepared_model

# =============================================================================
# Post-Training Int8 Quantization Utilities
# =============================================================================

def quantize_model_to_int8(checkpoint_path, output_path):
    """
    Convert a trained model to FULL int8 quantized version for on-device deployment

    Args:
        checkpoint_path: Path to the trained model checkpoint
        output_path: Path to save the quantized model

    Returns:
        dict: Quantization statistics
    """
    print(f"Converting model to FULL int8 quantization: {checkpoint_path}")
    print("Target: ALL linear layers (embeddings + attention + MLP)")

    # Load checkpoint and create model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    gptconf = PicoGPTConfig(**checkpoint['model_args'])
    model = PicoGPT(gptconf)

    # Clean state dict of quantization artifacts
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    quantization_keys = []

    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        if any(quant_keyword in k for quant_keyword in [
            'activation_post_process', 'weight_fake_quant', 'fake_quant_enabled',
            'observer_enabled', 'scale', 'zero_point', 'min_val', 'max_val', 'eps'
        ]):
            quantization_keys.append(k)

    for k in quantization_keys:
        if k in state_dict:
            state_dict.pop(k)

    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded model with {model.get_num_params():,} parameters")

    # Apply FULL int8 quantization to ALL linear layers
    quantization_stats = quantize_linear_layers_in_model(model)

    # Save quantized model
    save_data = {
        'model_state_dict': model.state_dict(),
        'quantization_stats': quantization_stats,
        'model_args': {
            'n_layer': 3, 'n_head': 4, 'n_embd': 192, 'block_size': 128,
            'bias': False, 'vocab_size': 65, 'dropout': 0.0
        },
        'quantization_type': 'full_int8'
    }

    torch.save(save_data, output_path)
    print(f"Saved int8 quantized model to: {output_path}")

    # Print summary
    if '_summary' in quantization_stats:
        summary = quantization_stats['_summary']
        print(f"\nQuantization Summary:")
        print(f"  Layers quantized: {summary['total_layers_quantized']}")
        print(f"  Memory saved: {summary['total_memory_saved_bytes']/1024:.1f}KB")
        print(f"  Compression ratio: {summary['total_compression_ratio']:.1f}x")

    return quantization_stats


def main():
    """Command-line interface for full int8 quantization"""
    parser = argparse.ArgumentParser(description='Convert trained picoGPT model to FULL int8 quantized version')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for quantized model')

    args = parser.parse_args()

    print("Performing FULL int8 quantization for on-device deployment")
    print("All linear layers (embeddings + attention + MLP) will be quantized to int8")

    # Perform full quantization
    stats = quantize_model_to_int8(args.checkpoint, args.output)

    print("\n✓ FULL int8 quantization complete!")
    print(f"Use sample_pico_int8.py to test the quantized model")


if __name__ == "__main__":
    main()