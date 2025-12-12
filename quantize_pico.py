"""
QAT (Quantization Aware Training) utilities for Pico GPT models
Essential functions only for training and finalizing quantized models
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.ao.quantization import QConfig
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver
from torch.ao.quantization.fake_quantize import FakeQuantize
import copy
import os

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

def count_quantized_parameters(model):
    """
    Count parameters in a quantized model, including packed parameters

    Args:
        model: Quantized model

    Returns:
        Total parameter count
    """
    total_params = 0

    for name, module in model.named_modules():
        # Only count leaf modules (those without child modules)
        if len(list(module.children())) > 0:
            continue

        # Count packed parameters from quantized linear layers
        if hasattr(module, '_packed_params'):
            try:
                weight, bias = module._packed_params.unpack()
                total_params += weight.numel()
                if bias is not None:
                    total_params += bias.numel()
            except Exception as e:
                # Fallback: try to get attributes from the quantized linear layer
                if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                    weight_params = module.in_features * module.out_features
                    total_params += weight_params
        else:
            # Count regular parameters (embeddings, layernorms, etc.)
            for param in module.parameters():
                total_params += param.numel()

    return total_params

def finalize_qat_model(qat_model):
    """
    Convert QAT model to quantized model for inference

    Args:
        qat_model: Model trained with QAT

    Returns:
        Final quantized model for inference
    """
    print(f"Finalizing QAT model...")
    original_param_count = sum(p.numel() for p in qat_model.parameters())
    print(f"QAT model parameters before conversion: {original_param_count:,}")

    qat_model.eval()

    try:
        quantized_model = quant.convert(qat_model, inplace=False)

        # Use our custom parameter counting function
        quantized_param_count = count_quantized_parameters(quantized_model)
        regular_param_count = sum(p.numel() for p in quantized_model.parameters())

        print(f"Quantized model parameters after conversion:")
        print(f"  Regular parameters (embeddings, LayerNorm, etc.): {regular_param_count:,}")
        print(f"  Total parameters (including packed): {quantized_param_count:,}")

        param_diff = abs(original_param_count - quantized_param_count)
        if param_diff < 15000:  # Allow reasonable difference for quantization metadata
            print(f"✓ Parameter count preserved during quantization (diff: {param_diff:,})")
        else:
            print(f"⚠ Parameter count mismatch: {original_param_count:,} → {quantized_param_count:,}")

        return quantized_model
    except Exception as e:
        print(f"Error during quantization conversion: {e}")
        print(f"Returning original model (unquantized)")
        return qat_model

def compare_model_sizes(original_model, quantized_model):
    """
    Compare sizes of original and quantized models

    Args:
        original_model: Original model
        quantized_model: Quantized model

    Returns:
        Dictionary with size comparison metrics
    """
    def get_model_size(model):
        """Get model size in bytes"""
        param_size = 0
        buffer_size = 0

        # Handle regular parameters and buffers
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        # Handle packed parameters in quantized models
        for module in model.modules():
            if hasattr(module, '_packed_params'):
                try:
                    weight, bias = module._packed_params.unpack()
                    param_size += weight.nelement() * weight.element_size()
                    if bias is not None:
                        param_size += bias.nelement() * bias.element_size()
                except:
                    # Fallback estimation for packed params
                    if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                        # Estimate: int8 weights = 1 byte per parameter
                        param_size += module.in_features * module.out_features * 1

        return param_size + buffer_size

    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)
    compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0

    return {
        'original_size_mb': original_size / (1024 * 1024),
        'quantized_size_mb': quantized_size / (1024 * 1024),
        'compression_ratio': compression_ratio,
        'size_reduction_percent': (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
    }

def save_quantized_model(model, path):
    """Save quantized model"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Quantized model saved to {path}")

def load_quantized_model(model_class, config, path, device='cpu'):
    """Load quantized model"""
    # Create model instance
    model = model_class(config)

    # Prepare for quantization to get the right structure
    model.qconfig = QConfig(
        activation=FakeQuantize.with_args(
            observer=MinMaxObserver.with_args(
                quant_min=0,
                quant_max=127
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

    # Skip embedding quantization
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            module.qconfig = None

    prepared_model = quant.prepare_qat(model, inplace=False)
    quantized_model = quant.convert(prepared_model, inplace=False)

    # Load state dict
    quantized_model.load_state_dict(torch.load(path, map_location=device))
    quantized_model.eval()
    quantized_model.to(device)

    return quantized_model