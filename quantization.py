"""
Int8 quantization utilities for tiny GPT models
Supports both post-training quantization (PTQ) and quantization-aware training (QAT)
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization.quantize_fx import prepare_fx, convert_fx
import copy
import os
from typing import Dict, Any, Optional

def get_quantization_config(backend='fbgemm'):
    """Get quantization configuration for the specified backend"""
    if backend == 'fbgemm':
        return quant.get_default_qconfig('fbgemm')
    elif backend == 'qnnpack':
        return quant.get_default_qconfig('qnnpack')
    else:
        raise ValueError(f"Unsupported backend: {backend}")

class QuantizedTinyGPT(nn.Module):
    """Wrapper for quantized TinyGPT model"""

    def __init__(self, model, quantization_config=None):
        super().__init__()
        self.model = model
        self.quantization_config = quantization_config or get_quantization_config('fbgemm')

    def forward(self, idx, targets=None):
        return self.model(idx, targets)

def prepare_model_for_quantization(model, backend='fbgemm', calibration_dataset=None):
    """
    Prepare model for quantization

    Args:
        model: TinyGPT model to quantize
        backend: Quantization backend ('fbgemm' or 'qnnpack')
        calibration_dataset: Dataset for calibration (for PTQ)

    Returns:
        Prepared model ready for quantization
    """
    # Set backend
    torch.backends.quantized.engine = backend

    # Clone the model to avoid modifying the original
    model_copy = copy.deepcopy(model)
    model_copy.eval()

    # Set quantization configuration
    qconfig = get_quantization_config(backend)
    model_copy.qconfig = qconfig

    # Prepare the model
    prepared_model = quant.prepare(model_copy, inplace=False)

    return prepared_model

def calibrate_model(prepared_model, calibration_dataloader, device='cpu'):
    """
    Calibrate the model with representative data

    Args:
        prepared_model: Model prepared for quantization
        calibration_dataloader: DataLoader with calibration data
        device: Device to run calibration on
    """
    prepared_model.eval()
    prepared_model.to(device)

    print("Starting model calibration...")
    with torch.no_grad():
        for i, (data, targets) in enumerate(calibration_dataloader):
            if i >= 100:  # Use first 100 batches for calibration
                break
            data = data.to(device)
            targets = targets.to(device) if targets is not None else None
            _ = prepared_model(data, targets)

            if (i + 1) % 20 == 0:
                print(f"Calibration step {i + 1}/100")

    print("Calibration completed!")

def quantize_model(prepared_model):
    """
    Convert prepared model to quantized model

    Args:
        prepared_model: Model prepared and calibrated for quantization

    Returns:
        Quantized model
    """
    quantized_model = quant.convert(prepared_model, inplace=False)
    return quantized_model

def post_training_quantization(model, calibration_dataloader, backend='fbgemm', device='cpu'):
    """
    Perform post-training quantization on the model

    Args:
        model: Original TinyGPT model
        calibration_dataloader: DataLoader for calibration
        backend: Quantization backend
        device: Device for calibration

    Returns:
        Quantized model
    """
    print(f"Starting post-training quantization with {backend} backend...")

    # Prepare model
    prepared_model = prepare_model_for_quantization(model, backend)

    # Calibrate
    calibrate_model(prepared_model, calibration_dataloader, device)

    # Convert to quantized model
    quantized_model = quantize_model(prepared_model)

    print("Post-training quantization completed!")
    return quantized_model

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

    # Set QAT configuration
    qconfig = torch.quantization.get_default_qat_qconfig(backend)
    model_copy.qconfig = qconfig

    # Prepare for QAT
    prepared_model = quant.prepare_qat(model_copy, inplace=False)

    print("Model ready for quantization-aware training!")
    return prepared_model

def finalize_qat_model(qat_model):
    """
    Convert QAT model to quantized model for inference

    Args:
        qat_model: Model trained with QAT

    Returns:
        Final quantized model for inference
    """
    qat_model.eval()
    quantized_model = quant.convert(qat_model, inplace=False)
    return quantized_model

def compare_model_sizes(original_model, quantized_model):
    """
    Compare sizes of original and quantized models

    Args:
        original_model: Original model
        quantized_model: Quantized model

    Returns:
        Dictionary with size comparison
    """
    def get_model_size(model):
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        return param_size + buffer_size

    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)

    compression_ratio = original_size / quantized_size

    return {
        'original_size_mb': original_size / (1024 * 1024),
        'quantized_size_mb': quantized_size / (1024 * 1024),
        'compression_ratio': compression_ratio,
        'size_reduction_percent': (1 - quantized_size / original_size) * 100
    }

def save_quantized_model(model, path):
    """Save quantized model"""
    torch.save(model.state_dict(), path)
    print(f"Quantized model saved to {path}")

def load_quantized_model(model_class, config, path, device='cpu'):
    """Load quantized model"""
    # Create model structure
    model = model_class(config)

    # Prepare for quantization to get the right structure
    model.qconfig = get_quantization_config('fbgemm')
    prepared_model = quant.prepare(model, inplace=False)
    quantized_model = quant.convert(prepared_model, inplace=False)

    # Load state dict
    quantized_model.load_state_dict(torch.load(path, map_location=device))
    quantized_model.eval()

    return quantized_model

def benchmark_model_inference(model, test_input, device='cpu', num_runs=100):
    """
    Benchmark model inference speed

    Args:
        model: Model to benchmark
        test_input: Sample input tensor
        device: Device for benchmarking
        num_runs: Number of runs for averaging

    Returns:
        Average inference time in milliseconds
    """
    model.eval()
    model.to(device)
    test_input = test_input.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_input)

    # Benchmark
    torch.cuda.synchronize() if device.startswith('cuda') else None
    start_time = torch.cuda.Event(enable_timing=True) if device.startswith('cuda') else None
    end_time = torch.cuda.Event(enable_timing=True) if device.startswith('cuda') else None

    if device.startswith('cuda'):
        start_time.record()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(test_input)
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
    else:
        import time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(test_input)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000

    avg_time = elapsed_time / num_runs
    return avg_time

def quantization_error_analysis(original_model, quantized_model, test_dataloader, device='cpu'):
    """
    Analyze quantization error by comparing outputs

    Args:
        original_model: Original floating-point model
        quantized_model: Quantized model
        test_dataloader: Test data for comparison
        device: Device for computation

    Returns:
        Dictionary with error metrics
    """
    original_model.eval()
    quantized_model.eval()
    original_model.to(device)
    quantized_model.to(device)

    mse_errors = []
    max_errors = []

    with torch.no_grad():
        for i, (data, targets) in enumerate(test_dataloader):
            if i >= 50:  # Analyze first 50 batches
                break

            data = data.to(device)

            # Get outputs
            orig_logits, _ = original_model(data)
            quant_logits, _ = quantized_model(data)

            # Calculate errors
            mse_error = torch.nn.functional.mse_loss(orig_logits, quant_logits).item()
            max_error = torch.max(torch.abs(orig_logits - quant_logits)).item()

            mse_errors.append(mse_error)
            max_errors.append(max_error)

    return {
        'avg_mse_error': sum(mse_errors) / len(mse_errors),
        'avg_max_error': sum(max_errors) / len(max_errors),
        'max_mse_error': max(mse_errors),
        'max_absolute_error': max(max_errors)
    }