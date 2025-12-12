"""
Int8 quantized model implementation for picoGPT
Provides Int8Linear layer and quantization utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .pico_model import PicoGPTConfig, PicoGPT


class Int8Linear(nn.Module):
    """
    Int8 quantized linear layer that stores weights as int8
    and performs inference with on-the-fly dequantization

    This provides true int8 weight storage with 4x memory compression
    compared to float32 weights while maintaining inference quality.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Store quantized weights as int8 with scale and zero_point
        self.weight_int8 = nn.Parameter(
            torch.zeros((out_features, in_features), dtype=torch.int8),
            requires_grad=False
        )
        self.weight_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.weight_zero_point = nn.Parameter(torch.tensor(0), requires_grad=False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    @classmethod
    def from_float(cls, float_linear):
        """
        Convert a float32 linear layer to int8 quantized version

        Args:
            float_linear: Original nn.Linear layer

        Returns:
            Int8Linear: Quantized equivalent layer
        """
        int8_linear = cls(
            float_linear.in_features,
            float_linear.out_features,
            float_linear.bias is not None
        )

        # Quantize weights to int8 using symmetric quantization
        weight = float_linear.weight.data
        weight_min = weight.min()
        weight_max = weight.max()

        # Calculate scale for symmetric quantization (-127 to 127)
        weight_range = max(abs(weight_min), abs(weight_max))
        scale = weight_range / 127.0

        if scale == 0:
            scale = 1.0

        # Quantize to int8
        weight_quantized = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)

        # Store quantized parameters
        int8_linear.weight_int8.data = weight_quantized
        int8_linear.weight_scale.data = torch.tensor(scale)
        int8_linear.weight_zero_point.data = torch.tensor(0)  # Symmetric quantization

        # Copy bias if present
        if float_linear.bias is not None:
            int8_linear.bias.data = float_linear.bias.data.clone()

        return int8_linear

    def forward(self, input):
        """
        Forward pass with on-the-fly dequantization

        Args:
            input: Input tensor

        Returns:
            Output tensor from quantized linear transformation
        """
        # Dequantize weights during forward pass
        weight_float = self.weight_int8.to(torch.float32) * self.weight_scale
        return F.linear(input, weight_float, self.bias)

    def get_memory_footprint(self):
        """
        Calculate memory footprint in bytes

        Returns:
            int: Total memory usage in bytes
        """
        weight_memory = self.weight_int8.numel() * 1  # 1 byte per int8
        scale_memory = 4  # 4 bytes for float32
        zero_point_memory = 4  # 4 bytes for int32
        bias_memory = self.bias.numel() * 4 if self.bias is not None else 0
        return weight_memory + scale_memory + zero_point_memory + bias_memory

    def get_compression_info(self):
        """
        Get compression statistics compared to float32

        Returns:
            dict: Compression information
        """
        int8_size = self.get_memory_footprint()
        float32_size = self.weight_int8.numel() * 4  # 4 bytes per float32
        if self.bias is not None:
            float32_size += self.bias.numel() * 4

        return {
            'int8_size_bytes': int8_size,
            'float32_size_bytes': float32_size,
            'compression_ratio': float32_size / int8_size,
            'memory_saved_bytes': float32_size - int8_size,
            'memory_saved_percent': (1 - int8_size / float32_size) * 100
        }


def quantize_linear_layers_in_model(model):
    """
    Quantize ALL linear layers in a model to int8 for full on-device deployment

    Args:
        model: PyTorch model to quantize

    Returns:
        dict: Quantization statistics for each layer
    """
    quantization_stats = {}
    total_original_size = 0
    total_quantized_size = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Calculate original size
            original_size = module.weight.numel() * 4  # 4 bytes per float32
            if module.bias is not None:
                original_size += module.bias.numel() * 4

            # Convert to int8
            int8_module = Int8Linear.from_float(module)
            quantized_size = int8_module.get_memory_footprint()

            # Replace module in model
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, int8_module)
            else:
                setattr(model, child_name, int8_module)

            # Record statistics
            quantization_stats[name] = {
                'original_size_bytes': original_size,
                'quantized_size_bytes': quantized_size,
                'compression_ratio': original_size / quantized_size,
                'memory_saved_bytes': original_size - quantized_size,
                'memory_saved_percent': (1 - quantized_size / original_size) * 100
            }

            total_original_size += original_size
            total_quantized_size += quantized_size

            print(f"Quantized {name}: {original_size/1024:.1f}KB → {quantized_size/1024:.1f}KB "
                  f"({original_size/quantized_size:.1f}x compression)")

    # Add summary statistics
    if quantization_stats:
        quantization_stats['_summary'] = {
            'total_layers_quantized': len([k for k in quantization_stats.keys() if not k.startswith('_')]),
            'total_original_size_bytes': total_original_size,
            'total_quantized_size_bytes': total_quantized_size,
            'total_compression_ratio': total_original_size / total_quantized_size if total_quantized_size > 0 else 1.0,
            'total_memory_saved_bytes': total_original_size - total_quantized_size,
            'total_memory_saved_percent': (1 - total_quantized_size / total_original_size) * 100 if total_original_size > 0 else 0
        }

        print(f"\nTotal quantization: {total_original_size/1024:.1f}KB → {total_quantized_size/1024:.1f}KB "
              f"({total_original_size/total_quantized_size:.1f}x compression)")

    return quantization_stats


# Removed redundant load_int8_model_from_checkpoint - use quantize_pico.py instead