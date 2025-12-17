"""
Configuration for PicoGPT model - TensorFlow version
Matches PyTorch configuration with CIMv3 hardware constraints
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PicoGPTConfig:
    """Configuration for PicoGPT model"""

    # Model architecture
    block_size: int = 128  # Maximum sequence length
    vocab_size: int = 65   # Vocabulary size (Shakespeare char set)
    n_layer: int = 3       # Number of transformer layers
    n_head: int = 4        # Number of attention heads
    n_embd: int = 128      # Embedding dimension
    dropout: float = 0.1   # Dropout probability
    bias: bool = False     # Whether to use bias in Linear layers (CIMv3: prefer False)

    def __post_init__(self):
        """Validate configuration against CIMv3 hardware constraints"""

        # CIMv3 constraint: n_embd must be 128, 256, or 512
        valid_d_models = [128, 256, 512]
        if self.n_embd not in valid_d_models:
            raise ValueError(
                f"CIMv3 Hardware Constraint: n_embd must be in {valid_d_models}, got {self.n_embd}"
            )

        # Ensure n_embd is divisible by n_head
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )

        # CIMv3 recommendation: d_head = 64 for optimal performance
        d_head = self.n_embd // self.n_head
        if d_head != 64:
            print(f"Warning: d_head = {d_head}, CIMv3 optimized for d_head = 64")

        # CIMv3 constraint: block_size should not exceed 128
        if self.block_size > 128:
            print(f"Warning: block_size = {self.block_size} exceeds CIMv3 recommendation of 128")

        # CIMv3 constraint: bias should be False for efficiency
        if self.bias:
            print("Warning: CIMv3 prefers bias=False for parameter efficiency")

    @property
    def head_dim(self):
        """Dimension per attention head"""
        return self.n_embd // self.n_head
