"""
Pico GPT model with int8 quantization support
Optimized for ~200-300K parameters
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.quantization import QuantStub, DeQuantStub

class LayerNorm(nn.Module):
    """LayerNorm without bias for parameter efficiency"""

    def __init__(self, ndim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, None, 1e-5)

class CausalSelfAttention(nn.Module):
    """Compact self-attention for tiny models"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Combined QKV projection for efficiency
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Use flash attention if available
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # Causal mask
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        # QKV projections
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Attention
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """Compact MLP block"""

    def __init__(self, config):
        super().__init__()
        # Smaller expansion ratio for tiny models
        hidden_dim = 2 * config.n_embd  # 2x instead of 4x
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.relu = nn.ReLU()  # CIMv3 only supports ReLU activation
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """
    CIMv3 Hardware-Compatible Transformer block
    Sequence: MHA → FFN → LN2 (removing pre-normalization LN1)
    Matches CIMv3 execution: PIPE mode (MHA) → PARL mode (FFN) → LN mode
    """

    def __init__(self, config):
        super().__init__()
        # Remove LN1 for CIMv3 hardware compatibility
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        # Keep only LN2 for post-FFN normalization
        self.ln_2 = LayerNorm(config.n_embd)

    def forward(self, x):
        # CIMv3 sequence: MHA → FFN → LN2
        # MHA with residual connection (no pre-norm)
        x = x + self.attn(x)
        # FFN followed by LN2 and residual
        x = self.ln_2(x + self.mlp(x))
        return x

@dataclass
class PicoGPTConfig:
    block_size: int = 128
    vocab_size: int = 65  # Shakespeare character set
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 128  # CIMv3 compatible: 128/256/512 only
    dropout: float = 0.1
    bias: bool = False

    def __post_init__(self):
        """Validate CIMv3 hardware constraints"""
        # CIMv3 Reg #1 constraint: d_model (n_embd) must be 128, 256, or 512
        valid_d_models = [128, 256, 512]
        if self.n_embd not in valid_d_models:
            raise ValueError(f"CIMv3 constraint violation: n_embd must be one of {valid_d_models}, got {self.n_embd}")

        # PyTorch constraint: d_model % num_heads == 0
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"CIMv3 constraint violation: n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")

        # CIMv3 optimized for d_head = 64
        d_head = self.n_embd // self.n_head
        if d_head != 64 and self.n_embd >= 256:
            print(f"Warning: CIMv3 is optimized for d_head=64, got {d_head}. Consider adjusting n_head.")

        # CIMv3 max sequence length constraint (reg.window)
        if self.block_size > 128:
            print(f"Warning: CIMv3 max supported seq_len is 128, got {self.block_size}. May cause hardware limitations.")

        # Bias-free operations preferred for CIMv3 INT8 GEMM
        if self.bias:
            print("Warning: CIMv3 prefers bias=False for INT8 GEMM operations.")

        print(f"CIMv3 Config Validation: ✓ d_model={self.n_embd}, n_head={self.n_head}, d_head={d_head}")

class PicoGPT(nn.Module):
    """Pico GPT model optimized for int8 quantization"""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Quantization stubs for int8 support
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying for parameter efficiency
        # Share the same parameter between input embedding and output projection
        self.lm_head.weight = self.transformer.wte.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Special scaled init for projection layers
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"

        # Quantize inputs
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Token and position embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Apply quantization stub
        x = self.quant(x)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm and projection
        x = self.transformer.ln_f(x)

        # Dequantize before final projection
        x = self.dequant(x)

        if targets is not None:
            # Training mode: compute loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference mode: only compute logits for last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        """Crop the block size if needed"""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """Load pretrained model (not applicable for tiny model, but kept for compatibility)"""
        assert model_type == 'pico-gpt'

        config = PicoGPTConfig()
        if override_args is not None:
            for k, v in override_args.items():
                setattr(config, k, v)

        model = cls(config)
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Configure optimizer with weight decay"""
        # Separate parameters that should and shouldn't decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Validate that we've considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}

        # Filter out parameters that don't exist (due to weight tying)
        decay = {pn for pn in decay if pn in param_dict}
        no_decay = {pn for pn in no_decay if pn in param_dict}

        # Validate parameter sets
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {inter_params} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters {param_dict.keys() - union_params} were not separated into either decay/no_decay set!"

        # Create optimizer groups
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate new tokens"""
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # Forward pass
            logits, _ = self(idx_cond)
            # Apply temperature and top-k sampling
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx