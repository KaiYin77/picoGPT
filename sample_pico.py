"""
Sample script for picoGPT models
Clean FP32 inference only - use sample_pico_int8.py for quantized models
"""

import os
import pickle
import argparse
import torch
import tiktoken
from contextlib import nullcontext

from model.pico_model import PicoGPTConfig, PicoGPT

def parse_args():
    parser = argparse.ArgumentParser(description='Sample from picoGPT FP32 model')
    parser.add_argument('--out_dir', type=str, default='out-pico-shakespeare-char', help='Output directory')
    parser.add_argument('--start', type=str, default="\n", help='Initial prompt or FILE:path')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to generate')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Max new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k sampling (None to disable)')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--dtype', type=str, default='float32', help='Data type')
    parser.add_argument('--compile', action='store_true', help='Compile model')
    return parser.parse_args()

def load_model(args):
    """Load FP32 model from checkpoint"""

    # Determine checkpoint path
    ckpt_path = os.path.join(args.out_dir, 'ckpt_l3_h4_e192_bs32_bl128_lr3e-3_iter8k_baseline.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')

    print(f"Loading FP32 model from {ckpt_path}")

    # Load tokenizer
    meta_path = os.path.join('data', 'shakespeare_char', 'meta.pkl')
    if not os.path.exists(meta_path):
        meta_path = os.path.join('data', 'graham_char', 'meta.pkl')

    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta['vocab_size']
        encode = lambda s: [meta['stoi'][c] for c in s]
        decode = lambda l: ''.join([meta['itos'][i] for i in l])
        print(f"Using character-level tokenizer (vocab_size={vocab_size})")
    else:
        print("Using GPT-2 tokenizer")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
        vocab_size = 50304

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    gptconf = PicoGPTConfig(**checkpoint['model_args'])
    model = PicoGPT(gptconf)

    # Load state dict and clean any artifacts
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

    if quantization_keys:
        print(f"Cleaned {len(quantization_keys)} quantization artifacts from checkpoint")

    model.load_state_dict(state_dict, strict=False)
    model.to(args.device)

    print(f"Loaded FP32 model with {model.get_num_params():,} parameters")
    return model, encode, decode

def main():
    args = parse_args()

    # Setup
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed) if args.device.startswith('cuda') else None
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Load model
    model, encode, decode = load_model(args)
    model.eval()

    # Compile if requested
    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)

    # Prepare start string
    if args.start.startswith('FILE:'):
        with open(args.start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    else:
        start = args.start

    start_ids = encode(start)
    x = torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...]

    # Generate samples
    print(f"\nGenerating {args.num_samples} samples...")

    for i in range(args.num_samples):
        print(f"Sample {i+1}:")
        print("-" * 40)

        with torch.no_grad():
            with ctx:
                y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
                generated_text = decode(y[0].tolist())
                print(generated_text)

    # Print model info
    print(f"\nModel Summary:")
    print(f"  Parameters: {model.get_num_params() if hasattr(model, 'get_num_params') else sum(p.numel() for p in model.parameters()):,}")
    print(f"  Model type: FP32")
    print(f"  Device: {args.device}")
    print(f"  Generated {args.num_samples} samples successfully!")
    print(f"\n💡 For quantized inference, use: python sample_pico_int8.py")

if __name__ == '__main__':
    main()