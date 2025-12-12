"""
Sample script for tiny quantized GPT models
Supports both regular and quantized model inference
"""

import os
import pickle
import argparse
import torch
import tiktoken
from contextlib import nullcontext

from model.pico_model import PicoGPTConfig, PicoGPT
from quantization import load_quantized_model, benchmark_model_inference

def parse_args():
    parser = argparse.ArgumentParser(description='Sample from tiny GPT model')
    parser.add_argument('--out_dir', type=str, default='out-tiny', help='Output directory')
    parser.add_argument('--start', type=str, default="\n", help='Initial prompt or FILE:path')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('--max_new_tokens', type=int, default=500, help='Max new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k sampling (None to disable)')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type')
    parser.add_argument('--compile', action='store_true', help='Compile model')
    parser.add_argument('--use_quantized', action='store_true', help='Use quantized model')
    parser.add_argument('--quantized_model_path', type=str, default=None, help='Path to quantized model')
    parser.add_argument('--benchmark', action='store_true', help='Run inference benchmark')
    return parser.parse_args()

def load_model(args):
    """Load model (quantized or regular)"""

    # Determine paths
    ckpt_path = os.path.join(args.out_dir, 'ckpt_l3_h4_e192_bs32_bl128_lr3e-3_iter8k_baseline.pt')
    quantized_path = args.quantized_model_path or os.path.join(args.out_dir, 'quantized_model.pt')

    # Load meta information
    meta_path = os.path.join('data', 'graham_char', 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta['vocab_size']
        encode = lambda s: [meta['stoi'][c] for c in s]
        decode = lambda l: ''.join([meta['itos'][i] for i in l])
    else:
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
        vocab_size = 50304

    if args.use_quantized and os.path.exists(quantized_path):
        print(f"Loading quantized model from {quantized_path}")

        # Load checkpoint to get model configuration
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            model_args = checkpoint['model_args']
        else:
            # Default tiny model args
            model_args = {
                'n_layer': 3,
                'n_head': 4,
                'n_embd': 192,
                'block_size': 128,
                'bias': False,
                'vocab_size': vocab_size,
                'dropout': 0.0,
            }

        # Create config
        gptconf = PicoGPTConfig(**model_args)

        # Load quantized model
        model = load_quantized_model(PicoGPT, gptconf, quantized_path, args.device)
        print("Loaded quantized model successfully")

    else:
        print(f"Loading regular model from {ckpt_path}")

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        gptconf = PicoGPTConfig(**checkpoint['model_args'])
        model = PicoGPT(gptconf)

        # Load state dict
        state_dict = checkpoint['model']

        # Remove DDP prefix
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        # Remove quantization-related parameters
        quantization_keys = []
        for k in state_dict.keys():
            if any(quant_keyword in k for quant_keyword in [
                'activation_post_process', 'weight_fake_quant', 'fake_quant_enabled',
                'observer_enabled', 'scale', 'zero_point', 'min_val', 'max_val', 'eps'
            ]):
                quantization_keys.append(k)

        for k in quantization_keys:
            state_dict.pop(k)

        print(f"Removed {len(quantization_keys)} quantization-related parameters")

        model.load_state_dict(state_dict, strict=False)

        model.to(args.device)
        print("Loaded regular model successfully")

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

    # Benchmark if requested
    if args.benchmark:
        print("\nRunning inference benchmark...")
        avg_time = benchmark_model_inference(model, x, args.device, num_runs=100)
        print(f"Average inference time: {avg_time:.2f} ms")
        print(f"Tokens per second: {len(start_ids) / (avg_time / 1000):.1f}")

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
    print(f"\nModel info:")
    print(f"  Parameters: {model.get_num_params() if hasattr(model, 'get_num_params') else sum(p.numel() for p in model.parameters()):,}")
    print(f"  Model type: {'Quantized' if args.use_quantized else 'Regular'}")
    print(f"  Device: {args.device}")

if __name__ == '__main__':
    main()