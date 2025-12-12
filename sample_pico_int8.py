"""
Sample script for int8 quantized picoGPT models
Dedicated inference with true int8 weights
"""

import os
import pickle
import argparse
import torch
import tiktoken
from contextlib import nullcontext

from model.pico_model import PicoGPTConfig, PicoGPT
from model.pico_model_int8 import Int8Linear


def parse_args():
    parser = argparse.ArgumentParser(description='Sample from FULL int8 quantized picoGPT for on-device deployment')
    parser.add_argument('--out_dir', type=str, default='out-pico-shakespeare-char',
                       help='Output directory containing trained models')
    parser.add_argument('--start', type=str, default="\n", help='Initial prompt or FILE:path')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('--max_new_tokens', type=int, default=500, help='Max new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k sampling')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--dtype', type=str, default='float32', help='Data type for non-quantized operations')
    parser.add_argument('--compile', action='store_true', help='Compile model for faster inference')
    parser.add_argument('--show_stats', action='store_true', help='Show detailed model statistics')
    return parser.parse_args()


def load_int8_model(out_dir):
    """Load FULL int8 quantized model from file"""

    # Try to load existing quantized model
    quantized_model_path = os.path.join(out_dir, 'ckpt_int8.pt')

    if os.path.exists(quantized_model_path):
        print(f"Loading pre-quantized FULL int8 model from {quantized_model_path}")
        return load_existing_quantized_model(quantized_model_path)
    else:
        print(f"Pre-quantized model not found, creating FULL int8 model from checkpoint...")
        return create_quantized_model_on_demand(out_dir)


def load_existing_quantized_model(model_path):
    """Load a previously saved quantized model"""
    checkpoint = torch.load(model_path, map_location='cpu')

    model_args = checkpoint['model_args']
    quantization_stats = checkpoint['quantization_stats']
    state_dict = checkpoint['model_state_dict']

    # Create base model
    gptconf = PicoGPTConfig(**model_args)
    model = PicoGPT(gptconf)

    # Identify quantized layers and replace them
    quantized_layers = set()
    for key in state_dict.keys():
        if key.endswith('.weight_int8'):
            layer_name = key[:-len('.weight_int8')]
            quantized_layers.add(layer_name)

    # Replace linear layers with Int8Linear where needed
    for layer_name in quantized_layers:
        parent_name = '.'.join(layer_name.split('.')[:-1])
        child_name = layer_name.split('.')[-1]

        if parent_name:
            parent = model.get_submodule(parent_name)
            original_layer = getattr(parent, child_name)
        else:
            parent = model
            original_layer = getattr(model, child_name)

        # Create Int8Linear replacement
        int8_layer = Int8Linear(original_layer.in_features, original_layer.out_features,
                               original_layer.bias is not None)

        # Replace the layer
        if parent_name:
            setattr(parent, child_name, int8_layer)
        else:
            setattr(model, child_name, int8_layer)

    # Load state dict
    model.load_state_dict(state_dict)

    print(f"Loaded quantized model with {len(quantized_layers)} int8 layers")
    return model, quantization_stats


def create_quantized_model_on_demand(out_dir):
    """Create FULL int8 quantized model on-the-fly from checkpoint"""
    from quantize_pico import quantize_model_to_int8

    # Find checkpoint
    checkpoint_path = os.path.join(out_dir, 'ckpt_l3_h4_e192_bs32_bl128_lr3e-3_iter8k_baseline.pt')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(out_dir, 'ckpt.pt')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found in {out_dir}")

    # Use full quantization function
    save_path = os.path.join(out_dir, 'quantized_int8_full.pt')
    print(f"Creating FULL int8 quantized model from {checkpoint_path}")

    quantization_stats = quantize_model_to_int8(checkpoint_path, save_path)

    # Load the saved quantized model
    return load_existing_quantized_model(save_path)


def load_tokenizer():
    """Load appropriate tokenizer"""
    # Try character-level tokenizers first
    for dataset in ['shakespeare_char', 'graham_char']:
        meta_path = os.path.join('data', dataset, 'meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            vocab_size = meta['vocab_size']
            encode = lambda s: [meta['stoi'][c] for c in s]
            decode = lambda l: ''.join([meta['itos'][i] for i in l])
            print(f"Using {dataset} tokenizer (vocab_size={vocab_size})")
            return encode, decode, vocab_size

    # Fallback to GPT-2
    print("Using GPT-2 tokenizer")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    return encode, decode, 50304


def analyze_model_stats(model, quantization_stats):
    """Display detailed model statistics"""
    print("\n" + "="*70)
    print("INT8 QUANTIZED MODEL ANALYSIS")
    print("="*70)

    total_params = 0
    int8_params = 0
    total_memory = 0

    print("\nLayer-by-layer breakdown:")
    for name, module in model.named_modules():
        if isinstance(module, Int8Linear):
            params = module.weight_int8.numel()
            memory = module.get_memory_footprint()
            compression_info = module.get_compression_info()

            total_params += params
            int8_params += params
            total_memory += memory

            print(f"✓ {name}:")
            print(f"    {params:,} int8 parameters, {memory/1024:.1f}KB")
            print(f"    Compression: {compression_info['compression_ratio']:.1f}x, "
                  f"Saved: {compression_info['memory_saved_bytes']/1024:.1f}KB")

        elif hasattr(module, 'weight'):
            params = module.weight.numel()
            memory = params * 4  # float32
            total_params += params
            total_memory += memory

            print(f"  {name}: {params:,} float32 parameters, {memory/1024:.1f}KB")

    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Int8 parameters: {int8_params:,} ({int8_params/total_params*100:.1f}%)")
    print(f"  Total memory: {total_memory/1024:.1f}KB ({total_memory/1024/1024:.2f}MB)")

    # Show quantization statistics
    if quantization_stats and '_summary' in quantization_stats:
        summary = quantization_stats['_summary']
        print(f"\nQuantization Impact:")
        print(f"  Layers quantized: {summary['total_layers_quantized']}")
        print(f"  Memory saved: {summary['total_memory_saved_bytes']/1024:.1f}KB")
        print(f"  Overall compression: {summary['total_compression_ratio']:.1f}x")

    print("="*70)


def main():
    args = parse_args()

    print("Int8 Quantized picoGPT Inference")

    # Setup
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Load FULL int8 quantized model
    model, quantization_stats = load_int8_model(args.out_dir)
    model.to(args.device)
    model.eval()

    # Show detailed stats if requested
    if args.show_stats:
        analyze_model_stats(model, quantization_stats)

    # Compile if requested
    if args.compile:
        print("Compiling quantized model...")
        model = torch.compile(model)

    # Load tokenizer
    encode, decode, vocab_size = load_tokenizer()

    # Prepare input
    if args.start.startswith('FILE:'):
        with open(args.start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    else:
        start = args.start

    start_ids = encode(start)
    x = torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...]

    # Generate samples
    for i in range(args.num_samples):
        print(f"\nSample {i+1}:")
        print("-" * 50)

        with torch.no_grad():
            with ctx:
                y = model.generate(x, args.max_new_tokens,
                                 temperature=args.temperature, top_k=args.top_k)
                generated_text = decode(y[0].tolist())
                print(generated_text)

    # Final summary
    print("")
    print("")
    print("INFERENCE SUMMARY")
    print("-" * 50)
    print("")
    print("")

    # Count int8 parameters
    int8_params = sum(m.weight_int8.numel() for m in model.modules() if isinstance(m, Int8Linear))
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Model type: FULL int8 quantized")
    print(f"Int8 parameters: {int8_params:,}/{total_params:,} ({int8_params/total_params*100:.1f}%)")
    print(f"Device: {args.device}")
    print(f"Samples generated: {args.num_samples}")

    if quantization_stats and '_summary' in quantization_stats:
        summary = quantization_stats['_summary']
        print(f"Memory savings: {summary['total_memory_saved_bytes']/1024:.1f}KB "
              f"({summary['total_compression_ratio']:.1f}x compression)")

if __name__ == '__main__':
    main()