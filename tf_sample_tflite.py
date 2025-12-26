"""
Sample from TFLite model (with KV-cache) on PC
This tests if the TFLite model itself has issues or just the device
"""
import os
import pickle
import argparse
import tensorflow as tf
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Sample from TFLite model")
    parser.add_argument("--tflite_path", type=str,
                        default="out-tf-sub-pico-tinystories-tiny-bpe/model.tflite",
                        help="Path to TFLite model")
    parser.add_argument("--start", type=str, default="One day, ",
                        help="Start prompt")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=200,
                        help="Top-k filtering")
    parser.add_argument("--sampling", action="store_true",
                        help="Enable stochastic sampling (default: greedy decode)")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed")
    parser.add_argument("--data_dir", type=str,
                        default="data/tinystories_tiny_bpe",
                        help="Data directory for tokenizer")
    return parser.parse_args()

def main():
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print("="*70)
    print("TFLite Model Sampling (with KV-cache)")
    print("="*70)
    print(f"Model: {args.tflite_path}")
    print(f"Prompt: {repr(args.start)}")
    print(f"Tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Sampling: {'On' if args.sampling else 'Greedy'}")
    print(f"Seed: {args.seed}")
    print("="*70)
    print()

    # Load tokenizer
    from tf_data import CharacterTokenizer
    tokenizer = CharacterTokenizer(args.data_dir)
    print(f"Loaded tokenizer: vocab_size={tokenizer.vocab_size}")

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=args.tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Model inputs: {len(input_details)}")
    for i, inp in enumerate(input_details):
        print(f"  {i}: {inp['name']:20s} shape={inp['shape'].tolist()}")
    print()

    # Load model config
    config_path = os.path.join(os.path.dirname(args.tflite_path), "model_config.pkl")
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    if config.n_layer != 1:
        print(
            f"Warning: expected n_layer=1 for sub-pico tinystories, got {config.n_layer}."
        )

    # Encode prompt
    prompt_tokens = tokenizer.encode(args.start)
    print(f"Prompt tokens: {prompt_tokens}")
    print()

    # Initialize KV caches (one per layer)
    head_dim = config.n_embd // config.n_head
    k_caches = [
        np.zeros((1, config.n_head, config.block_size, head_dim), dtype=np.float32)
        for _ in range(config.n_layer)
    ]
    v_caches = [
        np.zeros((1, config.n_head, config.block_size, head_dim), dtype=np.float32)
        for _ in range(config.n_layer)
    ]

    expected_inputs = 2 * config.n_layer + 2  # token + (k,v)*layers + position
    expected_outputs = 2 * config.n_layer + 1  # logits + (k,v)*layers
    if len(input_details) != expected_inputs or len(output_details) != expected_outputs:
        print(
            f"WARNING: I/O count mismatch (inputs={len(input_details)} expected={expected_inputs}, "
            f"outputs={len(output_details)} expected={expected_outputs}). Check KV-cache export."
        )

    token_input_idx = input_details[0]["index"]
    cache_input_base = 1
    position_input_idx = input_details[cache_input_base + 2 * config.n_layer]["index"]
    logits_output_idx = output_details[0]["index"]
    cache_output_base = 1

    # Process prompt tokens
    print("Processing prompt...")
    cache_position = 0
    for token_id in prompt_tokens:
        # Set inputs
        interpreter.set_tensor(
            token_input_idx, np.array([[token_id]], dtype=np.int32)
        )
        for layer in range(config.n_layer):
            k_idx = input_details[cache_input_base + 2 * layer]["index"]
            v_idx = input_details[cache_input_base + 2 * layer + 1]["index"]
            interpreter.set_tensor(k_idx, k_caches[layer])
            interpreter.set_tensor(v_idx, v_caches[layer])
        interpreter.set_tensor(
            position_input_idx, np.array(cache_position, dtype=np.int32)
        )

        # Run inference
        interpreter.invoke()

        # Get outputs and update caches
        logits = interpreter.get_tensor(logits_output_idx)
        for layer in range(config.n_layer):
            k_new = interpreter.get_tensor(
                output_details[cache_output_base + 2 * layer]["index"]
            )
            v_new = interpreter.get_tensor(
                output_details[cache_output_base + 2 * layer + 1]["index"]
            )
            # Update cache at current position
            k_caches[layer][0, :, cache_position:cache_position+1, :] = k_new
            v_caches[layer][0, :, cache_position:cache_position+1, :] = v_new
        cache_position += 1

    # Print prompt
    print("="*70)
    print(args.start, end='', flush=True)

    # Generate tokens (limit to avoid cache overflow)
    # Total tokens (prompt + generation) must not exceed block_size
    max_generate = min(args.max_new_tokens, config.block_size - cache_position)

    if max_generate <= 0:
        print(f"ERROR: Prompt length ({cache_position}) exceeds or equals block_size ({config.block_size}). Cannot generate.")
        return

    generated_tokens = []
    for i in range(max_generate):
        # Get logits from last token
        logits = interpreter.get_tensor(logits_output_idx)
        logits = logits[0, 0, :]  # Shape: [vocab_size]

        # Apply temperature
        if args.temperature != 1.0:
            logits = logits / args.temperature

        # Top-k filtering
        if args.top_k > 0 and args.top_k < len(logits):
            # Get top-k indices
            top_k_indices = np.argsort(logits)[-args.top_k:]
            # Set others to -inf
            mask = np.ones_like(logits) * -1e10
            mask[top_k_indices] = 0
            logits = logits + mask

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        if args.sampling:
            next_token = np.random.choice(len(probs), p=probs)
        else:
            next_token = int(np.argmax(probs))
        generated_tokens.append(next_token)

        # Print token
        print(tokenizer.itos[next_token].replace(tokenizer.sp_marker, " "),
              end='', flush=True)

        # Update model for next token
        interpreter.set_tensor(
            token_input_idx, np.array([[next_token]], dtype=np.int32)
        )
        for layer in range(config.n_layer):
            k_idx = input_details[cache_input_base + 2 * layer]["index"]
            v_idx = input_details[cache_input_base + 2 * layer + 1]["index"]
            interpreter.set_tensor(k_idx, k_caches[layer])
            interpreter.set_tensor(v_idx, v_caches[layer])
        interpreter.set_tensor(
            position_input_idx, np.array(cache_position, dtype=np.int32)
        )

        # Run inference
        interpreter.invoke()

        # Get outputs and update caches
        for layer in range(config.n_layer):
            k_new = interpreter.get_tensor(
                output_details[cache_output_base + 2 * layer]["index"]
            )
            v_new = interpreter.get_tensor(
                output_details[cache_output_base + 2 * layer + 1]["index"]
            )
            # Update cache (guaranteed to be within bounds due to max_generate limiting)
            k_caches[layer][0, :, cache_position:cache_position+1, :] = k_new
            v_caches[layer][0, :, cache_position:cache_position+1, :] = v_new
        cache_position += 1

    print()
    print()
    print("="*70)
    print(f"Generated {len(generated_tokens)} tokens")
    print("="*70)

if __name__ == "__main__":
    main()
