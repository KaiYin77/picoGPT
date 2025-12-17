"""
Sampling script for PicoGPT - TensorFlow version
Generate text from trained model
"""

import os
import pickle
import argparse
import tensorflow as tf
import numpy as np

from tf_model.model import PicoGPT
from tf_data import CharacterTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from PicoGPT TensorFlow model")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="out-tf-pico-shakespeare-char",
        help="Output directory containing trained model",
    )
    parser.add_argument(
        "--start", type=str, default="\n", help="Start prompt for generation"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1, help="Number of samples to generate"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Number of tokens to generate per sample",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more random)",
    )
    parser.add_argument(
        "--top_k", type=int, default=200, help="Top-k filtering (None to disable)"
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/shakespeare_char",
        help="Data directory for tokenizer",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seeds for reproducibility
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    model_weights_path = os.path.join(args.out_dir, "model.keras")
    config_path = os.path.join(args.out_dir, "model_config.pkl")

    if not os.path.exists(model_weights_path) or not os.path.exists(config_path):
        print(f"Error: Model files not found in {args.out_dir}")
        print(f"Expected 'model.keras' and 'model_config.pkl'.")
        print(f"Please train a model first using tf_train.py")
        return

    # 1. Load configuration
    print(f"\nLoading model configuration from {config_path}...")
    with open(config_path, "rb") as f:
        model_config = pickle.load(f)

    # 2. Create model from configuration
    print("Creating model from configuration...")
    model = PicoGPT(model_config)

    # 3. Build model with a dummy pass before loading weights
    print("Building model before loading weights...")
    dummy_input = tf.zeros((1, model.config.block_size), dtype=tf.int32)
    _ = model(dummy_input, training=False)

    # 4. Load weights into the constructed model
    print(f"Loading weights from {model_weights_path}...")
    model.load_weights(model_weights_path)

    print(f"Model loaded successfully!")

    # Load tokenizer (needed before building model)
    print(f"\nLoading tokenizer from {args.data_dir}...")
    tokenizer = CharacterTokenizer(args.data_dir)

    # Build the model by running a dummy forward pass
    print("Building model...")
    dummy_input = tf.zeros((1, model.config.block_size), dtype=tf.int32)
    _ = model(dummy_input, training=False)

    print(f"Model built! Parameters: {model.get_num_params():,}")

    # Print generation settings

    print("=" * 60)
    print("PicoGPT Text Generation - TensorFlow")
    print("=" * 60)
    print(f"  Start prompt: {repr(args.start)}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Tokens per sample: {args.max_new_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Random seed: {args.seed}")

    # Encode start string
    start_ids = tokenizer.encode(args.start)
    x = tf.constant([start_ids], dtype=tf.int32)

    for i in range(args.num_samples):
        print(f"{'='*60}")
        print(f"Sample {i+1}/{args.num_samples}")
        print(f"{'='*60}")

        # Generate
        y = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        # Decode
        generated_text = tokenizer.decode(y[0].numpy().tolist())
        print(generated_text)
        print()

if __name__ == "__main__":
    main()
