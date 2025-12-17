"""
Training script for PicoGPT - TensorFlow version
Matches PyTorch training behavior with TF best practices
"""

import os
import time
import pickle
import argparse
import importlib.util
import tensorflow as tf
import numpy as np

from tf_model.config import PicoGPTConfig
from tf_model.model import PicoGPT
from tf_data import create_infinite_dataset, CharacterTokenizer


def load_config_from_file(config_path):
    """Load configuration from Python file"""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


def get_lr(iteration, config):
    """
    Cosine learning rate schedule with linear warmup

    Args:
        iteration: Current iteration
        config: Training configuration

    Returns:
        Current learning rate
    """
    # Linear warmup
    if iteration < config.warmup_iters:
        return config.learning_rate * iteration / config.warmup_iters

    # After decay iterations, use minimum LR
    if iteration > config.lr_decay_iters:
        return config.min_lr

    # Cosine decay
    decay_ratio = (iteration - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@tf.function
def train_step(model, x, y, optimizer, grad_clip):
    """
    Single training step with gradient clipping

    Args:
        model: PicoGPT model
        x: Input token IDs (B, T)
        y: Target token IDs (B, T)
        optimizer: Optimizer
        grad_clip: Gradient clipping threshold

    Returns:
        loss: Scalar loss value
    """
    with tf.GradientTape() as tape:
        logits, loss = model(x, y, training=True)

    # Compute gradients
    # In Keras 3, LossScaleOptimizer handles scaling automatically
    gradients = tape.gradient(loss, model.trainable_variables)

    # Gradient clipping
    if grad_clip > 0.0:
        gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)

    # Apply gradients (optimizer handles loss scaling internally in Keras 3)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


@tf.function
def eval_step(model, x, y):
    """
    Single evaluation step

    Args:
        model: PicoGPT model
        x: Input token IDs (B, T)
        y: Target token IDs (B, T)

    Returns:
        loss: Scalar loss value
    """
    logits, loss = model(x, y, training=False)
    return loss


def estimate_loss(model, train_dataset, val_dataset, eval_iters):
    """
    Estimate loss on train and val sets

    Args:
        model: PicoGPT model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        eval_iters: Number of iterations to average over

    Returns:
        Dictionary with 'train' and 'val' loss values
    """
    losses = {'train': [], 'val': []}

    # Train loss
    train_iter = iter(train_dataset)
    for _ in range(eval_iters):
        try:
            x, y = next(train_iter)
            loss = eval_step(model, x, y)
            losses['train'].append(loss.numpy())
        except StopIteration:
            train_iter = iter(train_dataset)

    # Val loss
    val_iter = iter(val_dataset)
    for _ in range(eval_iters):
        try:
            x, y = next(val_iter)
            loss = eval_step(model, x, y)
            losses['val'].append(loss.numpy())
        except StopIteration:
            val_iter = iter(val_dataset)

    return {
        'train': np.mean(losses['train']),
        'val': np.mean(losses['val'])
    }


def main():
    parser = argparse.ArgumentParser(description='Train PicoGPT with TensorFlow')
    parser.add_argument('--config', type=str, default='tf_config/train_pico_shakespeare_char.py',
                       help='Path to training config file')
    args = parser.parse_args()

    # Load configuration
    print(f"Loading config from {args.config}")
    config = load_config_from_file(args.config)

    # Set mixed precision policy if enabled
    if config.mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f'Mixed precision enabled: {policy.name}')
        print('Compute dtype:', policy.compute_dtype)
        print('Variable dtype:', policy.variable_dtype)

    # Create output directory
    os.makedirs(config.out_dir, exist_ok=True)

    # Load tokenizer
    data_dir = os.path.join('data', config.dataset)
    tokenizer = CharacterTokenizer(data_dir)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = create_infinite_dataset(
        data_dir,
        split='train',
        batch_size=config.batch_size,
        block_size=config.block_size
    )

    val_dataset = create_infinite_dataset(
        data_dir,
        split='val',
        batch_size=config.batch_size,
        block_size=config.block_size
    )

    # Create model configuration
    model_config = PicoGPTConfig(
        block_size=config.block_size,
        vocab_size=tokenizer.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        bias=config.bias
    )

    print("\nModel configuration:")
    print(f"  block_size: {model_config.block_size}")
    print(f"  vocab_size: {model_config.vocab_size}")
    print(f"  n_layer: {model_config.n_layer}")
    print(f"  n_head: {model_config.n_head}")
    print(f"  n_embd: {model_config.n_embd}")
    print(f"  dropout: {model_config.dropout}")
    print(f"  bias: {model_config.bias}")

    # Create model
    model = PicoGPT(model_config)

    # Build model by running a dummy forward pass
    dummy_input = tf.zeros((1, config.block_size), dtype=tf.int32)
    _ = model(dummy_input, training=False)

    print(f'\nModel parameters: {model.get_num_params():,}')
    print(f'Non-embedding parameters: {model.get_num_params(non_embedding=True):,}')

    # Create optimizer
    # Keras 3 handles loss scaling automatically with mixed_precision policy
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        beta_1=config.beta1,
        beta_2=config.beta2,
        epsilon=1e-8
    )

    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")

    train_iter = iter(train_dataset)
    best_val_loss = float('inf')
    t0 = time.time()

    for iteration in range(config.max_iters):
        # Update learning rate
        lr = get_lr(iteration, config)
        optimizer.learning_rate.assign(lr)

        # Evaluation
        if iteration % config.eval_interval == 0 or iteration == config.max_iters - 1:
            losses = estimate_loss(model, train_dataset, val_dataset, config.eval_iters)
            print(f"iter {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # Save checkpoint
            if losses['val'] < best_val_loss or config.always_save_checkpoint:
                best_val_loss = losses['val']

                # Save model
                checkpoint_path = os.path.join(config.out_dir, 'model.keras')
                model.save(checkpoint_path)  # Keras 3 infers format from extension

                # Save config
                config_path = os.path.join(config.out_dir, 'model_config.pkl')
                with open(config_path, 'wb') as f:
                    pickle.dump(model_config, f)

                print(f"  └─ Saved checkpoint to {config.out_dir} (val_loss: {losses['val']:.4f})")

        # Training step
        x, y = next(train_iter)
        loss = train_step(model, x, y, optimizer, config.grad_clip)

        # Logging
        if iteration % config.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            loss_val = loss.numpy()
            print(f"iter {iteration}: loss {loss_val:.4f}, time {dt*1000/config.log_interval:.2f}ms/iter, lr {lr:.6f}")

    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {config.out_dir}")


if __name__ == '__main__':
    main()
