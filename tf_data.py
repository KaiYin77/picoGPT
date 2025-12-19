"""
Data pipeline for PicoGPT - TensorFlow version
Compatible with existing PyTorch data files (train.bin, val.bin, meta.pkl)
"""

import os
import pickle
import numpy as np
import tensorflow as tf


class CharacterTokenizer:
    """
    Character-level tokenizer compatible with PyTorch version
    Loads from meta.pkl file created by PyTorch data preparation
    """

    def __init__(self, data_dir):
        """
        Load tokenizer from meta.pkl

        Args:
            data_dir: Path to data directory containing meta.pkl
        """
        meta_path = os.path.join(data_dir, 'meta.pkl')

        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Tokenizer metadata not found at {meta_path}. "
                f"Please run the PyTorch data preparation script first."
            )

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        self.vocab_size = meta['vocab_size']
        self.stoi = meta.get('stoi')  # string to index
        self.itos = meta.get('itos')  # index to string
        self.level = meta.get('level', 'char')
        self.encoding = meta.get('encoding', 'latin-1')
        self.merges = meta.get('merges')
        self.bpe_type = meta.get('bpe_type')
        self.sp_marker = meta.get('sp_marker')

        print(
            f"Loaded tokenizer: vocab_size={self.vocab_size}, "
            f"level={self.level}"
        )

    def encode(self, text):
        """
        Encode string to list of token IDs

        Args:
            text: Input string

        Returns:
            List of integer token IDs
        """
        if self.level == 'byte':
            data = text.encode(self.encoding, errors='replace')
            return list(data)
        if self.level == 'bpe':
            return self._encode_bpe(text)
        return [self.stoi[c] for c in text]

    def decode(self, tokens):
        """
        Decode list of token IDs to string

        Args:
            tokens: List of integer token IDs

        Returns:
            Decoded string
        """
        if self.level == 'byte':
            return bytes(tokens).decode(self.encoding, errors='replace')
        if self.level == 'bpe':
            return self._decode_bpe(tokens)
        return ''.join([self.itos[i] for i in tokens])

    def _merge_tokens(self, tokens, pair, new_token):
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(new_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def _encode_bpe(self, text):
        if not self.merges or not self.stoi:
            raise ValueError("BPE tokenizer missing merges or stoi")
        if self.bpe_type == 'tiny_sp' and self.sp_marker:
            text = text.replace(' ', self.sp_marker)
        tokens = list(text)
        for pair in self.merges:
            merged_token = ''.join(pair)
            tokens = self._merge_tokens(tokens, pair, merged_token)
        return [self.stoi[t] for t in tokens]

    def _decode_bpe(self, tokens):
        if not self.itos:
            raise ValueError("BPE tokenizer missing itos")
        text = ''.join(self.itos[i] for i in tokens)
        if self.bpe_type == 'tiny_sp' and self.sp_marker:
            text = text.replace(self.sp_marker, ' ')
        return text


def create_dataset(data_dir, split='train', batch_size=32, block_size=128, shuffle=True):
    """
    Create TensorFlow dataset from binary files (finite dataset for validation)

    Args:
        data_dir: Path to data directory (e.g., 'data/shakespeare_char')
        split: 'train' or 'val'
        batch_size: Batch size
        block_size: Sequence length
        shuffle: Whether to shuffle data

    Returns:
        tf.data.Dataset yielding (input_ids, target_ids) tuples
        Each with shape (batch_size, block_size)
    """
    # Load binary data
    bin_file = os.path.join(data_dir, f'{split}.bin')

    if not os.path.exists(bin_file):
        raise FileNotFoundError(
            f"Data file not found at {bin_file}. "
            f"Please run the PyTorch data preparation script first."
        )

    data = np.memmap(bin_file, dtype=np.uint16, mode='r')
    print(f"Loaded {split} data: {len(data):,} tokens")

    # Convert to int32 (TensorFlow has better support for int32 than uint16)
    data = data.astype(np.int32)

    def data_generator():
        """Generator function for dataset"""
        num_samples = len(data) - block_size
        indices = np.arange(num_samples)

        if shuffle:
            np.random.shuffle(indices)

        for idx in indices:
            x = data[idx:idx+block_size]
            y = data[idx+1:idx+1+block_size]
            yield x, y

    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(block_size,), dtype=tf.int32),
            tf.TensorSpec(shape=(block_size,), dtype=tf.int32)
        )
    )

    # Batch and prefetch
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_infinite_dataset(data_dir, split='train', batch_size=32, block_size=128):
    """
    Create infinite dataset for training (matches PyTorch behavior)

    PyTorch version uses random sampling on each batch.
    This version creates an infinite repeating dataset with random crop positions.

    Args:
        data_dir: Path to data directory
        split: 'train' or 'val'
        batch_size: Batch size
        block_size: Sequence length

    Returns:
        tf.data.Dataset that yields infinite batches of (input_ids, target_ids)
    """
    bin_file = os.path.join(data_dir, f'{split}.bin')

    if not os.path.exists(bin_file):
        raise FileNotFoundError(
            f"Data file not found at {bin_file}. "
            f"Please run the PyTorch data preparation script first."
        )

    # Load as memmap for memory efficiency
    data = np.memmap(bin_file, dtype=np.uint16, mode='r')
    data_len = len(data)

    print(f"Loaded {split} data: {data_len:,} tokens (infinite mode)")

    def random_crop_generator():
        """Infinite generator with random crops (matches PyTorch)"""
        while True:
            # Random starting position
            idx = np.random.randint(0, data_len - block_size)
            x = data[idx:idx+block_size].astype(np.int32)
            y = data[idx+1:idx+1+block_size].astype(np.int32)
            yield x, y

    dataset = tf.data.Dataset.from_generator(
        random_crop_generator,
        output_signature=(
            tf.TensorSpec(shape=(block_size,), dtype=tf.int32),
            tf.TensorSpec(shape=(block_size,), dtype=tf.int32)
        )
    )

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def get_batch(dataset_iter):
    """
    Get a single batch from dataset iterator

    Args:
        dataset_iter: Iterator from tf.data.Dataset

    Returns:
        Tuple of (input_ids, target_ids) tensors
    """
    return next(dataset_iter)
