"""
Prepare the TinyStories dataset with a tiny BPE tokenizer.
Saves train.bin, val.bin containing token ids, and meta.pkl with tokenizer info.

Dataset: https://huggingface.co/datasets/roneneldan/TinyStories
Version: default (main branch)

TinyStories is a dataset of ~2M short stories generated with GPT-3.5/4,
specifically designed for training small language models with simple vocabulary.
"""
import os
import pickle
import numpy as np
from collections import defaultdict
import time

# BPE config
TARGET_VOCAB_SIZE = 256  # adjust as needed for compression vs. size
SP_MARKER = "\u2581"  # SentencePiece-style space marker

# Dataset config
MAX_TRAIN_SAMPLES = 100000  # Limit samples to avoid memory issues (None = all)
MAX_VAL_SAMPLES = 10000     # Limit validation samples
BPE_TRAIN_SAMPLES = 10000   # Limit BPE training to smaller subset for speed


def get_pair_stats(tokens):
    """Get statistics of adjacent token pairs."""
    stats = defaultdict(int)
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        stats[pair] += 1
    return stats


def merge_tokens(tokens, pair, new_token):
    """Merge all occurrences of a token pair into a new token."""
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


def train_bpe_optimized(text, target_vocab_size):
    """
    Optimized BPE training with incremental pair statistics updates.
    Significantly faster than the naive approach for large text.
    """
    print(f"Training BPE on {len(text):,} characters...")
    tokens = list(text.replace(" ", SP_MARKER))
    vocab = sorted(set(tokens))
    vocab_set = set(vocab)
    merges = []

    if target_vocab_size <= len(vocab):
        return vocab, merges

    print(f"Initial vocab size: {len(vocab)}")

    # Initial pair statistics
    print("Computing initial pair statistics...")
    start_time = time.time()
    stats = get_pair_stats(tokens)
    print(f"  Done in {time.time() - start_time:.2f}s")

    iteration = 0
    while len(vocab) < target_vocab_size:
        iteration += 1

        if not stats:
            print("No more pairs to merge")
            break

        # Find most frequent pair
        best_pair = max(stats, key=stats.get)
        best_count = stats[best_pair]
        new_token = "".join(best_pair)

        if new_token in vocab_set:
            print(f"Token '{new_token}' already exists, stopping")
            break

        # Progress reporting
        if len(vocab) % 10 == 0 or iteration <= 5:
            elapsed = time.time() - start_time
            rate = iteration / elapsed if elapsed > 0 else 0
            remaining = (target_vocab_size - len(vocab)) / rate if rate > 0 else 0
            print(f"  Vocab: {len(vocab)}/{target_vocab_size} | "
                  f"Merging '{best_pair[0]}{best_pair[1]}' ({best_count}x) | "
                  f"Speed: {rate:.1f} merges/s | ETA: {remaining:.0f}s")

        # Merge tokens efficiently with incremental stats update
        new_tokens = []
        i = 0

        # Track pairs affected by this merge for incremental update
        pairs_to_remove = defaultdict(int)
        pairs_to_add = defaultdict(int)

        while i < len(tokens):
            # Check if we can merge at position i
            if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                # Record pairs that will be removed
                if i > 0:
                    old_left_pair = (tokens[i - 1], tokens[i])
                    pairs_to_remove[old_left_pair] += 1
                if i + 2 < len(tokens):
                    old_right_pair = (tokens[i + 1], tokens[i + 2])
                    pairs_to_remove[old_right_pair] += 1

                # Add new token
                new_tokens.append(new_token)

                # Record new pairs that will be created
                if len(new_tokens) > 1:
                    new_left_pair = (new_tokens[-2], new_token)
                    pairs_to_add[new_left_pair] += 1
                if i + 2 < len(tokens):
                    new_right_pair = (new_token, tokens[i + 2])
                    pairs_to_add[new_right_pair] += 1

                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        # Update statistics incrementally
        # Remove the merged pair
        del stats[best_pair]

        # Update affected pairs
        for pair, count in pairs_to_remove.items():
            stats[pair] = max(0, stats[pair] - count)
            if stats[pair] == 0:
                del stats[pair]

        for pair, count in pairs_to_add.items():
            stats[pair] += count

        tokens = new_tokens
        merges.append(best_pair)
        vocab.append(new_token)
        vocab_set.add(new_token)

    total_time = time.time() - start_time
    print(f"\n✓ BPE training completed in {total_time:.1f}s")
    return vocab, merges


def train_bpe(text, target_vocab_size):
    """Wrapper to use optimized BPE training."""
    return train_bpe_optimized(text, target_vocab_size)


def encode(text, merges, stoi):
    tokens = list(text.replace(" ", SP_MARKER))
    for pair in merges:
        merged_token = "".join(pair)
        tokens = merge_tokens(tokens, pair, merged_token)

    # Filter unknown tokens (special characters not in vocab)
    encoded = []
    for t in tokens:
        if t in stoi:
            encoded.append(stoi[t])
        else:
            # Skip unknown character (or map to a default token)
            # Uncomment next line to see warnings:
            # print(f"Warning: Unknown character '{t}' (U+{ord(t):04X}), skipping")
            pass
    return encoded


def decode(ids, itos):
    text = "".join(itos[i] for i in ids)
    return text.replace(SP_MARKER, " ")


def clean_text(text):
    """
    Clean text to ASCII-only (remove special Unicode characters).
    Keeps: letters, digits, basic punctuation, newlines
    """
    # Define allowed characters
    allowed_chars = set(
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '0123456789'
        ' .,!?;:\'"()-\n\t'
    )

    cleaned = []
    removed_count = {}

    for char in text:
        if char in allowed_chars:
            cleaned.append(char)
        else:
            # Track removed characters for reporting
            removed_count[char] = removed_count.get(char, 0) + 1

    if removed_count:
        print(f"Filtered {sum(removed_count.values()):,} special characters:")
        # Show top 10 most common removed characters
        top_removed = sorted(removed_count.items(), key=lambda x: x[1], reverse=True)[:10]
        for char, count in top_removed:
            print(f"  '{char}' (U+{ord(char):04X}): {count:,} times")

    return ''.join(cleaned)


# Download and prepare TinyStories dataset
print("="*60)
print("TinyStories Dataset Preparation")
print("="*60)
print("Loading TinyStories dataset from HuggingFace...")
print("This may take a few minutes on first run (dataset will be cached)")
print()

try:
    from datasets import load_dataset

    # Load TinyStories dataset
    # The dataset has 'train' and 'validation' splits
    dataset = load_dataset("roneneldan/TinyStories")

    print(f"✓ Dataset loaded successfully!")
    print(f"  Train samples: {len(dataset['train']):,}")
    print(f"  Validation samples: {len(dataset['validation']):,}")
    print()

    # Limit samples to avoid memory issues
    if MAX_TRAIN_SAMPLES and len(dataset['train']) > MAX_TRAIN_SAMPLES:
        print(f"Limiting train samples to {MAX_TRAIN_SAMPLES:,}")
        train_dataset = dataset['train'].select(range(MAX_TRAIN_SAMPLES))
    else:
        train_dataset = dataset['train']

    if MAX_VAL_SAMPLES and len(dataset['validation']) > MAX_VAL_SAMPLES:
        print(f"Limiting validation samples to {MAX_VAL_SAMPLES:,}")
        val_dataset = dataset['validation'].select(range(MAX_VAL_SAMPLES))
    else:
        val_dataset = dataset['validation']

    # Extract text from stories
    print("Extracting text from stories...")
    train_data = "\n".join(train_dataset['text'])
    val_data = "\n".join(val_dataset['text'])

except ImportError:
    print("="*60)
    print("ERROR: datasets library not installed!")
    print("="*60)
    print("Install with:")
    print("  pip install datasets")
    print("  or: uv pip install datasets")
    print()
    exit(1)
except Exception as e:
    print(f"ERROR: Failed to load dataset: {e}")
    print()
    print("Fallback: trying to read from local input.txt file...")
    input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
    if not os.path.exists(input_file_path):
        print(f"ERROR: {input_file_path} not found!")
        exit(1)
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = f.read()
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

print()
print(f"Dataset statistics (before cleaning):")
print(f"  Train: {len(train_data):,} characters")
print(f"  Val:   {len(val_data):,} characters")
print()

# Clean text to remove special Unicode characters
print("Cleaning text (removing non-ASCII characters)...")
train_data = clean_text(train_data)
val_data = clean_text(val_data)

print()
print(f"Dataset statistics (after cleaning):")
print(f"  Train: {len(train_data):,} characters")
print(f"  Val:   {len(val_data):,} characters")
print()

# Use smaller sample for BPE training (faster) but encode full dataset
print(f"Preparing BPE training sample ({BPE_TRAIN_SAMPLES:,} samples)...")
if BPE_TRAIN_SAMPLES and BPE_TRAIN_SAMPLES < MAX_TRAIN_SAMPLES:
    try:
        from datasets import load_dataset
        bpe_dataset = dataset['train'].select(range(BPE_TRAIN_SAMPLES))
        bpe_train_data = "\n".join(bpe_dataset['text'])
        bpe_train_data = clean_text(bpe_train_data)
        print(f"  BPE training sample: {len(bpe_train_data):,} characters")
    except:
        # Fallback: use subset of train_data
        char_limit = int(len(train_data) * BPE_TRAIN_SAMPLES / MAX_TRAIN_SAMPLES)
        bpe_train_data = train_data[:char_limit]
        print(f"  BPE training sample: {len(bpe_train_data):,} characters (subset)")
else:
    bpe_train_data = train_data
    print(f"  Using full training data for BPE")

print()

# Train BPE on smaller training sample (much faster!)
print("Training BPE tokenizer...")
vocab, merges = train_bpe(bpe_train_data, TARGET_VOCAB_SIZE)
itos = {i: t for i, t in enumerate(vocab)}
stoi = {t: i for i, t in itos.items()}

print()
print(f"✓ BPE training complete!")
print(f"  Vocabulary size: {len(vocab):,}")
print(f"  Number of merges: {len(merges):,}")
print()

# Encode train/val
print("Encoding train/val splits...")
train_ids = encode(train_data, merges, stoi)
val_ids = encode(val_data, merges, stoi)

print(f"✓ Encoding complete!")
print(f"  Train: {len(train_ids):,} tokens")
print(f"  Val:   {len(val_ids):,} tokens")
print()

# Export to bin files
output_dir = os.path.dirname(__file__)
if not output_dir:
    output_dir = "."

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(output_dir, "train.bin"))
val_ids.tofile(os.path.join(output_dir, "val.bin"))

# Save the meta information
meta = {
    "vocab_size": len(vocab),
    "itos": itos,
    "stoi": stoi,
    "merges": merges,
    "level": "bpe",
    "bpe_type": "tiny_sp",
    "sp_marker": SP_MARKER,
    "dataset": "TinyStories",
    "source": "https://huggingface.co/datasets/roneneldan/TinyStories",
    "max_train_samples": MAX_TRAIN_SAMPLES,
    "max_val_samples": MAX_VAL_SAMPLES,
}
with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("="*60)
print("✓ Dataset preparation complete!")
print("="*60)
print(f"Output directory: {os.path.abspath(output_dir)}")
print()
print("Files created:")
print(f"  ✓ train.bin  ({len(train_ids):,} tokens)")
print(f"  ✓ val.bin    ({len(val_ids):,} tokens)")
print(f"  ✓ meta.pkl   (vocab + tokenizer)")
print()
print("Next steps:")
print(f"  cd {os.path.abspath(os.path.join(output_dir, '../..'))} ")
print(f"  python train_tf.py data/tinystories_tiny_bpe --out_dir=out-tf-tinystories")
print("="*60)
