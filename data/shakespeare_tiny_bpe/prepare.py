"""
Prepare the Shakespeare dataset with a tiny BPE tokenizer.
Saves train.bin, val.bin containing token ids, and meta.pkl with tokenizer info.
"""
import os
import pickle
import numpy as np

# BPE config
TARGET_VOCAB_SIZE = 256  # adjust as needed for compression vs. size
SP_MARKER = "\u2581"  # SentencePiece-style space marker


def get_pair_stats(tokens):
    stats = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        stats[pair] = stats.get(pair, 0) + 1
    return stats


def merge_tokens(tokens, pair, new_token):
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


def train_bpe(text, target_vocab_size):
    tokens = list(text.replace(" ", SP_MARKER))
    vocab = sorted(set(tokens))
    vocab_set = set(vocab)
    merges = []

    if target_vocab_size <= len(vocab):
        return vocab, merges

    while len(vocab) < target_vocab_size:
        stats = get_pair_stats(tokens)
        if not stats:
            break

        best_pair = max(stats, key=stats.get)
        new_token = "".join(best_pair)

        if new_token in vocab_set:
            # Avoid duplicates; stop if no new symbols can be added
            break

        tokens = merge_tokens(tokens, best_pair, new_token)
        merges.append(best_pair)
        vocab.append(new_token)
        vocab_set.add(new_token)

    return vocab, merges


def encode(text, merges, stoi):
    tokens = list(text.replace(" ", SP_MARKER))
    for pair in merges:
        merged_token = "".join(pair)
        tokens = merge_tokens(tokens, pair, merged_token)
    return [stoi[t] for t in tokens]


def decode(ids, itos):
    text = "".join(itos[i] for i in ids)
    return text.replace(SP_MARKER, " ")


# read the dataset
input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
with open(input_file_path, "r") as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# create the train and test splits
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

# train BPE on training split
vocab, merges = train_bpe(train_data, TARGET_VOCAB_SIZE)
itos = {i: t for i, t in enumerate(vocab)}
stoi = {t: i for i, t in itos.items()}

print(f"vocab size: {len(vocab):,}")
print(f"num merges: {len(merges):,}")

# encode train/val
train_ids = encode(train_data, merges, stoi)
val_ids = encode(val_data, merges, stoi)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

# save the meta information as well
meta = {
    "vocab_size": len(vocab),
    "itos": itos,
    "stoi": stoi,
    "merges": merges,
    "level": "bpe",
    "bpe_type": "tiny_sp",
    "sp_marker": SP_MARKER,
}
with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)
