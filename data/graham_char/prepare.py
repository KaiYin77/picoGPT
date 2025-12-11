"""
Prepare the Graham Essays dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
from bs4 import BeautifulSoup
import html2text
import time
import re

def download_graham_essays():
    """Download all Paul Graham essays from his website."""
    print("Downloading Paul Graham essays...")

    # Get the articles page
    articles_url = "https://paulgraham.com/articles.html"
    response = requests.get(articles_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all essay links
    essay_links = []
    for table in soup.find_all('table'):
        for link in table.find_all('a'):
            href = link.get('href')
            if href and href.endswith('.html') and not href.startswith('http'):
                essay_links.append(f"https://paulgraham.com/{href}")

    print(f"Found {len(essay_links)} essays")

    # Download each essay
    all_text = ""

    for i, url in enumerate(essay_links):
        try:
            print(f"Downloading essay {i+1}/{len(essay_links)}: {url}")
            response = requests.get(url)
            response.raise_for_status()

            # Convert HTML to text
            text = html2text.html2text(response.text)

            # Clean up the text a bit
            text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Remove excessive newlines
            text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces

            all_text += text + "\n\n"

            # Be nice to the server
            time.sleep(0.1)

        except Exception as e:
            print(f"Error downloading {url}: {e}")
            continue

    return all_text

# Download or load the Graham essays dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    try:
        data = download_graham_essays()
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(data)
    except Exception as e:
        print(f"Error downloading essays: {e}")
        print("Falling back to a smaller sample...")
        # Fallback: just get a few essays
        sample_urls = [
            "https://paulgraham.com/startupideas.html",
            "https://paulgraham.com/do.html",
            "https://paulgraham.com/mean.html"
        ]
        data = ""

        for url in sample_urls:
            try:
                response = requests.get(url)
                text = html2text.html2text(response.text)
                data += text + "\n\n"
                time.sleep(0.1)
            except:
                continue

        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(data)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Dataset preparation complete!")