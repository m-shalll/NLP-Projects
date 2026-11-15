# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: nlp
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Data Setup

# %%
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt

SEED = 17


dataset = load_dataset("eriktks/conll2000", revision="refs/convert/parquet") #download parquet instead of py script
print(dataset)

# POS label names (id -> tag string); falls back to an error message if the metadata is missing
pos_feature = dataset["train"].features["pos_tags"].feature
assert hasattr(pos_feature, "names"), "pos_tags has no label names; send me the features print"
pos_names = pos_feature.names
print(len(pos_names), "POS tags:", pos_names)

def extract(split):
    sents = [list(ex["tokens"]) for ex in split]
    tags  = [[pos_names[t] for t in ex["pos_tags"]] for ex in split]
    return sents, tags

train_full_sents, train_full_tags = extract(dataset["train"])
test_sents, test_tags = extract(dataset["test"])

rng = np.random.RandomState(SEED)
perm = rng.permutation(len(train_full_sents))
split_val = int(len(train_full_sents) * 0.2)
val_idx, train_idx = perm[:split_val], perm[split_val:]

train_sents = [train_full_sents[i] for i in train_idx]
train_tags = [train_full_tags[i] for i in train_idx]
val_sents = [train_full_sents[i] for i in val_idx]
val_tags = [train_full_tags[i] for i in val_idx]

print(f"train: {len(train_sents)} | val: {len(val_sents)} | test: {len(test_sents)}")
print(list(zip(train_sents[0], train_tags[0]))[:10])

assert all(len(s) == len(t) for s, t in zip(train_sents, train_tags))
assert all(len(s) == len(t) for s, t in zip(val_sents, val_tags))
assert all(len(s) == len(t) for s, t in zip(test_sents, test_tags))
assert len(set(train_idx) & set(val_idx)) == 0



# %% [markdown]
# # Part 1: Sparse and Dense Word Representations
# ## 1.1 Preprocessing and Vocabulary: Prepare the text and build a vocabulary from training
#

# %%
import re 
from collections import Counter

DIGIT_REGEX = re.compile(r'\d')

def normalise(token):
    token = token.lower()
    return "<NUM>" if DIGIT_REGEX.search(token) else token

MIN_COUNT = 2 # How many times a token must appear to be included in the vocabulary

def build_vocab(sentences, min_count=MIN_COUNT):
    cnt = Counter(normalise(token) for sentence in sentences for token in sentence)
    words = [word for word, count in cnt.items() if count >= min_count and word != "<UNK>"]
    words.sort(key=lambda x: (-cnt[x], x))  # Sort by frequency (descending) and then alphabetically
    idx = ["<UNK>"] + words  # Add <UNK> token at the beginning
    word2idx = {word: i for i, word in enumerate(idx)}
    return cnt, idx, word2idx

def encode(sentences, stoi):
    """list of token lists -> list of int arrays (one array per sentence)"""
    return [np.array([stoi.get(normalise(t), 0) for t in s], dtype=np.int64) for s in sentences]

cnt, idx, word2idx = build_vocab(train_sents, min_count=MIN_COUNT)   # train split only
v = len(idx)

train_ids = encode(train_sents, word2idx)
val_ids   = encode(val_sents, word2idx)

num_train_tokens = sum(len(s) for s in train_ids)
num_val_tokens   = sum(len(s) for s in val_ids)
print(f"Vocabulary size: {v} | distinct normalised types in train: {len(cnt)}")
print(f"Train tokens: {num_train_tokens} | val tokens: {num_val_tokens}")
print(f"<UNK> rate: {sum((s == 0).sum() for s in train_ids) / num_train_tokens:.2%} (train), "
      f"{sum((s == 0).sum() for s in val_ids) / num_val_tokens:.2%} (val)")
print("Most common words:", idx[1:16])    # idx[0] is <UNK>


# %%
