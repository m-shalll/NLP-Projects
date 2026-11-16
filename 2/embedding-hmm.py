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


# %% [markdown]
# # 1.2: PMI

# %%
import math
from collections import Counter, defaultdict

WINDOW_SIZE = 2   # tokens to the left and right of the target that count as context
ALPHA = 0.75      # context distribution smoothing

# 1. Count (target, context) pairs
pair_counts = Counter()
for sentence in train_ids:
    sentence = sentence.tolist()
    for i, target in enumerate(sentence):
        for j in range(max(0, i - WINDOW_SIZE), min(len(sentence), i + WINDOW_SIZE + 1)):
            if i != j:
                pair_counts[(target, sentence[j])] += 1

# 2. Marginals
total = sum(pair_counts.values())
word_totals, context_counts = Counter(), Counter()
for (target, context), count in pair_counts.items():
    word_totals[target] += count
    context_counts[context] += count

context_smooth = {context: count ** ALPHA for context, count in context_counts.items()}
context_norm = sum(context_smooth.values())

# 3. Sparse rows: raw counts and PPMI
count_rows = defaultdict(dict)
ppmi_rows = defaultdict(dict)

for (target, context), count in pair_counts.items():
    count_rows[target][context] = count
    prob_targetContext = count / total
    prob_target = word_totals[target] / total
    prob_context = context_smooth[context] / context_norm
    pmi = math.log(prob_targetContext / (prob_target * prob_context))
    if pmi > 0:                                   # positive PMI only
        ppmi_rows[target][context] = pmi

# 4. Norms and cosine similarity
def compute_norms(rows):
    return {t: math.sqrt(sum(x ** 2 for x in ctxs.values())) for t, ctxs in rows.items()}

count_norms = compute_norms(count_rows)
ppmi_norms = compute_norms(ppmi_rows)

def cosine_sim(rows, norms, word, k=8, min_freq=0):
    """Top-k cosine neighbours; candidates seen fewer than min_freq times in train are skipped."""
    query = word2idx[word]
    query_row = rows[query]
    scores = []

    for target, row in rows.items():
        if target == query or target == 0 or not row:        # skip itself, <UNK>, empty rows
            continue
        if cnt[idx[target]] < min_freq:                      # optional rare-word filter
            continue
        small, large = (query_row, row) if len(query_row) < len(row) else (row, query_row)
        dot = sum(x * large[context] for context, x in small.items() if context in large)
        scores.append((dot / (norms[query] * norms[target] + 1e-8), target))

    scores.sort(reverse=True)
    return [idx[target] for _, target in scores[:k]]

# 5. Sparsity stats
nonzero_counts = sum(len(r) for r in count_rows.values())
nonzero_ppmi = sum(len(r) for r in ppmi_rows.values())
print(f"raw counts: {nonzero_counts:,} non-zero entries ({nonzero_counts / v**2:.2%} density)")
print(f"PPMI:       {nonzero_ppmi:,} non-zero entries ({nonzero_ppmi / v**2:.2%} density)")

# 6. Neighbours: raw counts, PPMI, and PPMI with the rare-word filter
MIN_NEIGHBOR_FREQ = 5
for word in ["company", "market", "bank", "interest", "profit"]:
    if word in word2idx:
        print(f"\n{word}")
        print("  counts:       ", cosine_sim(count_rows, count_norms, word))
        print("  PPMI:         ", cosine_sim(ppmi_rows, ppmi_norms, word))
        print(f"  PPMI (freq>={MIN_NEIGHBOR_FREQ}):", cosine_sim(ppmi_rows, ppmi_norms, word, min_freq=MIN_NEIGHBOR_FREQ))
