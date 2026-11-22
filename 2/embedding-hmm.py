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


# %% [markdown]
# # Part 2 · POS tagging with a Hidden Markov Model

# %% [markdown]
# ## 2.1 · Estimate initial, transition and emission probabilities
# Each nonempty training sentence contributes one initial tag, one emission per token, and one transition per adjacent tag pair within that sentence. With the sentence length given, decoding uses one initial probability and no end-state factor. Rare training words share ID 0, so their observed tags teach the model how to score an unknown word.
#
# For each count row, add $k$ to every entry and divide by the row total plus $k$ times the number of columns. The initial distribution is a single row. Emissions have many more columns than transitions, so the validation grid tries smaller emission pseudocounts separately.

# %%
# Keep model state indices local to the baseline.
hmm_tags = sorted({tag for sentence in train_tags for tag in sentence})
tag_id = {tag: i for i, tag in enumerate(hmm_tags)}
n_tags = len(hmm_tags)
start_counts = np.zeros(n_tags, dtype=np.int64)
transition_counts = np.zeros((n_tags, n_tags), dtype=np.int64)
emission_counts = np.zeros((n_tags, len(word2idx)), dtype=np.int64)

for words, tags in zip(train_ids, train_tags, strict=True):
    assert len(words) == len(tags)
    if len(words) == 0:
        continue
    states = np.array([tag_id[tag] for tag in tags])
    start_counts[states[0]] += 1
    np.add.at(transition_counts, (states[:-1], states[1:]), 1)
    np.add.at(emission_counts, (states, words), 1)

def log_distribution(counts, k):
    weights = counts + k
    return np.log(weights / weights.sum(axis=-1, keepdims=True))

def estimate_hmm(sequence_k, emission_k):
    return (log_distribution(start_counts, sequence_k),
            log_distribution(transition_counts, sequence_k),
            log_distribution(emission_counts, emission_k))

# %% [markdown]
# ## 2.2 · Viterbi decoding
# At each position, keep the best cumulative log probability for each possible current tag. For current tag $j$, choose the previous tag $i$ that maximizes the previous score plus $\log P(j\mid i)$, then add $\log P(w_t\mid j)$. Save the winning predecessor for each $j$. After the last word, trace back from the highest-scoring final tag. Using sums of logs avoids underflow from multiplying many small probabilities.

# %%
def viterbi(word_ids, model):
    log_start, log_transition, log_emission = model
    if len(word_ids) == 0:
        return np.empty(0, dtype=np.int64)

    scores = log_start + log_emission[:, word_ids[0]]
    back = np.zeros((len(word_ids), len(scores)), dtype=np.int64)
    states = np.arange(len(scores))
    for t in range(1, len(word_ids)):
        candidates = scores[:, None] + log_transition  # previous x current
        back[t] = candidates.argmax(axis=0)
        scores = candidates[back[t], states] + log_emission[:, word_ids[t]]

    path = np.empty(len(word_ids), dtype=np.int64)
    path[-1] = scores.argmax()
    for t in range(len(word_ids) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path

# %% [markdown]
# ### Choose smoothing on validation sentences
# The initial and transition distributions share one pseudocount. Select the pair with highest validation token accuracy, breaking ties by grid order. Keep the trained vocabulary and counts fixed throughout.

# %%
best_correct = -1
val_gold = [np.array([tag_id.get(t, -1) for t in tags]) for tags in val_tags]
val_total = sum(map(len, val_gold))
for sequence_k in (0.01, 0.1, 1.0):
    for emission_k in (0.001, 0.01, 0.1):
        model = estimate_hmm(sequence_k, emission_k)
        correct = sum(np.count_nonzero(viterbi(words, model) == gold)
                      for words, gold in zip(val_ids, val_gold, strict=True))
        print(f"sequence k={sequence_k:g}  emission k={emission_k:g}  "
              f"validation accuracy={correct / val_total:.4%}")
        if correct > best_correct:
            best_correct = correct
            best_settings = (sequence_k, emission_k)

baseline_hmm = estimate_hmm(*best_settings)
print("Selected (sequence k, emission k):", best_settings)

# %% [markdown]
# ## 2.3 · Test evaluation
# Decode each complete sentence once. Apply each evaluation mask afterward so unknown words retain their context. A confusion matrix counts gold tags in rows and predictions in columns. For each tag, precision divides correct predictions by predicted count, and recall divides them by gold count. Macro F1 averages per-tag F1 over tags present in gold or predictions for that subset. Undefined scores count as zero. Micro precision, recall and F1 all equal token accuracy for this single-label task.

# %%
test_ids = encode(test_sents, word2idx)
eval_id = {tag: i for i, tag in enumerate(pos_names)}
state_to_eval = np.array([eval_id[tag] for tag in hmm_tags])
groups = ("All tokens", "Known to vocabulary", "Mapped to <UNK>", "Unseen in training")
confusion = {group: np.zeros((len(pos_names), len(pos_names)), dtype=np.int64)
             for group in groups}

for tokens, words, tags in zip(test_sents, test_ids, test_tags, strict=True):
    predicted = state_to_eval[viterbi(words, baseline_hmm)]
    gold = np.array([eval_id[tag] for tag in tags], dtype=np.int64)
    unknown = words == word2idx["<UNK>"]
    unseen = np.array([normalise(w) not in cnt for w in tokens], dtype=bool)
    masks = (np.ones(len(words), dtype=bool), ~unknown, unknown, unseen)
    for group, mask in zip(groups, masks):
        np.add.at(confusion[group], (gold[mask], predicted[mask]), 1)

def safe_ratio(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)

def metrics(matrix):
    hits = matrix.diagonal()
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    active = (support + predicted) > 0
    precision = safe_ratio(hits, predicted)
    recall = safe_ratio(hits, support)
    f1 = safe_ratio(2 * hits, predicted + support)
    if not np.any(active):
        return (0, np.nan, np.nan, np.nan, np.nan)
    return (int(support.sum()), hits.sum() / support.sum(),
            precision[active].mean(), recall[active].mean(), f1[active].mean())

print(f"{'Test subset':<21} {'Tokens':>7} {'Accuracy':>9} {'Macro P':>9} "
      f"{'Macro R':>9} {'Macro F1':>9}")
for group, matrix in confusion.items():
    size, acc, p, r, f = metrics(matrix)
    print(f"{group:<21} {size:7d} {acc:9.2%} {p:9.2%} {r:9.2%} {f:9.2%}")
