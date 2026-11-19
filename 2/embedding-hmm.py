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
#     display_name: Python 3
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
# # 1.3: TF-IDF (sentences as documents)
# %%
n_docs = len(train_ids)
doc_freq = Counter()
tf_rows = defaultdict(dict)                       # word -> {doc: count}
# one sentence = one document, so df is just how many sentences a word shows up in
for d, sent in enumerate(train_ids):
    for w, c in Counter(sent.tolist()).items():
        if w != 0:                                # skip <UNK>
            tf_rows[w][d] = c
            doc_freq[w] += 1

# log-scaled tf so a word repeated in one sentence doesnt dominate
tfidf_rows = defaultdict(dict)                    # word -> {doc: weight}
for w, docs in tf_rows.items():
    idf = math.log(n_docs / doc_freq[w])          # common words get close to 0 here
    for d, c in docs.items():
        tfidf_rows[w][d] = (1 + math.log(c)) * idf

tfidf_norms = compute_norms(tfidf_rows)           # same helper as in 1.2
nnz = sum(len(r) for r in tfidf_rows.values())
print(f"{n_docs:,} documents | {nnz:,} non-zeros ({nnz / (v * n_docs):.2%} density)")

for word in ["company", "market", "bank", "interest", "profit"]:
    if word in word2idx:
        print(f"{word}: {cosine_sim(tfidf_rows, tfidf_norms, word, min_freq=MIN_NEIGHBOR_FREQ)}")

# %% [markdown]
# # 1.4: Word2Vec (Skip-Gram, Negative Sampling)

# %%
import time

# clip so exp doesnt overflow
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -35, 35)))

# every (centre, context) pair inside the window, same window size as the pmi part
def build_pairs(id_sents, window):
    c, o = [], []
    for s in id_sents:
        s = s.tolist()
        for i, t in enumerate(s):
            if t == 0: continue                   # unk is never a centre
            for j in range(max(0, i - window), min(len(s), i + window + 1)):
                if j != i and s[j] != 0:          # ...and never a context
                    c.append(t); o.append(s[j])
    return np.array(c), np.array(o)

train_c, train_o = build_pairs(train_ids, WINDOW_SIZE)
val_c, val_o = build_pairs(val_ids, WINDOW_SIZE)

# noise distribution for negatives: unigram^0.75, unk gets count 0 so it is never sampled
unigram = np.array([cnt[w] if i > 0 else 0 for i, w in enumerate(idx)], dtype=float)
noise_cdf = np.cumsum(unigram ** 0.75 / (unigram ** 0.75).sum())
# sample by inverting the cdf, much faster than rng.choice with p=
def sample_neg(rng, shape): return np.minimum(np.searchsorted(noise_cdf, rng.random_sample(shape)), v - 1)

def loss_and_grads(W_in, W_out, c, o, neg):
    v_c, u_o, u_n = W_in[c], W_out[o], W_out[neg]              # (B,d) (B,d) (B,K,d)
    pos = np.einsum("bd,bd->b", v_c, u_o)
    ng = np.einsum("bd,bkd->bk", v_c, u_n)
    # logaddexp is the stable way to write -log(sigmoid(x))
    loss = np.logaddexp(0, -pos) + np.logaddexp(0, ng).sum(1)   # -log s(pos) - sum log s(-neg)
    # these are the derivatives from the markdown cell above
    g_pos, g_neg = sigmoid(pos) - 1, sigmoid(ng)
    grad_v = g_pos[:, None] * u_o + np.einsum("bk,bkd->bd", g_neg, u_n)
    grad_uo = g_pos[:, None] * v_c
    grad_un = g_neg[:, :, None] * v_c[:, None, :]
    return loss, grad_v, grad_uo, grad_un

# always 5 negatives and a fixed seed so val loss is comparable between runs
def val_loss(W_in, W_out, K=5):
    rng = np.random.RandomState(123)
    return np.mean([loss_and_grads(W_in, W_out, val_c[s:s+4096], val_o[s:s+4096],
                                   sample_neg(rng, (len(val_c[s:s+4096]), K)))[0].mean()
                    for s in range(0, len(val_c), 4096)])

def train_sgns(dim=100, K=5, epochs=5, lr0=0.025, lr_min=0.0025, batch=256, t=1e-3, verbose=True):
    rng = np.random.RandomState(SEED)
    W_in = (rng.random_sample((v, dim)) - 0.5) / dim            # small random init
    W_out = np.zeros((v, dim))                                  # context table starts at zero
    # drop frequent words more often ("the", ",", "."), otherwise they swamp every batch
    keep = np.minimum(1, np.sqrt(t / np.maximum(unigram / unigram.sum(), 1e-12)))   # subsample frequent words
    total = int(epochs * (keep[train_c] * keep[train_o]).sum() / batch) + 1         # steps, for the lr decay
    step, step_losses, log = 0, [], []
    for ep in range(1, epochs + 1):
        # new random subsample + shuffle every epoch
        m = (rng.random_sample(len(train_c)) < keep[train_c]) & (rng.random_sample(len(train_c)) < keep[train_o])
        ec, eo = train_c[m], train_o[m]
        p = rng.permutation(len(ec)); ec, eo = ec[p], eo[p]
        tot, n = 0.0, 0
        for s in range(0, len(ec), batch):
            c, o = ec[s:s+batch], eo[s:s+batch]
            neg = sample_neg(rng, (len(c), K))
            lr = max(lr_min, lr0 * (1 - step / total))          # linear decay, floor at lr_min
            loss, gv, guo, gun = loss_and_grads(W_in, W_out, c, o, neg)
            # add.at because the same word can show up several times in one batch
            np.add.at(W_in, c, -lr * gv)                                   # parameter updates
            np.add.at(W_out, o, -lr * guo)
            np.add.at(W_out, neg.ravel(), -lr * gun.reshape(-1, dim))
            step_losses.append(loss.mean()); tot += loss.sum(); n += len(c); step += 1
        log.append((ep, tot / n, val_loss(W_in, W_out)))
        if verbose: print(f"epoch {ep}: train {log[-1][1]:.4f} | val {log[-1][2]:.4f}")
    return W_in, step_losses, log

# hyperparameter choice on validation loss (K fixed at 5)
# short 3 epoch runs just to compare dims, test set is not touched
sweep = {d: train_sgns(dim=d, epochs=3, verbose=False)[2][-1][2] for d in (50, 100, 200)}
print("val loss by dim:", sweep)
best_dim = min(sweep, key=sweep.get)

# %%
# final run with the best dim, longer
w2v_emb, step_losses, log = train_sgns(dim=best_dim, epochs=10)

# moving average over 100 steps, the raw curve is too noisy to read
plt.figure(figsize=(5, 3.2))
plt.plot(np.convolve(step_losses, np.ones(100) / 100, mode="valid"))
plt.xlabel("step"); plt.ylabel("SGNS loss"); plt.title("Training loss"); plt.show()


# %% [markdown]
# # 1.5: Comparison

# %%
tfidf_norms = compute_norms(tfidf_rows)
# normalise rows once so cosine is just a dot product
E = w2v_emb / np.maximum(np.linalg.norm(w2v_emb, axis=1, keepdims=True), 1e-12)

def w2v_neighbours(word, k=8, min_freq=MIN_NEIGHBOR_FREQ):
    q = word2idx[word]
    sims = E @ E[q]
    sims[unigram < min_freq] = -np.inf; sims[q] = -np.inf      # same rare word filter as the sparse ones (also kills unk)
    return [idx[t] for t in np.argsort(-sims)[:k]]

# only words that exist in the vocab
for w in [w for w in ["company", "market", "bank", "profit", "said", "new"] if w in word2idx]:
    print(f"\n{w}")
    print("  PPMI:    ", cosine_sim(ppmi_rows, ppmi_norms, w, min_freq=MIN_NEIGHBOR_FREQ))
    print("  TF-IDF:  ", cosine_sim(tfidf_rows, tfidf_norms, w, min_freq=MIN_NEIGHBOR_FREQ))
    print("  Word2Vec:", w2v_neighbours(w))

# sparsity / dimensionality / storage (CSR: float32 value + int32 index = 8 bytes per non-zero)
# dense word2vec is just v * dim float32s
nnz_p = sum(len(r) for r in ppmi_rows.values()); nnz_t = sum(len(r) for r in tfidf_rows.values())
n_docs = len(train_ids)
print(f"\n{'':10}{'dims':>9}{'non-zeros':>12}{'density':>9}{'MB':>8}")
print(f"{'PPMI':10}{v:>9,}{nnz_p:>12,}{nnz_p / v**2:>9.2%}{nnz_p * 8 / 1e6:>8.2f}")
print(f"{'TF-IDF':10}{n_docs:>9,}{nnz_t:>12,}{nnz_t / (v * n_docs):>9.2%}{nnz_t * 8 / 1e6:>8.2f}")
print(f"{'Word2Vec':10}{best_dim:>9,}{v * best_dim:>12,}{'100%':>9}{v * best_dim * 4 / 1e6:>8.2f}")
