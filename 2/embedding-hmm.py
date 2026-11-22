# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
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
# ## 1.2: PMI

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
# ## 1.3: TF-IDF (sentences as documents)
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
# ## 1.4: Word2Vec (Skip-Gram, Negative Sampling)

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
# ## 1.5: Comparison

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

# %% [markdown]
# # Part 2: Hidden Markov Model and Viterbi

# %% [markdown]
# Interface that Part 3 relies on. Only the signatures and return formats are fixed here,
# the bodies (smoothing, unknown-word model, Viterbi, metrics) are Part 2's job.

# %%
def train_hmm(sentences, tag_sequences):
    """Estimate the HMM from tagged training sentences (raw tokens, tag strings).

    Returns a dict with at least
      'tags'      : list[str], the tag order used by every array below (T tags)
      'log_init'  : (T,)   log P(t_1 = t)
      'log_trans' : (T, T) log P(t_i = b | t_{i-1} = a), indexed [a, b]
    plus whatever emission tables Part 2 needs.
    """
    raise NotImplementedError("Part 2")

def is_unknown(hmm, token):
    """True if `token` (raw string) is treated as an unknown word by the HMM."""
    raise NotImplementedError("Part 2")

def emission_log_probs(hmm, sentence):
    """(n, T) array of log P(token_i | tag) for a raw token list, using Part 2's smoothing
    and unknown-word handling. Must be finite everywhere."""
    raise NotImplementedError("Part 2")

def viterbi(log_init, log_trans, log_emit):
    """Most likely tag-index sequence (list[int], length n) for (T,), (T, T), (n, T) log-prob arrays."""
    raise NotImplementedError("Part 2")

def evaluate(gold_tags, pred_tags, unknown_mask):
    """gold_tags / pred_tags: lists of tag-string lists, unknown_mask: lists of bools (same shape).

    Returns {'all': m, 'unknown': m} where m = {'accuracy', 'precision', 'recall', 'f1'}
    computed over all tokens and over unknown tokens only.
    """
    raise NotImplementedError("Part 2")

# %% [markdown]
# # Part 3: Embedding-Assisted HMM
#
# ## 3.1 Strategy
#
# The baseline HMM has almost nothing to go on for an unknown word: one shared unknown-word
# emission. The idea here is to let each Part 1 representation nominate **known words that could
# stand in for the unknown one**, and borrow their tag distributions.
#
# For an unknown token $x$:
#
# 1. **Get a query from the representation.**
#    - If the normalised form of $x$ is in the Part 1 vocabulary (e.g. `Bancorp` unseen by the
#      HMM but `bancorp` seen, or any number → `<NUM>`), use its own vector.
#    - Otherwise $x$ has no vector at all (it never appeared in train), so describe it through
#      its context words $C$ (same ±2 window as Part 1), each representation in its own way:
#      - **PMI:** the context words *are* PMI dimensions. Score each known word by the cosine between
#        its PPMI row and the indicator vector of $C$, i.e. "which words like to appear next to these
#        words" (first-order, syntagmatic).
#      - **TF-IDF:** $x$'s missing document vector is approximated by the documents its context words
#        occur in, so the score of $w$ is $\sum_{c \in C}\cos_\text{tfidf}(w, c)$ (topical relatedness).
#      - **Word2Vec:** the mean embedding of $C$ is compared with each known word's *average training
#        context* $\bar{c}_w = \sum_c n(w,c)\,e_c / \sum_c n(w,c)$, so we match the slot, not the
#        neighbours themselves (averaging $C$ alone would return words similar to "the", "of" ...).
# 2. **Nearest known words.** Keep the top-$k$ candidates that were seen at least
#    `MIN_NEIGHBOR_FREQ` times (so their tag distributions are reliable) and weight them by their
#    (positive) similarity $s_j$:
#    $\hat P(t \mid x) = (1-\epsilon)\frac{\sum_j s_j P(t \mid w_j)}{\sum_j s_j} + \epsilon P(t)$.
# 3. **Turn it into an emission score.** By Bayes, $P(x\mid t) \propto P(t\mid x)/P(t)$ (the $P(x)$
#    factor is the same for every tag, so Viterbi ignores it). This is interpolated log-linearly with
#    the baseline unknown emission:
#    $\log e(x,t) = (1-\lambda)\log P_\text{HMM}(x\mid t) + \lambda\,[\log \hat P(t\mid x) - \log P(t)]$.
#
# Known words keep their HMM emissions untouched, so all differences between the four systems come
# from unknown tokens. $k$ and $\lambda$ are tuned per representation on the validation split.

# %%
hmm = train_hmm(train_sents, train_tags)
tags = hmm["tags"]
T = len(tags)
tag2i = {t: i for i, t in enumerate(tags)}

# P(tag | word) for every Part 1 vocabulary entry, counted on the train split with the same normalise()
tag_word = np.zeros((v, T))
for sent, ts in zip(train_sents, train_tags):
    for w, t in zip(sent, ts):
        tag_word[word2idx.get(normalise(w), 0), tag2i[t]] += 1
p_tag_given_word = tag_word / np.maximum(tag_word.sum(1, keepdims=True), 1)
tag_prior = (tag_word.sum(0) + 1) / (tag_word.sum() + T)       # add-one so every tag has P(t) > 0

# only words with a trustworthy tag distribution may be borrowed from (unigram[0] = 0 drops <UNK>)
candidate = unigram >= MIN_NEIGHBOR_FREQ
print(f"{T} tags | {candidate.sum():,} candidate neighbour words")

def context_ids(sentence, i, window=WINDOW_SIZE):
    """vocab ids of the known words around position i (unk positions dropped)"""
    ids = [word2idx.get(normalise(sentence[j]), 0)
           for j in range(max(0, i - window), min(len(sentence), i + window + 1)) if j != i]
    return [c for c in ids if c != 0]

# %% [markdown]
# ### Similarity scorers
# Each scorer returns a length-`v` score vector over the vocabulary for one query:
# `self_scores(word_id)` when the token has its own vector, `slot_scores(context_ids)` otherwise.

# %%
from functools import lru_cache

# ---- PMI: column index of the (row-normalised) PPMI matrix, context -> (words, values) ----
_cols = defaultdict(lambda: ([], []))
for w, row in ppmi_rows.items():
    for c, x in row.items():
        _cols[c][0].append(w); _cols[c][1].append(x / ppmi_norms[w])
ppmi_cols = {c: (np.array(ws), np.array(xs)) for c, (ws, xs) in _cols.items()}
del _cols

def pmi_weighted(weights):
    """sum_c weights[c] * PPMI_w[c] / |PPMI_w| for every word w"""
    s = np.zeros(v)
    for c, wt in weights.items():
        if c in ppmi_cols:
            ws, xs = ppmi_cols[c]
            s[ws] += wt * xs                         # ws has no repeats within one column
    return s

def pmi_self(q):
    row = ppmi_rows.get(q, {})
    n = ppmi_norms.get(q, 0)
    return pmi_weighted({c: x / n for c, x in row.items()}) if n > 0 else np.zeros(v)

def pmi_slot(ctx):
    weights = Counter(ctx)
    n = math.sqrt(sum(x * x for x in weights.values()))
    return pmi_weighted({c: x / n for c, x in weights.items()})

# ---- TF-IDF: unit-length rows indexed by word and by document ----
_docs = defaultdict(lambda: ([], []))
tfidf_unit = {}
for w, docs in tfidf_rows.items():
    n = tfidf_norms[w]
    if n == 0:                                       # word in every sentence -> idf 0, no information
        continue
    tfidf_unit[w] = (np.fromiter(docs.keys(), int), np.fromiter(docs.values(), float) / n)
    for d, x in docs.items():
        _docs[d][0].append(w); _docs[d][1].append(x / n)
tfidf_docs = {d: (np.array(ws), np.array(xs)) for d, (ws, xs) in _docs.items()}
del _docs

@lru_cache(maxsize=4096)
def tfidf_cos_row(q):
    """cosine between word q and every word, through the documents q occurs in (float32 to keep the cache small)"""
    if q not in tfidf_unit:
        return np.zeros(v, dtype=np.float32)
    ds, xs = tfidf_unit[q]
    ws = np.concatenate([tfidf_docs[d][0] for d in ds])
    vals = np.concatenate([x * tfidf_docs[d][1] for d, x in zip(ds, xs)])
    return np.bincount(ws, weights=vals, minlength=v).astype(np.float32)

def tfidf_self(q): return tfidf_cos_row(q).astype(float)
def tfidf_slot(ctx): return np.sum([tfidf_cos_row(c) for c in ctx], axis=0).astype(float)

# ---- Word2Vec: unit embeddings E (from 1.5) and each word's average training context ----
ctx_centroid = np.zeros_like(E)
_pairs = np.array([p for p in pair_counts if p[1] != 0])         # unk contexts are dropped at query time too
_n = np.array([pair_counts[tuple(p)] for p in _pairs], dtype=float)
for s in range(0, len(_pairs), 200_000):                          # chunked so the (pairs, dim) product stays small
    p, n = _pairs[s:s + 200_000], _n[s:s + 200_000]
    np.add.at(ctx_centroid, p[:, 0], n[:, None] * E[p[:, 1]])
ctx_centroid /= np.maximum(np.linalg.norm(ctx_centroid, axis=1, keepdims=True), 1e-12)
del _pairs, _n

def w2v_self(q): return E @ E[q]
def w2v_slot(ctx):
    m = E[ctx].mean(0)
    return ctx_centroid @ (m / max(np.linalg.norm(m), 1e-12))

SCORERS = {
    "PMI":      (pmi_self, pmi_slot),
    "TF-IDF":   (tfidf_self, tfidf_slot),
    "Word2Vec": (w2v_self, w2v_slot),
}

def neighbours(name, sentence, i, k):
    """top-k candidate words for the unknown token sentence[i] as (ids, scores), best first; None if no query"""
    self_f, slot_f = SCORERS[name]
    q = word2idx.get(normalise(sentence[i]), 0)
    if q != 0:
        scores = self_f(q)
    else:
        ctx = context_ids(sentence, i)
        if not ctx:
            return None
        scores = slot_f(ctx)
        scores[ctx] = -np.inf                                   # a context word is not a stand-in for the slot
    scores = np.where(candidate, scores, -np.inf)
    top = np.argpartition(-scores, k)[:k]
    top = top[np.argsort(-scores[top])]
    return top, scores[top]

# quick look: which known words stand in for a few unknown validation tokens
_shown = 0
for sent in val_sents:
    for i, w in enumerate(sent):
        if _shown < 4 and is_unknown(hmm, w) and word2idx.get(normalise(w), 0) == 0 and context_ids(sent, i):
            print(f"\n{' '.join(sent[max(0, i - 3):i])} [{w}] {' '.join(sent[i + 1:i + 4])}")
            for name in SCORERS:
                print(f"  {name:9}", [idx[j] for j in neighbours(name, sent, i, 6)[0]])
            _shown += 1

# %% [markdown]
# ### Integration with the HMM emissions and decoding

# %%
K_MAX = 50
EPS = 1e-3

def neighbour_tag_dist(top, scores):
    """similarity-weighted P(t | x) from the neighbours' tag distributions, None if no positive evidence"""
    wts = np.maximum(scores, 0)
    if wts.sum() <= 0:
        return None
    dist = wts @ p_tag_given_word[top] / wts.sum()
    return (1 - EPS) * dist + EPS * tag_prior

def assisted_emission(base_row, dist, lam):
    emb_row = np.log(dist) - np.log(tag_prior)                 # P(x|t) up to a per-token constant
    return emb_row if lam == 1 else (1 - lam) * base_row + lam * emb_row

def prepare(sents):
    """everything that does not depend on k / lambda: baseline emissions, unknown masks, baseline
    predictions and the top-K_MAX neighbours of every unknown token under each representation"""
    data = []
    for sent in sents:
        if not sent:                                            # the dataset has a few empty sentences
            data.append({"emit": None, "unk": [], "base": [], "neigh": {name: {} for name in SCORERS}})
            continue
        emit = emission_log_probs(hmm, sent)
        unk = [is_unknown(hmm, w) for w in sent]
        base_pred = [tags[j] for j in viterbi(hmm["log_init"], hmm["log_trans"], emit)]
        neigh = {name: {} for name in SCORERS}
        for i in np.flatnonzero(unk):
            for name in SCORERS:
                nb = neighbours(name, sent, i, K_MAX)
                if nb is not None:
                    neigh[name][i] = nb
        data.append({"emit": emit, "unk": unk, "base": base_pred, "neigh": neigh})
    return data

def decode(data, name=None, k=10, lam=0.5):
    """tag every sentence; name=None is the baseline HMM. Only sentences with an unknown word are re-decoded."""
    preds = []
    for ex in data:
        nbs = ex["neigh"][name] if name else {}
        if not nbs:
            preds.append(ex["base"]); continue
        emit = ex["emit"].copy()
        for i, (top, sc) in nbs.items():
            dist = neighbour_tag_dist(top[:k], sc[:k])
            if dist is not None:
                emit[i] = assisted_emission(emit[i], dist, lam)
        preds.append([tags[j] for j in viterbi(hmm["log_init"], hmm["log_trans"], emit)])
    return preds

t0 = time.time()
val_data = prepare(val_sents)
val_unk = [ex["unk"] for ex in val_data]
n_unk = sum(map(sum, val_unk))
print(f"validation: {n_unk:,} unknown tokens ({n_unk / num_val_tokens:.2%}) | prepared in {time.time() - t0:.0f}s")

# %% [markdown]
# ### Tuning $k$ and $\lambda$ on the validation split
# Selected by overall validation accuracy (only unknown tokens and their neighbours can change, so this
# also accounts for knock-on effects through the transitions). The test set is not used here.

# %%
K_GRID = [1, 5, 10, 20, 50]
LAM_GRID = [0.25, 0.5, 0.75, 1.0]

best_cfg = {}
base_val = evaluate(val_tags, decode(val_data), val_unk)
print(f"baseline   acc {base_val['all']['accuracy']:.4f} | unk acc {base_val['unknown']['accuracy']:.4f}")
for name in SCORERS:
    grid = np.zeros((len(K_GRID), len(LAM_GRID)))
    for a, k in enumerate(K_GRID):
        for b, lam in enumerate(LAM_GRID):
            grid[a, b] = evaluate(val_tags, decode(val_data, name, k, lam), val_unk)["all"]["accuracy"]
    a, b = np.unravel_index(grid.argmax(), grid.shape)
    best_cfg[name] = (K_GRID[a], LAM_GRID[b])
    print(f"{name:10} best k={K_GRID[a]:<3} lambda={LAM_GRID[b]:<5} val acc {grid[a, b]:.4f}")

    plt.figure(figsize=(4, 2.8))
    for b, lam in enumerate(LAM_GRID):
        plt.plot(K_GRID, grid[:, b], marker="o", label=f"λ={lam}")
    plt.xscale("log"); plt.xlabel("k"); plt.ylabel("val accuracy"); plt.title(name); plt.legend(fontsize=7)
    plt.show()

# %% [markdown]
# ## 3.2 Comparison of the four systems on the test set

# %%
t0 = time.time()
test_data = prepare(test_sents)
test_unk = [ex["unk"] for ex in test_data]
n_unk = sum(map(sum, test_unk))
print(f"test: {n_unk:,} unknown tokens ({n_unk / sum(map(len, test_sents)):.2%}) | prepared in {time.time() - t0:.0f}s\n")

test_preds = {"Baseline HMM": decode(test_data)}
for name, (k, lam) in best_cfg.items():
    test_preds[f"{name}-assisted"] = decode(test_data, name, k, lam)

results = {sys: evaluate(test_tags, p, test_unk) for sys, p in test_preds.items()}
cols = ["accuracy", "precision", "recall", "f1"]
print(f"{'':20}" + "".join(f"{c[:4]:>8}" for c in cols) + "  |" + "".join(f"{'unk ' + c[:4]:>10}" for c in cols))
for sys, r in results.items():
    print(f"{sys:20}" + "".join(f"{r['all'][c]:>8.4f}" for c in cols) + "  |"
          + "".join(f"{r['unknown'][c]:>10.4f}" for c in cols))

# %% [markdown]
# ## 3.3 Error analysis
# Unknown tokens where an assisted system disagrees with the baseline: how often it fixes vs. breaks
# the baseline tag, which gold tags benefit, and a few concrete examples with the borrowed neighbours.

# %%
base = test_preds["Baseline HMM"]
gold_unk_tags = Counter(g for gs, us in zip(test_tags, test_unk) for g, u in zip(gs, us) if u)
top_tags = [t for t, _ in gold_unk_tags.most_common(8)]

def unk_acc_by_tag(preds):
    hit, tot = Counter(), Counter()
    for gs, ps, us in zip(test_tags, preds, test_unk):
        for g, p, u in zip(gs, ps, us):
            if u:
                tot[g] += 1; hit[g] += g == p
    return {t: hit[t] / tot[t] for t in top_tags}

print("unknown-token accuracy by gold tag (most frequent unknown tags)")
print(f"{'':20}" + "".join(f"{t:>7}" for t in top_tags))
for sys, p in test_preds.items():
    acc = unk_acc_by_tag(p)
    print(f"{sys:20}" + "".join(f"{acc[t]:>7.2f}" for t in top_tags))

for name in SCORERS:
    preds = test_preds[f"{name}-assisted"]
    fixed, broken = [], []
    for s, (sent, gs, bs, ps, us) in enumerate(zip(test_sents, test_tags, base, preds, test_unk)):
        for i, (g, b, p, u) in enumerate(zip(gs, bs, ps, us)):
            if u and b != p:
                (fixed if p == g else broken if b == g else []).append((s, i))
    print(f"\n=== {name}: fixed {len(fixed)} baseline errors, broke {len(broken)} correct tags ===")
    k = best_cfg[name][0]
    for label, cases in (("fixed", fixed[:3]), ("broken", broken[:3])):
        for s, i in cases:
            sent = test_sents[s]
            nb = neighbours(name, sent, i, min(k, 6))
            nb_str = [idx[j] for j in nb[0]] if nb else []
            print(f"[{label}] ...{' '.join(sent[max(0, i - 3):i])} [{sent[i]}] {' '.join(sent[i + 1:i + 3])}... "
                  f"gold={test_tags[s][i]} base={base[s][i]} {name}={preds[s][i]} | neighbours {nb_str}")

# %% [markdown]
# ### Discussion
#
# *(Re-check against the numbers above once Part 2 is in place.)*
#
# - **Where the help comes from.** The baseline can only fall back on tag priors (plus whatever
#   unknown-word model Part 2 uses), so it tends to dump unknown words into the most common open
#   classes (NN/NNP). Borrowed tag distributions mostly help when the neighbours are *syntactically*
#   alike, e.g. an unseen verb form after "has" or "to" getting VBN/VB neighbours, or an unseen
#   capitalised name surrounded by other names.
# - **PMI (sparse, first-order).** Its slot scores ask "which words appear right next to these
#   context words", which is close to what a tagger needs: words that fill the same ±2 slot usually
#   share a POS. It is strongest when the context contains informative function words, and weakest
#   when the context is itself rare (few non-zero PPMI entries) or the context is only punctuation.
# - **TF-IDF (sparse, document-level).** Sentences-as-documents capture topic ("stocks, shares,
#   trading") rather than syntax, so its neighbours mix nouns, verbs and adjectives of the same topic.
#   It can still lift the noun-ish tags, but it is the most likely to *hurt*, e.g. pushing an unseen
#   verb towards NN because the sentence is about markets. Worse, punctuation and stopwords occur in
#   almost every sentence, so their (low-idf but non-zero) rows overlap with *any* context and they
#   become hubs at the top of the neighbour list in context mode. That is why validation keeps $k$
#   and $\lambda$ small for TF-IDF: it is mainly useful in the self-vector case.
# - **Word2Vec (dense).** The context-centroid comparison is smooth: even rare context words have a
#   meaningful vector, so it degrades gracefully where PMI's rows are empty. On the other hand the
#   averaging blurs word order (left vs right neighbour), so it confuses tags that differ mostly by
#   position (e.g. JJ vs NN in "the ___ market", VBD vs VBN).
# - **Self-vector case.** Tokens whose normalised form is in the vocabulary (case variants, numbers →
#   `<NUM>`) get very reliable neighbours from all three representations because the word itself is
#   usually its own top neighbour; most of the "free" gains come from these.
# - **Tuned λ** reflects how much each representation is trusted: a small λ means the baseline
#   unknown model was already about as good, a large one means the borrowed distribution dominates.
