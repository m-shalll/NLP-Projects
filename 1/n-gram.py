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
# # Data Setup and Inspection

# %%
from datasets import load_dataset
import numpy as np

SEED = 17

dataset = load_dataset("stanfordnlp/imdb")

print(dataset)


def select_random_samples(dataset, samples, seed):
    labels = np.array(dataset["label"])
    rng = np.random.default_rng(seed)

    selected_samples = []
    for label in np.unique(labels):
        label_idx = np.where(labels == label)[0]
        selected_idx = rng.choice(label_idx, size=samples, replace=False)
        selected_samples.extend(selected_idx.tolist())
    rng.shuffle(selected_samples)

    texts = [dataset["text"][idx] for idx in selected_samples]
    labels = labels[selected_samples]

    return texts, labels

X_train, y_train = select_random_samples(dataset["train"], samples=2500, seed=SEED)

X_test, y_test = select_random_samples(dataset["test"], samples=500, seed=SEED)

print("Training reviews, labels:", len(X_train), len(y_train))
print("Testing reviews, labels:", len(X_test), len(y_test))

print("\nTraining class counts:")
print(f"Negative: {np.sum(y_train == 0)}, Positive: {np.sum(y_train == 1)}")

print("\nTesting class counts:")
print(f"Negative: {np.sum(y_test == 0)}, Positive: {np.sum(y_test == 1)}")
# Find indices for negative and positive reviews
negative_indices = np.where(y_train == 0)[0]
positive_indices = np.where(y_train == 1)[0]

# Inspect a negative review
negative_index = negative_indices[0]

print("\nNegative review")
print("Label:", y_train[negative_index])
print(X_train[negative_index])

# Inspect a positive review
positive_index = positive_indices[0]

print("\nPositive review")
print("Label:", y_train[positive_index])
print(X_train[positive_index])


# %% [markdown]
# # Part 1.1: Processing and tokenization

# %%
import re
import html

def clean_text(text):
    # Decode HTML entities, e.g. &amp; -> &
    text = html.unescape(text)

    # Remove HTML tags, e.g. <br /> and <div>
    text = re.sub(r"<[^>]+>", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def tokenize(text):
     # Keep words, contractions, numbers, and individual punctuation marks
    tokens = re.findall(
        r"[a-z]+(?:'[a-z]+)*|[0-9]+|[^\w\s]",
        text
    )
    return tokens

def process_review(text):
    cleaned_text = clean_text(text)
    tokens = ["<s>"] + tokenize(cleaned_text) + ["</s>"]
    return tokens

X_train_tokens = [process_review(review) for review in X_train]
X_test_tokens = [process_review(review) for review in X_test]



# %% [markdown]
# # Part 1.2: Class-Conditional Language Models

# %%
negative_reviews = [tokens for tokens, label in zip(X_train_tokens, y_train) if label == 0]
positive_reviews = [tokens for tokens, label in zip(X_train_tokens, y_train) if label == 1]

from collections import defaultdict

def count_bigrams(reviews):
    bigram_counts = defaultdict(int)
    context_counts = defaultdict(int)

    for tokens in reviews:
        for i in range(len(tokens) -1):
            prev_token = tokens[i]
            next_token = tokens[i + 1]
            bigram_counts[(prev_token, next_token)] += 1
            context_counts[prev_token] += 1

    return bigram_counts, context_counts

positive_bigram_counts, positive_context_counts = count_bigrams(positive_reviews)
negative_bigram_counts, negative_context_counts = count_bigrams(negative_reviews)

