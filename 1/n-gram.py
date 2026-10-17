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


# %%
