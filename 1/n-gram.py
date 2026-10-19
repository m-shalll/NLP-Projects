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
# # Data Setup and Inspection
#

# %%
from collections import defaultdict
import html
import re

import numpy as np
from datasets import load_dataset

SEED = 17
TRAIN_PER_CLASS = 2500
TEST_PER_CLASS = 500

dataset = load_dataset("stanfordnlp/imdb")
print(dataset)


def select_random_samples(split, samples_per_class, seed):
    labels = np.asarray(split["label"])
    rng = np.random.default_rng(seed)
    selected_indices = []

    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        if len(label_indices) < samples_per_class:
            raise ValueError(
                f"Class {label} has {len(label_indices)} examples; "
                f"{samples_per_class} are required."
            )
        selected_indices.extend(
            rng.choice(label_indices, size=samples_per_class, replace=False).tolist()
        )

    rng.shuffle(selected_indices)
    texts = [split["text"][index] for index in selected_indices]
    return texts, labels[selected_indices]


X_train, y_train = select_random_samples(
    dataset["train"], samples_per_class=TRAIN_PER_CLASS, seed=SEED
)
X_test, y_test = select_random_samples(
    dataset["test"], samples_per_class=TEST_PER_CLASS, seed=SEED
)

print("Training reviews, labels:", len(X_train), len(y_train))
print("Testing reviews, labels:", len(X_test), len(y_test))
# Find indices for negative and positive reviews.
print(
    "Training class counts:",
    {int(label): int(np.sum(y_train == label)) for label in np.unique(y_train)},
)
print(
    "Testing class counts:",
    {int(label): int(np.sum(y_test == label)) for label in np.unique(y_test)},
)
print("\nRaw negative review:\n", X_train[np.flatnonzero(y_train == 0)[0]])
print("\nRaw positive review:\n", X_train[np.flatnonzero(y_train == 1)[0]])


# %% [markdown]
# # Processing and tokenization
#

# %%
def clean_text(text):
    # Decode HTML entities, e.g. &amp; -> &
    text = html.unescape(text)

    # Remove HTML tags, e.g. <br /> and <div>
    text = re.sub(r"<[^>]+>", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Normalize whitespace
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    # Keep words, contractions, numbers, and individual punctuation marks.
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)*|[^\w\s]", text, flags=re.UNICODE)


def process_review(text):
    # Add sequence boundary markers for language-model training and scoring.
    return ["<s>"] + tokenize(clean_text(text)) + ["</s>"]


X_train_tokens = [process_review(review) for review in X_train]
X_test_tokens = [process_review(review) for review in X_test]
print("Example tokens:", X_train_tokens[0][:40])


# %% [markdown]
# # Class-Conditional Bigram Language Models
#

# %%
def count_bigrams(reviews):
    bigram_counts = defaultdict(int)
    context_counts = defaultdict(int)

    for tokens in reviews:
        for previous, following in zip(tokens, tokens[1:]):
            bigram_counts[(previous, following)] += 1
            context_counts[previous] += 1

    return bigram_counts, context_counts


class BigramModel:
    def __init__(self, bigram_counts, context_counts, vocab):
        self.bigram_counts = bigram_counts
        self.context_counts = context_counts
        self.vocab_size = len(vocab)
        if self.vocab_size == 0:
            raise ValueError("The language-model vocabulary cannot be empty.")

    def probability(self, previous, following):
        bigram_count = self.bigram_counts.get((previous, following), 0)
        context_count = self.context_counts.get(previous, 0)
        return (bigram_count + 1) / (context_count + self.vocab_size)  # Laplace smoothing


negative_reviews = [tokens for tokens, label in zip(X_train_tokens, y_train) if label == 0]
positive_reviews = [tokens for tokens, label in zip(X_train_tokens, y_train) if label == 1]
lm_vocab = {token for tokens in X_train_tokens for token in tokens}
lm_vocab.add("<UNK>")

positive_bigram_counts, positive_context_counts = count_bigrams(positive_reviews)
negative_bigram_counts, negative_context_counts = count_bigrams(negative_reviews)
positive_model = BigramModel(positive_bigram_counts, positive_context_counts, lm_vocab)
negative_model = BigramModel(negative_bigram_counts, negative_context_counts, lm_vocab)
print("Language-model vocabulary size:", len(lm_vocab))


# %% [markdown]
# # Log-Likelihood, Perplexity, and LM Classification
#

# %%
def map_unknown_tokens(tokens, vocab):
    # Map words absent from the training vocabulary to the unknown-token bucket.
    return [token if token in vocab else "<UNK>" for token in tokens]


def calculate_ll_and_perplexity(tokens, model, vocab):
    tokens = map_unknown_tokens(tokens, vocab)
    n_bigrams = len(tokens) - 1

    # A valid padded document contains at least one bigram.
    if n_bigrams <= 0:
        raise ValueError("At least two padded tokens are required to score a document.")

    log_likelihood = 0.0
    for previous, following in zip(tokens, tokens[1:]):
        probability = model.probability(previous, following)

        # Add-one smoothing should keep every probability in (0, 1].
        if not 0 < probability <= 1:
            raise ValueError(
                f"Invalid probability {probability} for bigram "
                f"({previous!r}, {following!r})."
            )
        log_likelihood += np.log(probability)

    perplexity = np.exp(-log_likelihood / n_bigrams)
    return float(log_likelihood), float(perplexity)


def score_document(text):
    tokens = process_review(text)
    positive_ll, positive_perplexity = calculate_ll_and_perplexity(
        tokens, positive_model, lm_vocab
    )
    negative_ll, negative_perplexity = calculate_ll_and_perplexity(
        tokens, negative_model, lm_vocab
    )
    return {
        "positive_log_likelihood": positive_ll,
        "positive_perplexity": positive_perplexity,
        "negative_log_likelihood": negative_ll,
        "negative_perplexity": negative_perplexity,
    }


def predict_with_language_models(tokenized_reviews):
    predictions = []
    for tokens in tokenized_reviews:
        _, positive_perplexity = calculate_ll_and_perplexity(
            tokens, positive_model, lm_vocab
        )
        _, negative_perplexity = calculate_ll_and_perplexity(
            tokens, negative_model, lm_vocab
        )
        predictions.append(int(positive_perplexity < negative_perplexity))
    return np.asarray(predictions, dtype=int)


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    accuracy = float(np.mean(y_true == y_pred))
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / denominator if denominator else 0.0
    return {"accuracy": accuracy, "f1": f1}


example_scores = score_document(X_test[0])
print("Example document scores:", example_scores)
y_pred_lm = predict_with_language_models(X_test_tokens)
lm_metrics = classification_metrics(y_test, y_pred_lm)
print("Bigram LM:", lm_metrics)


# %% [markdown]
# # Next-Word Prediction
#

# %%
def predict_next_words(sentence, model, max_length=20):
    if max_length < 0:
        raise ValueError("max_length must be non-negative.")

    tokens = tokenize(clean_text(sentence))
    if not tokens or tokens[0] != "<s>":
        tokens.insert(0, "<s>")
    generated = []
    candidates = sorted(lm_vocab - {"<s>", "<UNK>"})

    for _ in range(max_length):
        previous = tokens[-1] if tokens[-1] in lm_vocab else "<UNK>"
        next_word = max(candidates, key=lambda word: model.probability(previous, word))
        if next_word == "</s>":
            break
        generated.append(next_word)
        tokens.append(next_word)

    return generated


print(
    "Generated continuation:",
    predict_next_words("the movie", positive_model, max_length=10),
)


# %% [markdown]
# ### **Part 2:Vocabulary, Naive Bayes, and Log-Ratios**

# %% [markdown]
#     2.1 Building the vocabulary

# %%
from collections import Counter

train_features = []
#remove '<s>' and '</s>'(as we only want useful words)
reviews_naive= [tokens[1:-1] for tokens in X_train_tokens]

#count bigrams
bigram_counts_naive, _ = count_bigrams(reviews_naive)

#count unigrams 
unigram_counts_naive = Counter() #used counter to easliy get top 2000
for tokens in reviews_naive:
    unigrams = []

    for wrd in tokens:
        unigram = (wrd,)
        unigram_counts_naive[unigram] += 1
        unigrams.append(unigram)

    bigrams = list(zip(tokens, tokens[1:]))
   ## print("Bigrams for review:", bigrams)
    # Append once per review.
    train_features.append(unigrams + bigrams)
#combine both into 1 var        
feature_counts = Counter(unigram_counts_naive)
feature_counts.update(bigram_counts_naive)

#get the top 2000
top_features_naive=feature_counts.most_common(2000)

# make a vocab dictionary to map each feature to its index
vocabulary = {
    feature: index
    for index, (feature, count) in enumerate(top_features_naive)
}


# %% [markdown]
#     2.2 Counting Probabilites

# %%


def class_count():
   
    positive_counts = np.zeros(len(vocabulary), dtype=np.int64)
    negative_counts = np.zeros(len(vocabulary), dtype=np.int64)
    for features, label in zip(train_features, y_train):
        counts = positive_counts if label == 1 else negative_counts
        for f in features:
            idx = vocabulary.get(f)
            if idx is not None:
                counts[idx] += 1
    return positive_counts, negative_counts

def calculate_probabilites():
    positive_counts, negative_counts = class_count()
    total_positive=np.sum(positive_counts)
    total_negative=np.sum(negative_counts)
    v=len(vocabulary)
    positive_probabilities = (positive_counts + 1) / (total_positive + v)
    negative_probabilities = (negative_counts + 1) / (total_negative + v)
    return positive_probabilities, negative_probabilities


# %% [markdown]
#     2.3 compute Log Ratios

# %%
def compute_log_ratios():
    
    positive_probabilities, negative_probabilities = calculate_probabilites()
    log_ratios=np.log(positive_probabilities / negative_probabilities)
    return log_ratios
log_ratios=compute_log_ratios()


# %% [markdown]
#     2.4 Naive Bayes Prediction

# %%
def predict_naive_bayes(X):
    words = X[1:-1]
    unigrams = [(word,) for word in words]
    bigram=list(zip(X,X[1:]))
    features = unigrams + bigram 
    score = 0.0
    for feature in features:
        if feature in vocabulary:
            index=vocabulary[feature]
            score+=log_ratios[index]
    return int(score>0)



            


# %%
y_pred_nb = []
log_ratios = compute_log_ratios()
for tokens,label in zip(X_test_tokens,y_test):
    prediction=predict_naive_bayes(tokens,)
    y_pred_nb.append(prediction)
    
nb_metrics = classification_metrics(y_test, y_pred_nb)
print("Naive Bayes:", nb_metrics)


# %% [markdown]
# ## Part 3: NB-Weighted Logistic Regression
# ### 3.1 Construct NB-weighted feature matrices

# %%
def extract_feature_counts(tokens, vocabulary):
    """
    Extract unigram and bigram counts using the same representation
    used to construct the Part 2 vocabulary.

    tokens includes <s> and </s>, so remove these first.
    """
    words = tokens[1:-1]

    features = [(word,) for word in words]
    features += list(zip(words, words[1:]))

    counts = np.zeros(len(vocabulary), dtype=np.float64)

    for feature in features:
        index = vocabulary.get(feature)
        if index is not None:
            counts[index] += 1.0

    return counts


def build_nb_weighted_matrix(tokenized_reviews, vocabulary, log_ratios):
    """
    Build a matrix where each feature count is multiplied
    by its Naive Bayes log-ratio.
    """
    X_counts = np.zeros(
        (len(tokenized_reviews), len(vocabulary)),
        dtype=np.float64
    )

    for i, tokens in enumerate(tokenized_reviews):
        X_counts[i] = extract_feature_counts(tokens, vocabulary)

    # NB-weight each feature
    X_weighted = X_counts * log_ratios[np.newaxis, :]

    return X_weighted


# Construct the training and test matrices
X_train_weighted = build_nb_weighted_matrix(
    X_train_tokens, vocabulary, log_ratios
)

X_test_weighted = build_nb_weighted_matrix(
    X_test_tokens, vocabulary, log_ratios
)

print("Training feature matrix shape:", X_train_weighted.shape)
print("Testing feature matrix shape:", X_test_weighted.shape)

assert X_train_weighted.shape == (5000, 2000)
assert X_test_weighted.shape == (1000, 2000)
assert np.all(np.isfinite(X_train_weighted))
assert np.all(np.isfinite(X_test_weighted))



# %% [markdown]
# ### 3.2 Standardization and bias column

# %%
# Statistics are learned from the training set only.
feature_means = np.mean(X_train_weighted, axis=0)
feature_stds = np.std(X_train_weighted, axis=0)

# Avoid division by zero for constant features.
feature_stds[feature_stds == 0] = 1.0

X_train_standardized = (
    X_train_weighted - feature_means
) / feature_stds

X_test_standardized = (
    X_test_weighted - feature_means
) / feature_stds

# Add a bias column AFTER standardization.
X_train_lr = np.column_stack((
    np.ones(X_train_standardized.shape[0]),
    X_train_standardized
))

X_test_lr = np.column_stack((
    np.ones(X_test_standardized.shape[0]),
    X_test_standardized
))

print("Training matrix with bias:", X_train_lr.shape)
print("Testing matrix with bias:", X_test_lr.shape)
print("Mean of first standardized feature:",
      X_train_standardized[:, 0].mean())
print("Standard deviation of first standardized feature:",
      X_train_standardized[:, 0].std())

assert X_train_lr.shape == (5000, 2001)
assert X_test_lr.shape == (1000, 2001)



# %% [markdown]
# ### 3.3 Logistic Regression implementation

# %%
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.loss_history = []

    @staticmethod
    def sigmoid(z):
        # Clipping prevents numerical overflow in exp().
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def binary_cross_entropy(y_true, probabilities):
        # Clipping prevents log(0).
        eps = 1e-12
        probabilities = np.clip(probabilities, eps, 1.0 - eps)

        loss = -np.mean(
            y_true * np.log(probabilities)
            + (1 - y_true) * np.log(1 - probabilities)
        )
        return float(loss)

    def fit(self, X, y, print_every=100):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n_samples, n_features = X.shape

        # Initialize all coefficients to zero.
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.loss_history = []

        for iteration in range(self.n_iterations):
            # Forward pass
            scores = X @ self.weights
            probabilities = self.sigmoid(scores)

            # Calculate BCE
            loss = self.binary_cross_entropy(y, probabilities)
            self.loss_history.append(loss)

            # Compute gradient
            gradient = (X.T @ (probabilities - y)) / n_samples

            # Gradient Descent update
            self.weights -= self.learning_rate * gradient

            if (
                iteration == 0
                or (iteration + 1) % print_every == 0
                or iteration == self.n_iterations - 1
            ):
                print(
                    f"Iteration {iteration + 1:4d}/"
                    f"{self.n_iterations} | "
                    f"BCE Loss: {loss:.6f}"
                )

        return self

    def predict_proba(self, X):
        if self.weights is None:
            raise ValueError("Train the model before making predictions.")

        return self.sigmoid(np.asarray(X) @ self.weights)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)



# %% [markdown]
# ### 3.4 Train Logistic Regression

# %%
logistic_model = LogisticRegressionScratch(
    learning_rate=0.1,
    n_iterations=1000
)

logistic_model.fit(
    X_train_lr,
    y_train,
    print_every=100
)

print("\nInitial BCE loss:", logistic_model.loss_history[0])
print("Final BCE loss:", logistic_model.loss_history[-1])
print(
    "Loss decreased:",
    logistic_model.loss_history[-1] < logistic_model.loss_history[0]
)



# %% [markdown]
# ### 3.5 Evaluate Logistic Regression

# %%
y_pred_lr = logistic_model.predict(X_test_lr)

lr_metrics = classification_metrics(y_test, y_pred_lr)

print("NB-Weighted Logistic Regression Results")
print("-" * 45)
print(f"Accuracy: {lr_metrics['accuracy']:.4f}")
print(f"F1-Score: {lr_metrics['f1']:.4f}")

# Compare all three models
results = {
    "Bigram Language Model": lm_metrics,
    "Naive Bayes": nb_metrics,
    "NB-Weighted Logistic Regression": lr_metrics
}

print("\nFinal Model Comparison")
print("-" * 65)
print(f"{'Model':<35} {'Accuracy':>12} {'F1-Score':>12}")

for model_name, metrics in results.items():
    print(
        f"{model_name:<35} "
        f"{metrics['accuracy']:>12.4f} "
        f"{metrics['f1']:>12.4f}"
    )



# %% [markdown]
# ### 3.6 Visualize the training loss

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(logistic_model.loss_history)
plt.xlabel("Iteration")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Logistic Regression Training Loss")
plt.grid(True)
plt.tight_layout()
plt.show()


