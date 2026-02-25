import numpy as np
import random
import re
from collections import Counter

random.seed(0)
np.random.seed(0)

with open("sherlock.txt", "r", encoding="utf-8") as f:
    text = f.read()

start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

start = text.find(start_marker)
end = text.find(end_marker)

if start != -1 and end != -1:
    text = text[start:end]

chapter_start = text.find("I. A SCANDAL IN BOHEMIA")
if chapter_start != -1:
    text = text[chapter_start:]

text = text.lower()
text = re.sub(r"[^a-z\s]", " ", text)
words = text.split()

roman = {"i","ii","iii","iv","v","vi","vii","viii","ix","x"}
words = [w for w in words if w not in roman]

word_counts = Counter(words)

min_freq = 5
vocab = [w for w, c in word_counts.items() if c >= min_freq]

word_to_idx = {}
for i, w in enumerate(vocab):
    word_to_idx[w] = i

idx_to_word = {i: w for w, i in word_to_idx.items()}

indexed_words = [word_to_idx[w] for w in words if w in word_to_idx]

window_size = 2
training_pairs = []

for i in range(len(indexed_words)):
    center = indexed_words[i]
    start = max(0, i - window_size)
    end = min(len(indexed_words), i + window_size + 1)

    for j in range(start, end):
        if j != i:
            training_pairs.append((center, indexed_words[j]))

V = len(vocab)
D = 50

W_in = np.random.randn(V, D) * 0.01
W_out = np.random.randn(V, D) * 0.01

def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1 / (1 + np.exp(-x))

counts = np.array([word_counts[w] for w in vocab], dtype=np.float64)
neg_sampling_probs = counts ** 0.75
neg_sampling_probs /= neg_sampling_probs.sum()

def get_negative_samples(true_idx, K):
    negatives = set()
    while len(negatives) < K:
        sample = int(np.random.choice(V, p=neg_sampling_probs))
        if sample != true_idx:
            negatives.add(sample)
    return list(negatives)

def train_step(center_idx, context_idx, lr=0.005, K=5):
    v = W_in[center_idx]
    u_pos = W_out[context_idx]

    score_pos = np.dot(v, u_pos)
    prob_pos = sigmoid(score_pos)
    loss = -np.log(prob_pos + 1e-12)

    error = prob_pos - 1
    W_in[center_idx] -= lr * (error * u_pos)
    W_out[context_idx] -= lr * (error * v)

    v = W_in[center_idx]

    negatives = get_negative_samples(context_idx, K)

    for neg_idx in negatives:
        u_neg = W_out[neg_idx]
        score_neg = np.dot(v, u_neg)
        prob_neg = sigmoid(score_neg)
        loss += -np.log(1 - prob_neg + 1e-12)

        error_neg = prob_neg
        W_in[center_idx] -= lr * (error_neg * u_neg)
        W_out[neg_idx] -= lr * (error_neg * v)

        v = W_in[center_idx]

    return loss

epochs = 2
lr = 0.005
K = 5

for epoch in range(epochs):
    random.shuffle(training_pairs)
    total_loss = 0

    for i, (c, ctx) in enumerate(training_pairs):
        total_loss += train_step(c, ctx, lr, K)

        if (i + 1) % 20000 == 0:
            print("Epoch", epoch + 1, "step", i + 1, "avg loss", total_loss / (i + 1))

    print("End epoch", epoch + 1, "avg loss", total_loss / len(training_pairs))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

def most_similar(word, top_k=10):
    if word not in word_to_idx:
        print("Word not in vocab")
        return

    w_idx = word_to_idx[word]
    w_vec = W_in[w_idx]

    sims = []
    for i in range(V):
        if i == w_idx:
            continue
        sim = cosine_similarity(w_vec, W_in[i])
        sims.append((sim, i))

    sims.sort(reverse=True)

    print("Most similar to", word)
    for sim, idx in sims[:top_k]:
        print(idx_to_word[idx], sim)

stopwords = {
    "the","and","to","of","a","in","i","it","that","is","was","he","she","you",
    "as","for","with","his","her","at","on","be","but","not","have","had",
    "mr","sir","said","well","ask","asked","why","perhaps","quite","indeed",
    "do","did","done","so","then","than","very","from","this","there","here",
    "what","when","where","who","whom","which","how","can","could","would",
    "shall","should","may","might","must","will","just","now","only","also"
}

def most_similar_filtered(word, top_k=10):
    if word not in word_to_idx:
        print("Word not in vocab")
        return

    w_idx = word_to_idx[word]
    w_vec = W_in[w_idx]

    sims = []
    for i in range(V):
        if i == w_idx:
            continue

        candidate = idx_to_word[i]
        if candidate in stopwords:
            continue

        sim = cosine_similarity(w_vec, W_in[i])
        sims.append((sim, i))

    sims.sort(reverse=True)

    print("Most similar to", word, "(filtered)")
    for sim, idx in sims[:top_k]:
        print(idx_to_word[idx], sim)

most_similar("holmes")
print()
most_similar_filtered("holmes")
print("\n")

most_similar("watson")
print()
most_similar_filtered("watson")
print("\n")

most_similar("sherlock")
print()
most_similar_filtered("sherlock")