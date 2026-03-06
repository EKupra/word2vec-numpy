# Word2Vec in Pure NumPy (Skip-gram with Negative Sampling)

This project implements the core training loop of **Word2Vec** using only **NumPy**.  
No PyTorch, TensorFlow, or other ML frameworks are used.

Model variant: **Skip-gram with Negative Sampling**  
Dataset: *The Adventures of Sherlock Holmes* (Project Gutenberg, public domain)

## What This Implements

For each (center word, context word) pair, the code performs:

**Forward pass**
- score = dot(center_vector, context_vector)
- probability = sigmoid(score)

**Loss**
- Positive pair (target = 1):  
  `-log(sigmoid(score_pos))`
- Negative pairs (target = 0):  
  `-log(1 - sigmoid(score_neg))`

**Gradient**
- Uses the identity:  
  `dL/dscore = probability - target`
- Positive: `error = prob - 1`
- Negative: `error = prob - 0`

**Parameter Update**
- Stochastic Gradient Descent (SGD):parameter -= learning_rate * gradient
  
Two embedding matrices are trained:
- `W_in` (center word embeddings)
- `W_out` (context word embeddings)

## Negative Sampling

Negative words are sampled from: P(word) ∝ count(word)^0.75

## Preprocessing

- Removed Gutenberg header and footer
- Lowercased text
- Removed non-alphabetic characters
- Removed very rare words (min frequency = 5)
- Window size = 2

## Hyperparameters

- Embedding dimension: 50  
- Window size: 2  
- Negative samples per positive pair: 5  
- Learning rate: 0.005  
- Epochs: 2  
- Minimum word frequency: 5  

Random seeds are fixed for reproducibility.

## How to Run

1. Download *The Adventures of Sherlock Holmes* (UTF-8) from Project Gutenberg  
   https://www.gutenberg.org/ebooks/1661

2. Save the file as: sherlock.txt in the same directory as the script.

3. Install dependencies: pip install -r requirements.txt

4. Run: python Word2vec.py
