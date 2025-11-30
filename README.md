## Assignment-5


## Q1. Compute Scaled Dot-Product Attention (Python)

# Overview

Scaled dot-product attention computes how much each token should attend to others in a sequence.
Given Query (Q), Key (K), and Value (V) matrices, the attention mechanism:

Attention(Q, K, V) = softmax( (Q Kᵀ) / sqrt(dₖ) ) V

# Steps Implemented

- Compute similarity scores: Q K^T  
- Scale by sqrt(d_k)  
- Apply softmax normalization  
- Multiply softmax weights with V to compute context  
- Return both attention weights and context vector  

# Files Included

attention_numpy.py — implementation of scaled dot-product attention

Includes sample test with random Q, K, V inputs

# Expected Output

Attention weights shape: (batch, seq_len, seq_len)

Context vector shape: (batch, seq_len, d_v)


## Q2. Implement Simple Transformer Encoder Block (PyTorch)

# Overview

This task implements a simplified Transformer Encoder block from scratch using PyTorch, including:

1.Multi-head self-attention
2.Feed-forward network
3.Residual connections
4.Layer normalization

# Architecture Components

# 1. Multi-Head Self-Attention
Splits the input embeddings into multiple heads, performs self-attention independently in each head, and then concatenates the results back together.

---

# 2. Feed-Forward Network (FFN)
A position-wise feed-forward network applied to each token independently.  
It consists of two linear layers with a ReLU activation in between.

The mathematical form:

FFN(x) = ReLU(xW1 + b1)W2 + b2

# Steps Implemented

- Compute similarity scores: Q K^T  
- Scale by sqrt(d_k)  
- Apply softmax normalization  
- Use attention weights to compute the context vector  
- Return both attention weights and context

# Files included
- attention_numpy.py — implementation of scaled dot-product attention  
- Includes sample test with random Q, K, V inputs

# Expected output
Attention weights shape: (batch, seq_len, seq_len)  
Context vector shape: (batch, seq_len, d_v)  


