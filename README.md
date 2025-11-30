# Assignment-5


# Q1. Compute Scaled Dot-Product Attention (Python)

## Overview

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
