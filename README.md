# Assignment-5



Overview

Scaled dot-product attention computes how much each token should attend to others in a sequence.
Given Query (Q), Key (K), and Value (V) matrices, the attention mechanism:

Attention
(
𝑄
,
𝐾
,
𝑉
)
=
softmax
(
𝑄
𝐾
𝑇
𝑑
𝑘
)
𝑉
Attention(Q,K,V)=softmax(
d
k
	​

	​

QK
T
	​

)V
Steps Implemented

Compute similarity scores: 
𝑄
𝐾
𝑇
QK
T

Scale by 
𝑑
𝑘
d
k
	​

	​


Apply softmax normalization

Use attention weights to compute the context vector

Return both attention weights and context

Files Included

attention_numpy.py — implementation of scaled dot-product attention

Includes sample test with random Q, K, V inputs

Expected Output

Attention weights shape: (batch, seq_len, seq_len)

Context vector shape: (batch, seq_len, d_v)
