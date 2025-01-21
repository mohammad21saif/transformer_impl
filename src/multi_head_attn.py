import torch
import torch.nn as nn
import torch.nn.functional as F
from scaled_dot_attn import ScaledDotProductAttention
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.d_k = self.d_v = d_model//n_heads

        self.W_q = nn.Linear(d_model, self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.d_v, bias=False)
        self.W_o = nn.Linear(n_heads*self.d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(np.sqrt(self.d_k), dropout_rate)


    def split_heads(self, x):
        """
        x: (batch_size, seq_length, d_model)
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, seq_len, d_k)

        return x # (batch_size, n_heads, seq_len, head_dim)


    def group_heads(self, x):
        """
        x: (batch_size, seq_length, d_model)
        """
        batch_size = x.size(0)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)

        return x # (batch_size, seq_len, d_model)


    def forward(self, query, key, value, mask=None):
        """
        query: (batch_size, query_len, d_model)
        key: (batch_size, key_len, d_model)
        value: (batch_size, value_len, d_model)
        mask: (batch_size, 1, source_seq_len) for source mask
            (batch_size, target_seq_len, target_seq_len) for target mask
        """

        Q = self.split_heads(self.W_q(query)) # (batch_size, n_heads, query_len, d_model)
        K = self.split_heads(self.W_k(key)) # (batch_size, n_heads, key_len, d_model)
        V = self.split_heads(self.W_v(value)) # (batch_size, n_heads, value_len, d_model)

        if mask is not None:
            mask = mask.unsqueeze(1)

        x, attn = self.attention.forward(Q, K, V, mask) # x: (batch_size, seq_len, d_model), attn: (batch_size, n_heads, query_len, value_len)
        x = self.group_heads(x) # (batch_size, seq_len, d_model)
        x = self.W_o(x)

        return x, attn
