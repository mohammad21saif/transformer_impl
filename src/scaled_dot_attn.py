import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """ Computes scaled dot product attention
    """
    def __init__(self, scale, dropout_rate):
        super(ScaledDotProductAttention, self).__init__()

        self.scale = scale
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask=None):
        """ query: (batch_size, n_heads, query_len, head_dim)
            key: (batch_size, n_heads, key_len, head_dim)
            value: (batch_size, n_heads, value_len, head_dim)
            mask: (batch_size, 1, 1, source_seq_len) for source mask
                  (batch_size, 1, target_seq_len, target_seq_len) for target mask
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) # (batch_size, n_heads, query_len, value_len)
        scores = scores/self.scale # (batch_size, n_heads, query_len, value_len)

        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf')) # (batch_size, n_heads, query_len, value_len)
        
        attn_probs = nn.Softmax(scores) # (batch_size, n_heads, query_len, value_len)
        output = torch.matmul(self.dropout(attn_probs), value)

        return output

