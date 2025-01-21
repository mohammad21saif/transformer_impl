import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding.
    """
    def __init__(self, dropout_rate, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        pe = torch.zeros(max_len, d_model) # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float64).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0:2] = torch.sin(position*div_term)
        pe[:, 1:2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0).transpose(0,1) # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch_size, seq_length, d_model)
        """
        x = x + self.pe[:x.size(0), :] # (batch_size, seq_length, d_model)
        x = self.dropout(x) # (batch_size, seq_length, d_model)

        return x # (batch_size, seq_length, d_model)