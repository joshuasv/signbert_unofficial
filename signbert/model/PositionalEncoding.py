import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    A PyTorch module for adding positional encoding to input sequences.

    Positional encoding is used in transformers to give them awareness of the 
    order of elements in a sequence. This is especially important in tasks where 
    the relative positions of elements carry significant meaning.

    Attributes:
    dropout (nn.Dropout): Dropout layer for regularization.
    pe (Tensor): The positional encoding tensor.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize the PositionalEncoding module.

        Parameters:
        d_model (int): The dimension of the embeddings (and therefore the positional encodings).
        dropout (float): The dropout rate. Default is 0.1.
        max_len (int): The maximum length of the input sequences. Default is 5000.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Create a positional encoding matrix with sinusoidal functions
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Register pe as a buffer so it's not considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass of the PositionalEncoding module.

        Parameters:
        x (Tensor): The input tensor to which positional encoding will be added.

        Returns:
        Tensor: The input tensor with positional encoding added.
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)