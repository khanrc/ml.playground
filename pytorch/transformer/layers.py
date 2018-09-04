""" Layers
    - EncoderLayer
    - DecoderLayer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sublayers


class EncoderLayer(nn.Module):
    """ EncoderLayer
    Multi-head attention + pos-wise feed forward
    """
    def __init__(self, d_model, d_ff, h, drop_rate=0.1):
        """
        Args:
            d_model
            d_ff: ffn hidden layer size (d_model => d_ff => d_model)
            h: #heads
            dropout
        """
        super().__init__()
        self.slf_attn = sublayers.MultiHeadAttention(h, d_model, drop_rate)
        self.ffn = sublayers.PositionWiseFeedForward(d_model, d_ff, drop_rate)

    def forward(self, x, mask):
        out = self.slf_attn(x, x, x, mask)
        out = self.ffn(out)

        return out


class DecoderLayer(nn.Module):
    """ DecoderLayer
    Masked multi-head attention + multi-head attention with memory + pos-wise feed forward
    """
    def __init__(self, d_model, d_ff, h, drop_rate=0.1):
        super().__init__()
        self.slf_attn = sublayers.MultiHeadAttention(h, d_model, drop_rate)
        self.enc_attn = sublayers.MultiHeadAttention(h, d_model, drop_rate)
        self.ffn = sublayers.PositionWiseFeedForward(d_model, d_ff, drop_rate)

    def forward(self, memory, x, mem_mask, slf_mask):
        """
        memory: encoder last hidden representations (last outputs)
        x: decoder inputs = target sentence
        mem_mask: memory mask (padding mask)
        tgt_mask: target sentence (inputs) mask (padding + only-left)
        """
        out = self.slf_attn(x, x, x, slf_mask)
        out = self.enc_attn(x, memory, memory, mem_mask)
        out = self.ffn(out)

        return out
        m = memory
