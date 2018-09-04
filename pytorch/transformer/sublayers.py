""" Sub-layers and modules:
    - Scaled dot-product key-value attention
    - Multihead attention
    - Position-wise feed forward network
    - Positional encoding
    - Embedding
    - LinearSoftmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledAttention(nn.Module):
    """ Scaled dot-product key-value attention """
    def __init__(self, scale, drop_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout(drop_rate) if drop_rate else None
        self.scale = scale

    def forward(self, query, key, value, mask=None):
        """ Scaled dot-product key-value attention
        query: [B, Q, d_k]
        key  : [B, N, d_k]
        value: [B, N, d_v]
        N = length of sentence.

        for the multi-head case, H (=n_heads) is added like:
            query: [B, H, Q, d_k]
        """
        #d_k = query.size(-1)
        #scale = d_k ** 0.5

        # [B, Q, d_k] @ [B, d_k, N] => [B, Q, N]
        attn_score = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        if mask:
            attn_score.masked_fill_(mask == 0, -1e9)

        attn_dist = F.softmax(attn_score, dim=-1)

        # Attention dropout: which is dropped out from the original paper.
        # The current final version (v5) of Transformer paper does not have below attention dropout,
        # but which is involved in the v1 and other reference code.
        if dropout:
            attn_dist = self.dropout(attn_dist)

        # [B, Q, N] @ [B, N, d_v] => [B, Q, d_v]
        attn_output = torch.matmul(attn_dist, value)

        return attn_output, attn_dist


class MultiHeadAttention(nn.Module):
    """ Multi-head attention """
    def __init__(self, h, d_model, drop_rate=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = self.d_v = d_model // h
        self.h = h
        self.d_model = d_model
        # 3 linear projections (multi-headed)
        self.Wq = nn.Linear(d_model, h * self.d_k)
        self.Wk = nn.Linear(d_model, h * self.d_k)
        self.Wv = nn.Linear(d_model, h * self.d_v)
        # attention
        self.attention = ScaledAttention(scale=self.d_k**0.5)
        # last linear projection
        self.last_linear = nn.Linear(h * self.d_v, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, query, key, value, mask=None):
        """
        query: [B, Q, d_word]
        key  : [B, N, d_word]
        value: [B, N, d_word]
        N = length of sentence.
        d_word = word vector size = d_model
        """
        B, Q, _ = query.size()
        N = key.size(1)

        if mask:
            # same mask applied to all heads
            # add head dim
            mask.unsqueeze(1)

        # 1) linear projections of word-vectors
        query = self.Wq(query).view(B, Q, self.h, self.d_k).transpose(1, 2) # [B, H, Q, d_k]
        key   = self.Wk(key  ).view(B, N, self.h, self.d_k).transpose(1, 2) # [B, H, N, d_k]
        value = self.Wv(value).view(B, N, self.h, self.d_v).transpose(1, 2) # [B, H, N, d_v]

        # 2) scaled dot-product attention
        attn_out, attn_dist = self.attention(query, key, value, mask) # [B, H, Q, d_v], [B, H, Q, N]
        # attention distribution interface
        self.attn_dist = attn_dist

        # 3) last linear projection + dropout 
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Q, self.h*d_v) # [B, Q, d_model]
        out = self.last_linear(attn_out) # [B, Q, d_model]
        out = self.dropout(out)

        # 4) residual connection + layer norm
        out = self.layer_norm(out + query)

        return out


class PositionWiseFeedForward(nn.Module):
    """ two-layer word-wise FC """
    def __init__(self, d_model, d_ff, drop_rate=0.1):
        """ [B, Q, d_model] => [B, Q, d_ff] => [B, Q, d_model]
        nn.Linear apply FC to last dim
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(drop_rate)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out = F.relu(self.linear1(x)) # layer 1; [B, Q, d_ff]
        out = self.linear2(out) # layer 2; [B, Q, d_model]
        out = self.dropout(out)
        out = self.layer_norm(out + x)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout_rate=0.1):
        """ Positional encoding module
        max_len: maximum length of sentence

        Positional encoding vector only depends on the 'position'.
        So we can pre-compute the vectors in init phase.
        """
        super().__init__()
        # calc positional encoding
        pe = torch.empty(max_len, d_model) # [L, D]
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1) # [L, 1]
        indices = torch.arange(0, d_model, 2, dtype=torch.float32) # [D / 2]
        div = 10000. ** (indices / d_model)
        pe[:, 0::2] = torch.sin(pos / div)
        pe[:, 1::2] = torch.cos(pos / div)
        pe.unsqueeze_(0).requires_grad_(False) # expand batch dim

        """
        We can register: buffer, parameter, hook.
         - buffer: persistent state but not model parameter - like `running_mean` in BN.
         - parameter: model parameter
         - hook: forward/backward hook

        Through register buffer, tensor `pe` is now considered as model state. so,
        ```
        pos_encoder = PositionalEncoding(...)
        print(pos_encoder.pe) # accessible
        pos_encoder.to(device) # model device change
        ```
        """
        self.register_buffer('pe', pe)

    def forward(self, x):
        """ Add precomputed positional encoding vector"""
        pe_vector = self.pe[:, x_size(1)]
        return self.dropout(x + pe_vector)


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, pad_idx=0):
        """ Embedding from one-hot index to word vector
        d_model: word vector size
        """
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.scale = d_model ** 0.5

    def forward(self, x):
        return self.lut(x) * self.scale


class Generator(nn.Module):
    """ Last linear softmax on the decoder stacks """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # Generator linear weight is shared with embedding: so bias should be off.
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        logits = self.linear(x)
        return logits
