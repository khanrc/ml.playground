""" Models
    - Encoder
    - Decoder
    - EncDec
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import layers
import sublayers


def make_mask(x, pad_idx=0, left_only=False):
    """
    Args:
        x: sentence indices
        pad_idx: padding index
        left_only: only left-wise information is accessible
    Return:
        mask
    """
    # x: [B, N]
    mask = (x != pad_idx).unsqueeze(-2) # [B, 1, N]
    if left_only:
        N = x.size(1)
        left_only_mask = torch.ones(N, N, dtype=torch.uint8).tril() # [N, N]
        mask = mask & left_only_mask # [B, 1, N] & [N, N] => [B, 1, N, N]
        """ This part is a little confusing.
        mask is broadcasted to [B, 1, N, N] - repeated N times.
        left_only_mask is boadcasted to [B, 1, N, N] - repeated B times.

        So, for the last dim [N] in mask is broadcasted to [N, N] by repeating N times and
        the logical_and operation is applied to mask [N, N] and left_only_mask [N, N].
        And this is repeated B times, [B, 1, N, N].
        """

    return mask


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, d_ff, n_layers, n_head, max_len=5000,
                 drop_rate=0.1):
        """
        Args:
            src_vocab_size: source vocabulary size
            max_len: maximum sentence length
            d_model: model hidden units size = word vector size
            d_ff: ffn hidden units size
            n_layers: the number of stacks of layer
            n_head: the number of heads (=h)
            drop_rate: dropout rate
        """
        super().__init__()
        self.src_emb = sublayers.Embedding(src_vocab_size, d_model)
        self.pos_enc = sublayers.PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList(
            [layers.EncoderLayer(d_model, d_ff, n_head) for _ in range(n_layers)])

    def forward(self, x, mask):
        # embedding
        x = self.src_emb(x)
        # add positional vector
        x = self.pos_enc(x)
        # forward
        for layer in self.layers:
            x = layer(x, mask)

        return x


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, d_ff, n_layers, n_head, max_len=5000,
                 drop_rate=0.1):
        super().__init__()
        self.tgt_emb = sublayers.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = sublayers.PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList(
            [layers.DecoderLayer(d_model, d_ff, n_head) for _ in range(n_layers)])

    def forward(self, memory, x, mem_mask, slf_mask):
        """
        Args:
            memory: encoder last hidden representations (last outputs)
            x: decoder inputs = target sentence
            mem_mask: memory mask (padding mask)
            tgt_mask: target sentence (inputs) mask (padding + only-left)
        """
        x = self.tgt_emb(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(memory, x, mem_mask, slf_mask)

        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, max_len=5000, d_model=512, d_ff=2048,
                 n_layers=6, n_head=8, drop_rate=0.1, tgt_emb_gen_wshare=True,
                 emb_src_tgt_wshare=True):
        """
        Args:
            src_vocab_size: source vocabulary size
            tgt_vocab_size: target vocabulary size
            max_len: maximum length of sentence for both source and target
            d_model: model hidden units size = word vector size
            d_ff: feed-forward network hidden units size
            n_layers: the number of stacks of layer
            n_head: the number of head of attention
            drop_rate: dropout rate
            tgt_emb_gen_wshare: weight sharing between target embedding and generator
            emb_src_tgt_wshare: weight sharing between source and target embedding. This option
                is only available when src_vocab = tgt_vocab.
        """
        super().__init__()
        assert d_model % n_head == 0

        self.encoder = Encoder(src_vocab_size, d_model, d_ff, n_layers, n_head, max_len, drop_rate)
        self.decoder = Decoder(tgt_vocab_size, d_model, d_ff, n_layers, n_head, max_len, drop_rate)
        self.generator = sublayers.Generator(tgt_vocab_size, d_model)

        # initialization
        # [!] below code works well?
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # weight sharing
        if tgt_emb_gen_wshare:
            # should we care logit scale?
            self.generator.linear.weight = self.decoder.tgt_emb.lut.weight

        if emb_src_tgt_wshare:
            assert src_vocab_size == tgt_vocab_size,
                "Weight sharing between source embedding and " \
                "target embedding is available only when src_vocab == tgt_vocab."
            self.encoder.src_emb.lut.weight = self.decoder.tgt_emb.lut.weight

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Args:
            src: source sentence indices
            tgt: target sentence indices
            src_mask: source padding mask
            tgt_mask: target padding + left-only mask
        """
        # calc encoder hidden representation - memory
        enc_memory = self.encoder(src, src_mask)
        # calc decoder logits
        dec_logits = self.decoder(enc_memory, tgt, src_mask, tgt_mask)

        return dec_logits
