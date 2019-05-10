import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import ConvS2SAttention
from const import *


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class ConvEncoder(nn.Module):
    def __init__(self, in_dim, emb_dim, h_dim, n_layers, kernel_size, dropout, max_len):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel size must be odd"
        self.in_dim = in_dim # in_lang.n_words
        self.emb_dim = emb_dim
        self.h_dim = h_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(in_dim, emb_dim, padding_idx=PAD_idx)
         # SOS, EOS
        self.pos_embedding = nn.Embedding(max_len+2, emb_dim, padding_idx=PAD_idx)

        # embedding => conv input dim (= h_dim)
        self.emb2hid = nn.Linear(emb_dim, h_dim)
        # conv output => emb dim
        self.hid2emb = nn.Linear(h_dim, emb_dim)

        # Conv1d: [N, C, L]
        self.convs = nn.ModuleList()
        padding = (kernel_size - 1) // 2
        for i in range(n_layers):
            conv = nn.Conv1d(h_dim, 2*h_dim, kernel_size=kernel_size, padding=padding)
            self.convs.append(conv)

        self.dropout = nn.Dropout(dropout)
        self.scale = 0.5 ** 0.5

    def forward(self, src):
        """
        src: [B, L]
        """
        B, L = src.shape
        mask = (src != PAD_idx).long()
        # padding idx 처리가 필요한가?
        #pos_tokens = torch.arange(L, device='cuda').repeat(B, 1)
        # padded pos_tokens
        pos_tokens = mask.cumsum(dim=1) * mask + PAD_idx

        emb = self.embedding(src) # [B, L, emb_dim]
        pos_emb = self.pos_embedding(pos_tokens) # [B, L, emb_dim]
        emb = self.dropout(emb + pos_emb)

        out = self.emb2hid(emb).permute(0, 2, 1) # [B, h_dim, L]

        for conv in self.convs:
            skip_con = out
            # decoder 는 masking 이 필요 없음
            # padding 은 어차피 뒤쪽에 붙는데 decoder 는 뒤쪽을 참조하지 않음
            #out = out.masked_fill(mask.unsqueeze(1) == 0, 0.)
            out = conv(self.dropout(out)) # [B, h_dim*2, L]
            out = F.glu(out, dim=1) # [B, h_dim, L]
            # residual connection
            out = (out + skip_con) * self.scale # [B, h_dim, L]

        out = self.hid2emb(out.permute(0, 2, 1)) # encoder out z. [B, L, emb_dim]
        out = GradMultiply.apply(out, 1.0 / (2.0 * self.n_layers))
        attn_value = (out + emb) * self.scale # attention value (z+e)

        return out, attn_value, mask


class ConvDecoder(nn.Module):
    """
    ConvDecoder 는 매 타임스텝마다 굳이 이전 타임스텝들의 conv 연산을 새로 할 필요가 없음

    conv_in: [B, T, emb_dim] => [B, 1, h_dim]
        여기에 cached: [B, T-1, h_dim] 이 있다고 하면, concat(cached, conv_in) 하여 인풋을 생성하고,
        concat[-kernel_size:] 만 사용해서 conv inference 를 하면 됨.
        만약 T < kernel_size 인 경우 부족한 만큼 zero-padding.
    """
    def __init__(self, emb_dim, h_dim, out_dim, n_layers, kernel_size, dropout, max_len,
                 cache_mode='in'):
        """
        params:
            cache_mode: in / out. in 이면 논문 그대로고, out 이면 이전 타임스텝의 out 을 캐싱.
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.cache_mode = cache_mode

        self.embedding = nn.Embedding(out_dim, emb_dim, padding_idx=PAD_idx)
        self.pos_embedding = nn.Embedding(max_len+2, emb_dim, padding_idx=PAD_idx)

        self.emb2hid = nn.Linear(emb_dim, h_dim)
        self.hid2emb = nn.Linear(h_dim, emb_dim)

        # Conv1d: [N, C, L]
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            # padding = 0
            conv = nn.Conv1d(h_dim, 2*h_dim, kernel_size=kernel_size)
            attention = ConvS2SAttention(emb_dim, h_dim)
            self.layers.append(nn.ModuleList([conv, attention]))

        self.dropout = nn.Dropout(dropout)
        self.scale = 0.5 ** 0.5

        self.readout = nn.Linear(emb_dim, out_dim)

    def forward(self, tgt, enc_out, attn_value, enc_mask, cached=None, timestep=0):
        """
        tgt: [B, L] (L = tgt_len)
        enc_out: z. [B, src_len, emb_dim]
        attn_value: (z+e). [B, src_len, emb_dim]
        enc_mask: [B, src_len]
        cached: [[B, cL, h_dim] * n_layers] - input cache
        timestep: int. starting from 0.
        """
        B, L = tgt.shape
        if cached is not None:
            assert L == 1
            assert cached[0].size(2) in [0, self.kernel_size]
        mask = (tgt != PAD_idx).long()
        pos_tokens = mask.cumsum(dim=1) * mask + PAD_idx + timestep

        emb = self.embedding(tgt) # [B, L, emb_dim]
        pos_emb = self.pos_embedding(pos_tokens) # [B, L, emb_dim]
        emb = self.dropout(emb + pos_emb)

        out = self.emb2hid(emb).permute(0, 2, 1) # [B, h_dim, L]
        attn_ws = []

        for i, (conv, attention) in enumerate(self.layers):
            skip_con = out
            # dropout & future-masking by left-zero padding with caching
            out = self.dropout(out)
            ### in-caching
            if cached is not None:
                cache = cached[i] # conv inputs of just before timestep T-1
                if timestep == 0: # first timestep => non-cached yet
                    assert cache.size(2) == 0
                    conv_in = F.pad(out, [self.kernel_size-1, 0])
                else:
                    assert cache.size(2) == self.kernel_size
                    conv_in = torch.cat([cache[:, :, 1:], out], dim=2) # [B, h_dim, kernel_size]

                cached[i] = conv_in
            else:
                conv_in = F.pad(out, [self.kernel_size-1, 0])
            # conv & activation
            out = conv(conv_in) # [B, h_dim*2, L]
            out = F.glu(out, dim=1) # [B, h_dim, L]
            # attention
            # [B, h_dim, L], [B, L, src_len]
            attn_out, attn_w = attention(out, emb, enc_out, attn_value, enc_mask)
            # combine
            combine = (out + attn_out) * self.scale
            # residual
            out = (combine + skip_con) * self.scale # [B, h_dim, L]
            attn_ws.append(attn_w)

            ### out-caching
            if self.cache_mode == 'out' and cached is not None:
                cached[i] = torch.cat([conv_in[:, :, :-1], out], dim=2)

        out = self.hid2emb(out.permute(0, 2, 1)) # [B, L, emb_dim]
        out = self.dropout(out)
        out = self.readout(out) # [B, L, out_dim]

        avg_attn_w = sum(attn_ws) / len(attn_ws)

        return out, avg_attn_w, cached
