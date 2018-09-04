""" Transformer scheduled optimizer
Basic:
    opt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    ScheduledOptimizer(opt, d_model=512, warmup=4000)
"""

import torch
import torch.nn as nn


class ScheduledOptimizer(object):
    def __init__(self, optim, d_model, warmup=4000):
        super().__init__()
        self.optim = optim
        self.warmup = warmup
        self.step = 0
        self.init_lr = d_model ** -0.5

    def step(self):
        """ update lr and parameter """
        self._step += 1
        # update lr
        cur_lr = self.init_lr * min(self._step ** -0.5, self._step * self.warmup ** -1.5)
        for p in self.optim.param_groups:
            p['lr'] = cur_lr

        # update params
        self.optim.step()

    def zero_grad(self):
        self.optim.zero_grad()


class LabelSmoothing(nn.Module):
    def __init__(self, size, pad_idx, smoothing=0.1):
        super().__init__()
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def criterion(self, y, p, n_tokens):
        """ cross-entropy for distribution + ignore padding
        CE = -sum(ylogp)
        loss = sum(CE) / n_tokens
        sum for batch and divide the loss by n_tokens
        """
        return -torch.sum(y*torch.log(p)) / n_tokens

    def smooth_dist(self, size, target_index):
        dist = torch.empty(size, dtype=torch.float32)
        dist.fill_(self.smoothing / (size-2))
        dist.scatter_(dim=1, index=target_index.unsqueeze(1), 1-self.smoothing)
        dist[:, self.pad_idx] = 0.
        mask = (target != pad_idx)
        if mask.dim() > 1:
            dist.index_fill_(0, mask.squeeze(), 0.0)

        return dist
