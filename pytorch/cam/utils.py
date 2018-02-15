""" Utilities """

import torch
import torch.nn as nn
import numpy as np


def clear_hooks(model):
    """ clear all hooks from model """
    for module in model.modules():
        module._forward_hooks.clear()
        module._backward_hooks.clear()


def normalize(t, eps=1e-8):
    return (t - t.min()) / (t.max() - t.min() + eps)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
