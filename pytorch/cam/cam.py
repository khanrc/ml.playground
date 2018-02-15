""" CAM PyTorch implementation """

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import clear_hooks, normalize


class CAM(object):
    def __init__(self, model):
        """ The model should have ... => GAP => Linear structure at the end """
        self.model = model
        
        # clear hooks
        clear_hooks(model)
        # add hooks for last featuremap
        self.add_hook(model)

    def forward(self, X):
        self.last_fmap = None

        return self.model(X)

    def activation_map(self, index):
        """ compute activation map """
        assert self.last_fmap is not None

        # 1. get weights of last linear layer
        for module in list(self.model.modules())[::-1]:
            if isinstance(module, nn.Linear):
                last_linear_name, last_linear_weights = list(module.named_parameters())[0]
                assert 'weight' in last_linear_name
                break
        
        # 2. get weights of each channels
        weights = last_linear_weights[index]

        # 3. weighted feature maps
        weighted_fmaps = weights.view(-1, 1, 1) * self.last_fmap.squeeze() # C11 * CKK

        # 4. activation map
        activ_map = weighted_fmaps.sum(0) # 1KK
        
        # 5. normalize
        activ_map = normalize(activ_map)

        return activ_map.data.cpu().numpy()

    def add_hook(self, model):
        """ add hook to get last featuremap """
        def fw_hook(module, inp, out):
            self.last_fmap = out # variable

        for name, module in list(model.named_modules())[::-1]:
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(fw_hook)
                break
