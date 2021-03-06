""" GradCAM PyTorch implementation """

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import clear_hooks, normalize


class GradCAM(object):
    def __init__(self, model, target_layer=None):
        """
        if target_layer == None as default, the last conv layer is targetted.
        """
        self.model = model

        # clear hooks
        clear_hooks(model)
        # add hooks for last featuremap & last featuremap gradients
        self.add_hooks(model, target_layer)

    def forward(self, X):
        self.last_fmap = None
        self.last_fmap_grad = None

        return self.model(X)

    def backward(self, logits, index=-1):
        """
        default index is logits.max_index()
        """
        if index == -1:
            index = logits.max(1)[1].data[0]

        self.model.zero_grad()
        logits[:, index].backward()

    def activation_map(self):
        """ compute activation map """
        # last_fmap.size(): 1CKK
        # last_fmap_grad.size(): 1CKK
        assert (self.last_fmap is not None) and (self.last_fmap_grad is not None)

        # 1. get weights of channels (alpha)
        n_channels = self.last_fmap_grad.size(1)
        # flatten & mean: GAP of gradients
        weights = self.last_fmap_grad.view(n_channels, -1).mean(1)
        
        # 2. weighted feature maps
        weighted_fmaps = weights.view(-1, 1, 1) * self.last_fmap.squeeze() # C11 * CKK

        # 3. activation map
        activ_map = weighted_fmaps.sum(0) # 1KK
        activ_map = F.relu(activ_map)
        
        # 4. normalize
        activ_map = normalize(activ_map)

        return activ_map.data.cpu().numpy()

    def add_hooks(self, model, target_layer):
        """ 
        add hooks to get last featuremap and its gradient,
        which are used for calculating activation map.
        """
        def fw_hook(module, inp, out):
            self.last_fmap = out # variable
        def bw_hook(module, grad_in, grad_out):
            """ grad in/out is tuple
            grad_in: (activ_grad, w_grad, b_grad)
            grad_out: (activ_grad,)
            """
            self.last_fmap_grad = grad_out[0] # variable

        for name, module in list(model.named_modules())[::-1]:
            if (target_layer and name == target_layer) or \
               (not target_layer and isinstance(module, nn.Conv2d)):
                    module.register_forward_hook(fw_hook)
                    module.register_backward_hook(bw_hook)
                    break
