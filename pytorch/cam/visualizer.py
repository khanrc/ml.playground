import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np


class Heatmap(object):
    def __init__(self, X, activ_map, colormap=cv2.COLORMAP_JET):
        """
        make (transposed) X, (resized) activ_map, heatmap, and overlay map
        """
        # to numpy ndarray
        if isinstance(X, torch.autograd.Variable):
            X = X.data.cpu().numpy()
        elif torch.is_tensor(X):
            X = X.cpu().numpy()

        # (N)CHW => HWC
        X = X.squeeze().transpose(1, 2, 0)

        # resize
        activ_map = cv2.resize(activ_map, tuple(X.shape[:2]), interpolation=cv2.INTER_LINEAR)
        # heatmap
        activ_map_255 = np.uint8(activ_map*255)
        heatmap_255 = cv2.applyColorMap(activ_map_255, colormap)
        heatmap_255 = cv2.cvtColor(heatmap_255, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap_255) / 255.
        # overlay
        overlay = X*0.5 + heatmap*0.3

        self.X = X
        self.activ_map = activ_map
        self.heatmap = heatmap
        self.overlay = overlay

    def figure4(self):
        """
        Heatmap can be acqired using matplotlib like:
          `plt.imshow(activ_map, vmin=0., vmax=1., cmap=plt.cm.hot|jet, interpoliation='bilinear')`
        but it is hard to convert to ndarray.
        """
        fig = plt.figure(figsize=(12, 4))
        axes = []
        for i in range(1, 5):
            ax = fig.add_subplot(1, 4, i)
            axes.append(ax)
            ax.axis('off')
        #plt.jet()
        
        # org
        axes[0].imshow(self.X, vmin=0., vmax=1., cmap='Greys_r')

        # hot
        axes[1].imshow(self.activ_map, vmin=0., vmax=1., cmap=plt.cm.hot)
        
        # jet + colorbar
        heatmap_im = axes[2].imshow(self.heatmap)
        plt.colorbar(heatmap_im, fraction=0.046, pad=0.04, ax=axes[2])
        
        # overlay
        axes[3].imshow(self.overlay)
        
        return fig
