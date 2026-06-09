import torch
import torch.nn.functional as F
from torch import nn
from typing import Callable, Optional
import numpy as np

 
class FocalLossFunction(nn.Module):
    def __init__(self, apply_nonlin=None, gamma=2.0, alpha=None, do_bg=True, smooth=1e-5):
        super(FocalLossFunction, self).__init__()
        self.apply_nonlin = apply_nonlin        #activation function for raw logits
        self.gamma = gamma                      #focus parameter: high gamma -> less influence easy examples 
        self.do_bg = do_bg                      #include background in loss calculation
        self.smooth = smooth                    #clamping probabilities (numerical stability)

        #move alpha to the GPU
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, x, y):
        #applying nonlinearity to logits
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        #get spatial axes
        spatial_axes = tuple(range(2, x.ndim))

        #handle dimensions in the same way as nn-Unet's dice.py (for compatibility with 2D/3D/Cascade models)
        with torch.no_grad():
            # ensure y has a dimension for channel -> [B 1 D H W]
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))
            if x.shape == y.shape:
                y_onehot = y.float()
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                y_onehot.scatter_(1, y.long(), 1)

        if not self.do_bg:
            x = x[:, 1:]
            y_onehot = y_onehot[:, 1:]

        #clamp probabilities (numerical stability)
        p_t = torch.clamp(x, min=self.smooth, max=1.0 - self.smooth)

        #focal loss formula
        focal_voxel = -y_onehot * ((1.0 - p_t) ** self.gamma) * torch.log(p_t)

        #multiply with alpha (weighting factor for each class)
        if self.alpha is not None:
            alpha_t = self.alpha if self.do_bg else self.alpha[1:]
            alpha_map = alpha_t.view(1, -1, *([1] * len(spatial_axes))).to(x.device)
            focal_voxel = alpha_map * focal_voxel
        return focal_voxel.mean()