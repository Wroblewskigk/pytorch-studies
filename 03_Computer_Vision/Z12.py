"""
Exercise 12: Create a random tensor of shape [1, 3, 64, 64] and pass it through
a nn.Conv2d() layer with various hyperparameter settings (these can be any
settings you choose), what do you notice if the kernel_size parameter goes up
and down?
"""

import torch.nn as nn
import torch

x = torch.randn(1, 3, 64, 64)
for k in [3, 5, 7]:
    conv = nn.Conv2d(3, 8, kernel_size=k)
    out = conv(x)
    print(f"Kernel Size: {k}, Output Shape: {out.shape}")
