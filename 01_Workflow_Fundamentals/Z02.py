"""
Exercise 2: Building a Model

- Build a PyTorch model by subclassing nn.Module.
- Inside should be a randomly initialized nn.Parameter() with requires_grad=True, one for
weights and one for bias.
- Implement the forward() method to compute the linear regression function used to create the dataset.
- Once constructed, make an instance of it and check its state_dict().
- Note: If you'd like to use nn.Linear() instead of nn.Parameter() you can.
"""

import torch
from torch import nn

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True))
    def forward(self, x):
        return self.weight * x + self.bias

model = LinearRegressionModel()
print(model.state_dict())
