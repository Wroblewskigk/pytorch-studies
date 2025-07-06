"""
Exercise 10: Make a random tensor with shape (1, 1, 1, 10) and then create a new
tensor with all the 1 dimension removed to be left with a tensor of shape (10).
Set the seed to 7 when you create it and print out the first tensor, and it's shape
as well as the second tensor, and it's shape.
"""

import torch

torch.manual_seed(1234)

tensor_4d = torch.rand(1, 1, 1, 10)
tensor_1d = tensor_4d.squeeze()

print("Original tensor:", tensor_4d)
print("Original shape:", tensor_4d.shape)
print("Squeezed tensor:", tensor_1d)
print("Squeezed shape:", tensor_1d.shape)
