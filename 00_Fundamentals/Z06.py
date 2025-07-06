"""
Exercise 6: Create two random tensors of shape (2, 3) and send them both to the
GPU (you'll need access to a GPU for this). Set torch.manual_seed(1234) when creating
the tensors (this doesn't have to be the GPU random seed).
"""

import torch

# noinspection DuplicatedCode
torch.manual_seed(1234)
tensor1_seed0 = torch.rand(2, 3)
tensor2_seed0 = torch.rand(2, 3)

if torch.cuda.is_available():
    tensor1_seed0 = tensor1_seed0.to('cuda')
    tensor2_seed0 = tensor2_seed0.to('cuda')

print(tensor1_seed0)
print(tensor1_seed0.device)

print(tensor2_seed0)
print(tensor2_seed0.device)
