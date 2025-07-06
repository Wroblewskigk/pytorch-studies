"""
Exercise 7: Perform a matrix multiplication on the tensors you created in 6 (again,
you may have to adjust the shapes of one of the tensors).
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

result_gpu = torch.matmul(tensor1_seed0, tensor2_seed0.T)
print(result_gpu)
print(result_gpu.device)
