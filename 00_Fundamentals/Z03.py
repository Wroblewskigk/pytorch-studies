"""
Exercise 3: Perform a matrix multiplication on the tensor from 2 with another random
tensor with shape (1, 7) (hint: you may have to transpose the second tensor).
"""

import torch

tensor1 = torch.rand(7, 7)
tensor2 = torch.rand(1, 7)
result = torch.matmul(tensor1, tensor2.T)
print(result)
