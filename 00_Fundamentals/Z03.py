import torch

tensor1 = torch.rand(7, 7)
tensor2 = torch.rand(1, 7)
result = torch.matmul(tensor1, tensor2.T)
print(result)
