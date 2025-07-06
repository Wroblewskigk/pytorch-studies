import torch

torch.manual_seed(0)
tensor1_seed0 = torch.rand(7, 7)
print(tensor1_seed0)
tensor2_seed0 = torch.rand(1, 7)
print(tensor2_seed0)
result_seed0 = torch.matmul(tensor1_seed0, tensor2_seed0.T)
print(result_seed0)