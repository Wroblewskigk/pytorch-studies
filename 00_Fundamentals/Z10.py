import torch

torch.manual_seed(1234)

tensor_4d = torch.rand(1, 1, 1, 10)
tensor_1d = tensor_4d.squeeze()

print("Original tensor:", tensor_4d)
print("Original shape:", tensor_4d.shape)
print("Squeezed tensor:", tensor_1d)
print("Squeezed shape:", tensor_1d.shape)
