import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")

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

max_val = result_gpu.max()
min_val = result_gpu.min()
print(max_val, min_val)

max_idx = result_gpu.argmax()
min_idx = result_gpu.argmin()
print(max_idx, min_idx)