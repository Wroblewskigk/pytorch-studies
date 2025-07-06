import torch

# Yes, use torch.cuda.manual_seed() to set the GPU random seed.
# This is important for reproducibility when generating random
# numbers on the GPU

print(torch.cuda.manual_seed(1234))

# If CUDA is not available, this function is silently ignored