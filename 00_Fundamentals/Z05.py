"""
Exercise 5: Speaking of random seeds, we saw how to set it with torch.manual_seed()
but is there a GPU equivalent? (hint: you'll need to look into the documentation for
torch.cuda for this one). If there is, set the GPU random seed to 1234.
"""

import torch

# Yes, use torch.cuda.manual_seed() to set the GPU random seed.
# This is important for reproducibility when generating random
# numbers on the GPU

print(torch.cuda.manual_seed(1234))

# If CUDA is not available, this function is silently ignored