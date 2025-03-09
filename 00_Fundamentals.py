import torch

# Scalar
scalar = torch.tensor(7)
print (scalar)
print (scalar.dtype)
print (scalar.ndim)
print (scalar.item())

# Vector
vector = torch.tensor([1, 2, 3])
print (vector)
print (vector.dtype)
print (vector.ndim)
print (vector.shape)

# MATRIX
MATRIX = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print (MATRIX)
print (MATRIX.dtype)
print (MATRIX.ndim)
print (MATRIX.shape)

# TENSOR
TENSOR = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print (TENSOR)
print (TENSOR.dtype)
print (TENSOR.ndim)
print (TENSOR.shape)

"""Note: You might've noticed me using lowercase letters for scalar 
and vector and uppercase letters for MATRIX and TENSOR. This was on purpose. 
In practice, you'll often see scalars and vectors denoted as lowercase 
letters such as y or a. And matrices and tensors denoted as uppercase 
letters such as X or W.

You also might notice the names matrix and tensor used interchangeably. 
This is common. Since in PyTorch you're often dealing with torch.Tensors 
(hence the tensor name), however, the shape and dimensions of what's 
inside will dictate what it actually is."""

# Random TENSOR
randomTENSOR = torch.rand(size=(5, 4))
print (randomTENSOR)
print (randomTENSOR.dtype)
print (randomTENSOR.ndim)

# Zero TENSOR
ZEROS = torch.zeros(size=(5, 5))
print (ZEROS)
print (ZEROS.dtype)
print (ZEROS.ndim)
"""Analogicznie możesz zastosować torch.ones()"""

# Torch arange
ARANGE = torch.arange(0, 100, 10)
print (ARANGE)
print (ARANGE.dtype)
print (ARANGE.ndim)
print (ARANGE.shape)

# Similar TENSOR
TEN_ZEROS = torch.zeros_like(ARANGE)
print (TEN_ZEROS)
print (TEN_ZEROS.dtype)
print (TEN_ZEROS.ndim)
print (TEN_ZEROS.shape)
"""Analogicznie możesz zastosować torch.ones()"""

# Specific TENSOR
FLOAT32_TENSOR = torch.tensor([3.0, 6.0],
                              dtype=None,
                              device=None,
                              requires_grad=False)
print (FLOAT32_TENSOR)
print (FLOAT32_TENSOR.dtype)
print (FLOAT32_TENSOR.ndim)
print (FLOAT32_TENSOR.shape)
print (FLOAT32_TENSOR.device)

# Operations on TENSORS
TENSOR_TWO = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print (TENSOR_TWO)
TENSOR_TWO = TENSOR_TWO * 10
print (TENSOR_TWO)

# MATRIX multiplication
RESULT = torch.matmul(TENSOR_TWO, TENSOR)
print (TENSOR)
print (TENSOR_TWO)
print (RESULT)
print (RESULT.dtype)
print (RESULT.ndim)
print (RESULT.shape)
"""You can do matrix multiplication by hand but it's not recommended.
The in-built torch.matmul() method is faster."""

# Transposed MATRIX
TENSOR_A = torch.tensor([[1, 2], [3, 4], [5, 6]])
TENSOR_B = torch.tensor([[7, 8], [9, 10], [11, 12]])
print (TENSOR_A)
print (TENSOR_B.T)
print (torch.mm(TENSOR_A, TENSOR_B.T))
"""torch.mm() and torch.matmul() perform the same operation"""

"""Note: A matrix multiplication like this is also referred 
to as the dot product of two matrices."""

# Feed-forward layer with torch.nn.Linear()
"""Since the linear layer starts with a random weights matrix, let's 
make it reproducible (more on this later)"""
torch.manual_seed(42)
"""This uses matrix multiplication"""
"""in_features -> matches inner dimension of input"""
"""out_features -> describes outer value"""
LINEAR = torch.nn.Linear(in_features=2,
                         out_features=6)
print (LINEAR)
X = FLOAT32_TENSOR
print (X)
OUTPUT = LINEAR(X)
print(f"Input shape: {X.shape}\n")
print(f"Output:\n{OUTPUT}\n\nOutput shape: {OUTPUT.shape}")

# Min, max, mean, sum, etc.
X = torch.arange(0, 100, 10)
print (X)
print (X.dtype)
print(f"Minimum: {X.min()}")
print(f"Maximum: {X.max()}")
print(f"Mean: {X.type(torch.float32).mean()}")
print(f"Sum: {X.sum()}")
"""There are also torch methods torch.max(x), 
torch.min(x), torch.mean(x.type(torch.float32)), torch.sum(x)"""

# Positional min/max
tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")
"""Returns index of max and min values"""

# Reshaping, stacking, squeezing and unsqueezing
"""
Method	                        One-line description

torch.reshape(input, shape)	    Reshapes input to shape (if compatible), 
                                can also use torch.Tensor.reshape().
Tensor.view(shape)	            Returns a view of the original tensor in 
                                a different shape but shares the same data 
                                as the original tensor.
torch.stack(tensors, dim=0)	    Concatenates a sequence of tensors along 
                                a new dimension (dim), all tensors must 
                                be same size.
torch.squeeze(input)	        Squeezes input to remove all the dimenions 
                                with value 1.
torch.unsqueeze(input, dim)	    Returns input with a dimension value of 1 
                                added at dim.
torch.permute(input, dims)	    Returns a view of the original input with 
                                its dimensions permuted (rearranged) to 
                                dims.
"""