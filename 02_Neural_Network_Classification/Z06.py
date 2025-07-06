"""
Exercise 6: Replicate the Tanh (hyperbolic tangent) activation function in pure PyTorch.

- Feel free to reference the ML cheatsheet website for the formula.
"""

import torch
import matplotlib.pyplot as plt

# noinspection PyShadowingNames
def custom_tanh(x):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

x = torch.linspace(-3, 3, 100)
plt.plot(x.numpy(), torch.tanh(x).numpy(), label='torch.tanh')
plt.plot(x.numpy(), custom_tanh(x).numpy(), '--', label='custom_tanh')
plt.legend()
plt.title('Custom Tanh vs torch.tanh')
plt.show()

max_diff = torch.abs(torch.tanh(x) - custom_tanh(x)).max().item()
print(f"Maximum difference between torch.tanh and custom_tanh: {max_diff:.8f}")
