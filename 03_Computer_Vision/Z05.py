"""
Exercise 5: Load the torchvision.datasets.MNIST() train and test datasets.
"""

from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root=".",
                               train=True,
                               download=True,
                               transform=transform)

test_dataset = datasets.MNIST(root=".",
                              train=False,
                              download=True,
                              transform=transform)
