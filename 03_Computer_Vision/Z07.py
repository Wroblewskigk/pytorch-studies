"""
Exercise 7: Turn the MNIST train and test datasets into dataloaders using
torch.utils.data.DataLoader, set the batch_size=32.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# noinspection DuplicatedCode
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root=".",
                               train=True,
                               download=True,
                               transform=transform)

test_dataset = datasets.MNIST(root=".",
                              train=False,
                              download=True,
                              transform=transform)

fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    img, label = train_dataset[i]
    axes[i].imshow(img.squeeze(), cmap="gray")
    axes[i].set_title(f"Label: {label}")
    axes[i].axis('off')
plt.show()

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)