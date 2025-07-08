"""
Exercise 10: Make predictions using your trained model and visualize at least 5
of them comparing the prediction to the target label.
"""

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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

class TinyVGG(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10*14*14, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)
        return x

model = TinyVGG(in_channels=1, num_classes=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)

fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    axes[i].imshow(images[i].cpu().squeeze(), cmap="gray")
    axes[i].set_title(f"Pred: {predictions[i].item()}, True: {labels[i].item()}")
    axes[i].axis('off')
plt.show()
