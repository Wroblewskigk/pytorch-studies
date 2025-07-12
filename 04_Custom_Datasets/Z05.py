"""
Exercise 5: Try training the model you made in exercise 3 from
https://www.learnpytorch.io/04_pytorch_custom_datasets/ for 5, 20 and 50 epochs,
what happens to the results?

- Use torch.optim.Adam() with a learning rate of 0.001 as the optimizer.
"""

from pathlib import Path

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from Z03 import TinyVGG
from Z04 import test_step, train_step

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = Path("data/pizza_steak_sushi")

test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

test_dir = image_path / "test"
train_dir = image_path / "train"

test_data = datasets.ImageFolder(root=test_dir, transform=test_transforms)
train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)

test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)

model_0 = TinyVGG(input_shape=3, hidden_units=10, output_shape=3).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_0.parameters(), lr=0.001)

num_epochs = 5

for epoch in range(num_epochs):
    train_loss, train_acc = train_step(model_0, train_dataloader, loss_fn, optimizer, device)
    test_loss, test_acc = test_step(model_0, test_dataloader, loss_fn, device)
    print(f"Epoch {epoch+1}: "
          f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | "
          f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
