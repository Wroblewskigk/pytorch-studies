"""
Exercise 7: Double the data you're using with your model and train it for 20 epochs,
what happens to the results?

- Note: You can use link below to scale up your Food101 dataset.
https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/04_custom_data_creation.ipynb
- You can also find the already formatted double data (20% instead of 10% subset)
dataset on GitHub, you will need to write download code like in exercise 2 to get
it into this notebook.
"""

import requests
import zipfile
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch import nn
from Z03 import TinyVGG
from Z04 import train_step, test_step

DOWNLOAD_URL = "https://github.com/mrdbourke/pytorch-deep-learning/releases/download/foodvision/pizza_steak_sushi_20_percent.zip"
DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "pizza_steak_sushi_20_percent.zip"
EXTRACT_DIR = DATA_DIR / "pizza_steak_sushi_20_percent"

# Download if not already present
if not ZIP_PATH.exists():
    print("Downloading 20% subset dataset...")
    response = requests.get(DOWNLOAD_URL)
    ZIP_PATH.write_bytes(response.content)

# Extract if not already present
if not EXTRACT_DIR.exists():
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

# Define your transforms
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])
test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Update paths to point to the new dataset
# noinspection DuplicatedCode
train_dir = EXTRACT_DIR / "train"
test_dir = EXTRACT_DIR / "test"

# Load datasets
train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(root=test_dir, transform=test_transforms)

# Create DataLoaders
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# noinspection DuplicatedCode
model_0 = TinyVGG(input_shape=3, hidden_units=20, output_shape=3).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_0.parameters(), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    train_loss, train_acc = train_step(model_0, train_dataloader, loss_fn, optimizer, device)
    test_loss, test_acc = test_step(model_0, test_dataloader, loss_fn, device)
    print(f"Epoch {epoch+1}: "
          f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | "
          f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
