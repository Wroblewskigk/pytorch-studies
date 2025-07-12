"""
Exercise 2: Recreate the data loading functions we built in sections 1, 2, 3 and 4 of
https://www.learnpytorch.io/04_pytorch_custom_datasets/ You should have train and
test DataLoader's ready to use.
"""

import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from pathlib import Path

def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


# noinspection PyShadowingNames
def visualize_random_image(image_path):
    image_path_list = list(image_path.glob("*/*/*.jpg"))
    random_image_path = random.choice(image_path_list)
    image_class = random_image_path.parent.stem
    img = Image.open(random_image_path)
    print(f"Random image path: {random_image_path}")
    print(f"Image class: {image_class}")
    print(f"Image height: {img.height}")
    print(f"Image width: {img.width}")
    img_as_array = np.asarray(img)
    plt.figure(figsize=(10, 7))
    plt.imshow(img_as_array)
    plt.title(f"Image class: {image_class} "
              f"| Image shape: {img_as_array.shape} - [height, width, color_channels]")
    plt.axis(False)
    plt.show()

train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

image_path = Path("data/pizza_steak_sushi")
train_dir = image_path / "train"
test_dir = image_path / "test"

train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(root=test_dir, transform=test_transforms)

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)

