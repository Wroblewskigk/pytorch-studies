"""
Exercise 1: Turn the code to get the data (from section 1. Get Data above) into a Python
script, such as get_data.py.

- When you run the script using python get_data.py it should check if the data already exists
and skip downloading if it does.
- If the data download is successful, you should be able to access the pizza_steak_sushi images
from the data directory.
"""

import os
import requests
import zipfile
from pathlib import Path

# Define paths
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# Check if data already exists
if image_path.is_dir():
    print(f"{image_path} directory exists. Skipping download.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)
    # Dataset URL
    url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    zip_path = data_path / "pizza_steak_sushi.zip"
    # Download the data
    with open(zip_path, "wb") as f:
        print("Downloading pizza, steak, sushi data...")
        response = requests.get(url)
        f.write(response.content)
    # Unzip the data
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...")
        zip_ref.extractall(image_path)
    # Remove zip file after extraction
    os.remove(zip_path)
    print("Data ready!")
