"""
Exercise 1: Make a binary classification dataset with Scikit-Learn's
make_moons() function.

- For consistency, the dataset should have 1000 samples and a random_state=42.
- Turn the data into PyTorch tensors. Split the data into training and test sets using
train_test_split with 80% training and 20% testing.
"""

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch

# Create the dataset
# noinspection DuplicatedCode
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
