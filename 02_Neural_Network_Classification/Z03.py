"""
Exercise 3: Set up a binary classification compatible loss function and optimizer to
use when training the model.
"""

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim

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

class BinaryMoonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

model = BinaryMoonModel()

loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
