"""
Exercise 5: Make predictions with your trained model and plot them using the
plot_decision_boundary() function created in this notebook.
"""

import numpy as np
from matplotlib import pyplot as plt
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


# noinspection PyShadowingNames
def binary_accuracy(y_pred, y_true):
    y_pred_labels = torch.round(y_pred)
    return (y_pred_labels == y_true).float().mean().item() * 100


epochs = 1000
for epoch in range(epochs):
    # Training
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    acc = binary_accuracy(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Testing
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test)
        test_acc = binary_accuracy(test_pred, y_test)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1:4d} | "
              f"Train Loss: {loss.item():.4f} | Train Acc: {acc:.2f}% | "
              f"Test Loss: {test_loss.item():.4f} | Test Acc: {test_acc:.2f}%")

    # Stop early if test accuracy > 96%
    if test_acc > 96:
        break


# noinspection PyShadowingNames
def plot_decision_boundary(model, X, y):
    # Set model to evaluation mode
    model.eval()
    # Create a meshgrid of values
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    # Prepare grid for prediction
    grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    with torch.no_grad():
        predictions = model(grid)
        predictions = torch.round(predictions)  # For sigmoid output
    predictions = predictions.reshape(xx.shape).numpy()

    # Plot contour and data points
    # noinspection PyUnresolvedReferences
    plt.contourf(xx, yy, predictions, alpha=0.2, cmap=plt.cm.coolwarm)
    # noinspection PyUnresolvedReferences
    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze().numpy(), cmap=plt.cm.coolwarm, s=40)
    plt.title("Decision Boundary")
    plt.show()

# Make predictions and plot decision boundary
plot_decision_boundary(model, X, y)
