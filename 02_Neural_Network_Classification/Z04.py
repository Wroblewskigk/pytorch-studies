"""
Exercise 4: Create a training and testing loop to fit the model you created in 2 to the
data you created in 1.

- To measure model accuracy, you can create your own accuracy function or use the
accuracy function in TorchMetrics.
- Train the model for long enough for it to reach over 96% accuracy.
- The training loop should output progress every 10 epochs of the model's training
and test set loss and accuracy.
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
