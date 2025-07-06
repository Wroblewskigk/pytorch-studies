"""
Exercise 4: Making Predictions and Evaluating a Model (Inference)

- Make predictions with the trained model on the test data.
- Visualize these predictions against the original training and testing data.
- Note: You may need to ensure predictions are moved to CPU for plotting.
"""

import torch
from matplotlib import pyplot as plt
from torch import nn

# Set parameters
# noinspection DuplicatedCode
weight = 0.3
bias = 0.9
num_points = 100

# Create data
X = torch.linspace(0, 1, num_points).unsqueeze(1)
y = weight * X + bias

# Split data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Plot
plt.scatter(X_train, y_train, label="Train")
plt.scatter(X_test, y_test, label="Test")
plt.legend()
plt.title("Straight Line Data")
plt.show()

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True))
    def forward(self, x):
        return self.weight * x + self.bias

model = LinearRegressionModel()
print(model.state_dict())

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 300
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        model.eval()
        with torch.inference_mode():
            test_pred = model(X_test)
            test_loss = loss_fn(test_pred, y_test)
        print(f"Epoch {epoch+1}: Train loss {loss.item():.4f} | Test loss {test_loss.item():.4f}")

model.eval()
with torch.inference_mode():
    predictions = model(X_test).cpu()

plt.scatter(X_train.cpu(), y_train.cpu(), label="Train")
plt.scatter(X_test.cpu(), y_test.cpu(), label="Test")
plt.scatter(X_test.cpu(), predictions, label="Predictions", marker="x")
plt.legend()
plt.title("Model Predictions vs. Data")
plt.show()
