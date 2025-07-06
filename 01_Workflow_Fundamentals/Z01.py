"""
Exercise 1: Getting Data Ready

- Create a straight line dataset using the linear regression formula (weight * X + bias).
- Set weight=0.3 and bias=0.9. There should be at least 100 datapoints total.
- Split the data into 80% training, 20% testing.
- Plot the training and testing data so it becomes visual.
"""

import torch
import matplotlib.pyplot as plt

# Set parameters
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
