"""
Exercise 7: Create a multi-class dataset using the spirals data creation function
from CS231n (see below for the code).

- Construct a model capable of fitting the data (you may need a combination of linear
and non-linear layers).
- Build a loss function and optimizer capable of handling multi-class data (optional
extension: use the Adam optimizer instead of SGD, you may have to experiment with
different values of the learning rate to get it working).
- Make a training and testing loop for the multi-class data and train a model on it to
reach over 95% testing accuracy (you can use any accuracy measuring function here that
you like).
- Plot the decision boundaries on the spirals dataset from your model predictions, the
plot_decision_boundary() function should work for this dataset too.
"""

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# Points per class
N = 100
# Dimensionality
D = 2
# Number of classes
K = 3
X = np.zeros((N*K, D))
y = np.zeros(N*K, dtype='uint8')
for j in range(K):
    ix = range(N*j, N*(j+1))
    radius = np.linspace(0.0, 1, N)
    theta = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
    X[ix] = np.c_[radius * np.sin(theta), radius * np.cos(theta)]
    y[ix] = j

# noinspection PyUnresolvedReferences
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.title("Spiral Dataset")
plt.show()

class SpiralNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=100, output_dim=3):
        super(SpiralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

model = SpiralNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# noinspection PyShadowingNames
def accuracy(predictions, labels):
    return (predictions == labels).float().mean().item()

epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0 or epoch == epochs - 1:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predictions = torch.max(test_outputs, 1)
            acc = accuracy(predictions, y_test)
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Test Acc: {acc * 100:.2f}%")


# noinspection PyShadowingNames
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    with torch.no_grad():
        predictions = model(torch.tensor(grid, dtype=torch.float32))
        # noinspection PyArgumentList
        predictions = torch.argmax(predictions, axis=1).numpy()
    # noinspection PyUnresolvedReferences
    plt.contourf(xx,
                 yy,
                 predictions.reshape(xx.shape),
                 cmap=plt.cm.Spectral,
                 alpha=0.5)
    # noinspection PyUnresolvedReferences
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.title("Decision Boundary")
    plt.show()

plot_decision_boundary(model, X, y)
