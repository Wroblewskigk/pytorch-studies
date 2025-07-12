"""
Exercise 4: Create training and testing functions for model_0 from
https://www.learnpytorch.io/04_pytorch_custom_datasets/
"""
import torch


def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        # noinspection PyPep8Naming
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        predictions = torch.argmax(y_pred, dim=1)
        train_acc += (predictions == y).sum().item() / len(y)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(model, dataloader, loss_fn, device):
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            # noinspection PyPep8Naming
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            # Calculate accuracy
            predictions = torch.argmax(y_pred, dim=1)
            test_acc += (predictions == y).sum().item() / len(y)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc
