"""Step 3: Train the classifier and record the learning curves."""
import json

import torch
import torch.nn as nn
import torchmetrics

from step1_data import get_loaders, get_device
from step2_model import ImageClassifier


def evaluate_tm(model, data_loader, metric):
    model.eval()
    metric.reset()
    device = next(model.parameters()).device
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)
    return metric.compute()


def train(model, optimizer, loss_fn, metric, train_loader, valid_loader,
          n_epochs):
    device = next(model.parameters()).device
    history = {"train_loss": [], "train_acc": [], "valid_acc": []}
    for epoch in range(n_epochs):
        model.train()
        metric.reset()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)
        mean_loss = total_loss / len(train_loader)
        history["train_loss"].append(mean_loss)
        history["train_acc"].append(metric.compute().item())
        history["valid_acc"].append(
            evaluate_tm(model, valid_loader, metric).item())
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_loss'][-1]:.4f}, "
              f"train metric: {history['train_acc'][-1]:.4f}, "
              f"valid metric: {history['valid_acc'][-1]:.4f}")
    return history


if __name__ == "__main__":
    device = get_device()
    train_loader, valid_loader, test_loader, class_names = get_loaders()
    torch.manual_seed(42)
    model = ImageClassifier().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(task="multiclass",
                                     num_classes=10).to(device)
    history = train(model, optimizer, loss_fn, accuracy, train_loader,
                    valid_loader, n_epochs=20)
    with open("history.json", "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), "model.pt")
    print("Saved history.json and model.pt")
