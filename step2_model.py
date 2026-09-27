"""Step 2: Define the MLP image classifier."""
import torch
import torch.nn as nn

from step1_data import get_loaders, get_device


class ImageClassifier(nn.Module):
    def __init__(self, n_inputs=28 * 28, n_hidden1=300, n_hidden2=100,
                 n_classes=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_classes),
        )

    def forward(self, X):
        return self.mlp(X)  # raw logits; CrossEntropyLoss applies softmax


if __name__ == "__main__":
    device = get_device()
    torch.manual_seed(42)
    model = ImageClassifier().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", n_params)

    train_loader, valid_loader, test_loader, class_names = get_loaders()
    X_batch, y_batch = next(iter(train_loader))
    with torch.no_grad():
        logits = model(X_batch.to(device))
    print("Input batch shape:", X_batch.shape)
    print("Output shape:", logits.shape)
