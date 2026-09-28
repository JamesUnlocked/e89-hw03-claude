"""Step 5: Evaluate the saved model and inspect a few predictions."""
import torch
import torch.nn.functional as F
import torchmetrics

from step1_data import get_loaders, get_device
from step2_model import ImageClassifier
from step3_train import evaluate_tm


def load_model(path="model.pt", device="cpu"):
    model = ImageClassifier().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


if __name__ == "__main__":
    device = get_device()
    train_loader, valid_loader, test_loader, class_names = get_loaders()
    model = load_model(device=device)

    accuracy = torchmetrics.Accuracy(task="multiclass",
                                     num_classes=10).to(device)
    test_acc = evaluate_tm(model, test_loader, accuracy).item()
    print(f"Test accuracy: {test_acc:.4f}")

    X_new, y_new = next(iter(valid_loader))
    X_new, y_new = X_new[:3].to(device), y_new[:3]
    with torch.no_grad():
        y_pred_logits = model(X_new)
    y_pred = y_pred_logits.argmax(dim=1).cpu()
    print("\nPredicted class indices:", y_pred)
    print("Predicted class names:  ", [class_names[i] for i in y_pred])
    print("True class indices:     ", y_new)
    print("True class names:       ", [class_names[i] for i in y_new])

    y_proba = F.softmax(y_pred_logits, dim=1).cpu()
    print("\nSoftmax probabilities:")
    print(y_proba.round(decimals=3))

    y_top4_proba, y_top4 = torch.topk(y_proba, k=4, dim=1)
    print("\nTop-4 classes:")
    for n, (classes, probas) in enumerate(zip(y_top4, y_top4_proba)):
        top4 = ", ".join(f"{class_names[c]} ({p:.3f})"
                         for c, p in zip(classes, probas))
        print(f"  Image {n}: {top4}")
