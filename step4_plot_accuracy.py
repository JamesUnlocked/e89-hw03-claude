"""Step 4: Plot training and validation accuracy per epoch."""
import json

import matplotlib.pyplot as plt

TRAIN_COLOR = "#2a78d6"  # blue
VALID_COLOR = "#eb6834"  # orange


def load_history(path="history.json"):
    with open(path) as f:
        return json.load(f)


def plot_accuracy(history, path="training_accuracy.png"):
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_acc"], color=TRAIN_COLOR, linewidth=2,
            marker="o", markersize=6, label="Training accuracy")
    ax.plot(epochs, history["valid_acc"], color=VALID_COLOR, linewidth=2,
            marker="s", markersize=6, label="Validation accuracy")
    ax.set_title("Fashion MNIST MLP: Accuracy per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(list(epochs))
    ax.grid(True, color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


if __name__ == "__main__":
    history = load_history()
    plot_accuracy(history)
    print("Saved training_accuracy.png")
    plt.show()
