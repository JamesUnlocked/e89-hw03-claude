# Fashion MNIST Image Classifier with PyTorch

The "Building an Image Classifier with PyTorch" example from Géron, *Hands-On
Machine Learning with Scikit-Learn and PyTorch* (2025), chapter 10, built one
script at a time. Each script imports what it needs from the earlier ones.

## Setup

```
pip install -r requirements.txt
```

## Scripts (run in this order)

| # | Script | What it does | Run |
|---|--------|--------------|-----|
| 1 | `step1_data.py` | Downloads Fashion MNIST to `datasets/`, splits 55,000/5,000 train/valid, and provides `get_loaders()` and `get_device()`. | `python step1_data.py` |
| 2 | `step2_model.py` | Defines `ImageClassifier`, an MLP (784 → 300 → 100 → 10) that returns raw logits; prints the parameter count and output shape. | `python step2_model.py` |
| 3 | `step3_train.py` | Trains with SGD (lr=0.1) and cross-entropy for 20 epochs, then writes `history.json` and `model.pt`. | `python step3_train.py` |
| 4 | `step4_plot_accuracy.py` | Plots training and validation accuracy from `history.json` and saves `training_accuracy.png`. | `python step4_plot_accuracy.py` |
| 5 | `step5_evaluate.py` | Loads `model.pt`, reports test accuracy, and shows predictions, probabilities and top-4 classes for 3 validation images. | `python step5_evaluate.py` |

Steps 4 and 5 need the files that step 3 writes, so run step 3 first.
`model.pt` and `datasets/` are not committed.
