# HW03 Problem 1: Dialog Summary

A record of building the "Building an Image Classifier with PyTorch" example from
Géron's *Hands-On Machine Learning with Scikit-Learn and PyTorch* (2025),
chapter 10, one script at a time with Claude Code (Claude Opus 5.5).

- **Repository:** https://github.com/JamesUnlocked/e89-hw03-claude (branch `main`)
- **Environment:** Windows 11, Python at `C:\Anaconda\python.exe`, PyTorch 2.14.0 (CPU only), torchvision 0.29.0, torchmetrics 1.9.0, matplotlib 3.10.0
- **Device:** `get_device()` returned `cpu` on every run, because no CUDA or MPS device is available.

## Final results

| Measure | Value |
|---|---|
| Model parameters | 266,610 (784·300+300 = 235,500; 300·100+100 = 30,100; 100·10+10 = 1,010) |
| Final training accuracy (epoch 20) | 0.9287 |
| Final validation accuracy (epoch 20) | 0.8766 |
| Best validation accuracy | 0.8906 (epoch 17) |
| **Test accuracy** (epoch-20 weights) | **0.8755** |
| Final training loss (epoch 20) | 0.1881 |

## Requests and results

### 1. Step 1: data loading

**Request:** Write `step1_data.py`. It loads Fashion MNIST into `datasets/` using
`T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])`, splits the
60,000 training images into 55,000 train and 5,000 validation with
`torch.manual_seed(42)` and `random_split`, and provides `get_loaders(batch_size=32)`
and `get_device()`. `main` prints the first training sample's shape, dtype and
class name. The request also set rules for every step: name scripts
`stepN_<topic>.py`, import earlier steps, put executable code under
`if __name__ == "__main__":`, then run, fix and commit as "Step N: <topic>".

**Generated:** `step1_data.py`, commit `a38205e` ("Step 1: data"). I also added a
`get_datasets()` helper so `main` can look at one sample without building the
loaders.

**Output:** `Shape: torch.Size([1, 28, 28])`, `Dtype: torch.float32`,
`Class: Ankle boot`, `Device: cpu`.

**Issues:** The script worked on the first run. The shell command returned exit
code 1, but only because I added a `cat .gitignore` at the end and that file
didn't exist yet. It was not a problem with the script. I committed only
`step1_data.py` and left the downloaded `datasets/` folder out.

### 2. Step 2: model

**Request:** Write `step2_model.py` with `ImageClassifier(nn.Module)`: Flatten,
Linear(784, 300), ReLU, Linear(300, 100), ReLU, Linear(100, 10). It returns raw
logits. `main` seeds with 42, builds the model on the device, and prints the
parameter count and the output shape for one batch.

**Generated:** `step2_model.py`, commit `8c209b6` ("Step 2: model"). The layer
sizes are constructor arguments whose defaults match the request.

**Output:** `Total parameters: 266610`. Input batch `[32, 1, 28, 28]` gives output
`[32, 10]`.

**Issues:** None.

### 3. Step 3: training

**Request:** Write `step3_train.py` with
`train(model, optimizer, loss_fn, metric, train_loader, valid_loader, n_epochs)`,
modeled on Géron's `train2`. Each epoch it records mean training loss, training
accuracy and validation accuracy using
`torchmetrics.Accuracy(task="multiclass", num_classes=10)`, prints them, and
returns a history dict. `main` trains with SGD lr=0.1 and CrossEntropyLoss for 20
epochs, then saves `history.json` and `model.pt`. Add a `.gitignore` so
`datasets/` and `model.pt` are not committed.

**Generated:** `step3_train.py` (which also has an `evaluate_tm()` helper),
`.gitignore` (`datasets/`, `model.pt`, `__pycache__/`), and `history.json`, all in
commit `1145bef` ("Step 3: train"). `model.pt` was written to disk but not
committed.

**Issues:** `torchmetrics` was not installed (`ModuleNotFoundError: No module
named 'torchmetrics'`). I installed it with `python -m pip install torchmetrics`,
which gave version 1.9.0, and confirmed the torch install was unchanged at
2.14.0+cpu.

**Output (selected epochs):**

| Epoch | Train loss | Train acc | Valid acc |
|---|---|---|---|
| 1 | 0.6060 | 0.7816 | 0.8418 |
| 5 | 0.3149 | 0.8840 | 0.8768 |
| 10 | 0.2518 | 0.9045 | 0.8836 |
| 17 | 0.2032 | 0.9227 | 0.8906 |
| 20 | 0.1881 | 0.9287 | 0.8766 |

Training loss went down every epoch. Validation accuracy stopped improving after
about epoch 11 and moved between 0.868 and 0.891, so the model began to overfit.

### 4. Step 4: accuracy plot

**Request:** Write `step4_plot_accuracy.py`. It loads `history.json`, plots
training and validation accuracy per epoch on the same axes with a title, axis
labels, legend and grid, saves `training_accuracy.png`, and calls `plt.show()`.

**Generated:** `step4_plot_accuracy.py` and `training_accuracy.png`, commit
`b21c38d` ("Step 4: plot accuracy").

**Issues:** A normal `plt.show()` opens a window and waits until it is closed,
which would have stopped the automated session. I ran the script with
`MPLBACKEND=Agg`, a non-interactive backend, instead. The PNG was saved normally,
and matplotlib printed an expected warning:
`UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown`. The
script itself was not changed, so running it normally opens the plot window. I
opened the saved PNG to check that the labels and legend don't overlap the lines.

### 5. Step 5: evaluation

**Request:** Write `step5_evaluate.py`. It rebuilds `ImageClassifier`, loads
`model.pt` and reports test accuracy. Then, as in Géron, it takes the first 3
validation images and prints the predicted indices and names, the true labels,
the softmax probabilities rounded to 3 decimals, and the top-4 classes.

**Generated:** `step5_evaluate.py` (with a `load_model()` helper), commit
`50f90de` ("Step 5: evaluate").

**Output:** Test accuracy was 0.8755. All 3 validation images were classified
correctly:

| Image | Predicted = true | Top-4 |
|---|---|---|
| 0 | 7 Sneaker | Sneaker 0.920, Ankle boot 0.080, Sandal 0.000, Bag 0.000 |
| 1 | 4 Coat | Coat 0.997, Pullover 0.003, Shirt 0.000, Bag 0.000 |
| 2 | 2 Pullover | Pullover 0.737, Shirt 0.165, Coat 0.096, T-shirt/top 0.001 |

**Issues:** None.

### 6. README, requirements and push

**Request:** Add `requirements.txt` and a `README.md` listing the scripts in the
order to run them, commit, push to `origin main`, and report the Python
interpreter path.

**Generated:** `requirements.txt` (versions pinned to the ones used) and
`README.md`, commit `757b23a` ("Add README and requirements"). The push created
`main` on GitHub. Interpreter: `C:\Anaconda\python.exe`.

**Issues:** Four files were already in the folder before this work started and
are not part of this example: `HW03_PLAYBOOK.md`, `build_notebook.py`,
`prompts_P1.txt` and `p1_claude/`. I did not commit them and asked about them.

### 7. Excluding the extra files and writing this summary

**Request:** Keep those four files out of the repo and untracked, then write this
summary, commit and push it.

**Generated:** The four paths were added to `.git/info/exclude`, git's local
ignore file. It is not committed, so their names don't appear in the repo and
they stop showing up in `git status`. This file, `HW03_Prob1_dialog_summary.md`,
is in the final commit. A commit can't contain its own hash, so run `git log` to
see it.

## Notes that apply to every step

- Every commit printed `LF will be replaced by CRLF` warnings. These come from
  Git's Windows line-ending setting and don't affect the code.
- The random seed is 42 for the train/validation split and again for the model's
  starting weights, so runs can be reproduced on the same setup.
- `model.pt` holds the epoch-20 weights, not the best-validation weights from
  epoch 17.
