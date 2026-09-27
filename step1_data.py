"""Step 1: Load Fashion MNIST and build the data loaders."""
import torch
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, random_split


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_datasets():
    toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    train_and_valid_data = torchvision.datasets.FashionMNIST(
        root="datasets", train=True, download=True, transform=toTensor)
    test_data = torchvision.datasets.FashionMNIST(
        root="datasets", train=False, download=True, transform=toTensor)
    torch.manual_seed(42)
    train_data, valid_data = random_split(train_and_valid_data, [55_000, 5_000])
    return train_data, valid_data, test_data


def get_loaders(batch_size=32):
    train_data, valid_data, test_data = get_datasets()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    class_names = test_data.classes
    return train_loader, valid_loader, test_loader, class_names


if __name__ == "__main__":
    train_data, valid_data, test_data = get_datasets()
    class_names = test_data.classes
    X_sample, y_sample = train_data[0]
    print("Shape:", X_sample.shape)
    print("Dtype:", X_sample.dtype)
    print("Class:", class_names[y_sample])
    print("Device:", get_device())
