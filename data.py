import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import yaml

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MNISTData:
    def __init__(self):
        # Load configuration
        with open("config.yml", "r") as f:
            self.config = yaml.safe_load(f)

        torch.manual_seed(self.config["manual_seed"])

        # Set hardware settings based on device
        self.kwargs = (
            {"num_workers": self.config["num_workers"], "pin_memory": self.config["pin_memory"]}
            if get_device() == "cuda"
            else {}
        )

        # Create transforms
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=self.config["rotation_degrees"]),
            transforms.ToTensor(),
            transforms.Normalize(
                (self.config["normalize_mean"],), (self.config["normalize_std"],)
            ),
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (self.config["normalize_mean"],), (self.config["normalize_std"],)
            ),
        ])

    def get_train_loader(self):
        train_dataset = datasets.MNIST(
            self.config["train_data_path"],
            train=True,
            download=True,
            transform=self.train_transform
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            **self.kwargs
        )

    def get_test_loader(self):
        test_dataset = datasets.MNIST(
            self.config["test_data_path"],
            train=False,
            transform=self.test_transform
        )
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            **self.kwargs
        )
