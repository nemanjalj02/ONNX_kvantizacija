#odvajanje kalibracionog skupa koji se koristi u varijanti kvantizacije aktivacija
from pathlib import Path

import torch
from onnxruntime.quantization import CalibrationDataReader
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


class MNISTCalibrationDataReader(CalibrationDataReader):
    #priprema skupa podataka za kalibraciju
    #odvajanje 200 uzoraka, ista normalizacija kao i u pocetnom notebook-u
    def __init__(self, input_name: str, num_samples: int = 200, batch_size: int = 1):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_set = datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=transform)

        generator = torch.Generator().manual_seed(0)
        loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator)

        self.input_name = input_name
        self.iterator = iter(loader)
        self.num_batches = num_samples // batch_size

    def get_next(self):
        if self.num_batches <= 0:
            return None
        try:
            images, _ = next(self.iterator)
        except StopIteration:
            return None
        self.num_batches -= 1
        return {self.input_name: images.numpy()}
