#provera top 1 tacnosti za sve modele
import sys
from pathlib import Path

import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FP32_PATH = ROOT / "02_original_model" / "lenet5_prep.onnx"
QUANT_DIR = ROOT / "04_quantized_models"


def load_test_set():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_set = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=transform)
    return DataLoader(test_set, batch_size=256, shuffle=False)


def evaluate(onnx_path, loader) -> float:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    correct = 0
    total = 0
    for images, labels in loader:
        out = session.run(None, {input_name: images.numpy()})[0]
        preds = out.argmax(axis=1)
        correct += (preds == labels.numpy()).sum()
        total += labels.size(0)

    return correct / total


def main():
    if len(sys.argv) >= 2:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        paths = [FP32_PATH] + sorted(QUANT_DIR.glob("*.onnx"))

    loader = load_test_set()

    for path in paths:
        acc = evaluate(path, loader)
        size = path.stat().st_size
        print(f"{path.name:<28} top1_acc={acc:.4f}  velicina={size:>8} B")


if __name__ == "__main__":
    main()
