
from pathlib import Path

import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
QUANT_DIR = ROOT / "04_quantized_models"

FP32_PATH = ROOT / "02_original_model" / "lenet5_prep.onnx"
QUANT_MODELS = [
    ("V1 samo tezine - INT8", QUANT_DIR / "lenet5_weights_int8.onnx"),
    ("V1 samo tezine - INT16", QUANT_DIR / "lenet5_weights_int16.onnx"),
    ("V2 tezine+bias - INT8", QUANT_DIR / "lenet5_weights_bias_int8.onnx"),
    ("V2 tezine+bias - INT16", QUANT_DIR / "lenet5_weights_bias_int16.onnx"),
    ("V3 tezine+bias+akt. - INT8", QUANT_DIR / "lenet5_full_int8.onnx"),
    ("V3 tezine+bias+akt. - INT16", QUANT_DIR / "lenet5_full_int16.onnx"),
]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
test_set = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=transform)
loader = DataLoader(test_set, batch_size=256, shuffle=False)

all_images, all_labels = [], []
for images, labels in loader:
    all_images.append(images.numpy())
    all_labels.append(labels.numpy())
labels_flat = np.concatenate(all_labels)


def run_model(path):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    preds = []
    for images in all_images:
        out = session.run(None, {input_name: images})[0]
        preds.append(out.argmax(axis=1))
    return np.concatenate(preds)


def main():
    fp32_preds = run_model(FP32_PATH)

    for label, path in QUANT_MODELS:
        preds = run_model(path)
        diff_idx = np.where(preds != fp32_preds)[0]
        print(f"\n=== {label} ({len(diff_idx)} razlicitih predikcija od FP32) ===")
        print(f"{'idx':>6}{'tacna cifra':>13}{'FP32 predikcija':>17}{'INT8 predikcija':>17}{'FP32 tacno?':>13}{'INT8 tacno?':>13}")
        for i in diff_idx:
            true = labels_flat[i]
            fp32_p = fp32_preds[i]
            int8_p = preds[i]
            print(f"{i:>6}{true:>13}{fp32_p:>17}{int8_p:>17}{str(fp32_p==true):>13}{str(int8_p==true):>13}")


if __name__ == "__main__":
    main()
