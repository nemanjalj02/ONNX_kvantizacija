#skripta za proveru izvoza na 3 nivoa
#strukturno, preko onnx checker-a
#provera izlaza, preko dummy inputs. racunanje odstupanja
#provera kompatibilnosti opseta

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
ONNX_PATH = ROOT / "02_original_model" / "lenet5.onnx"
PTH_PATH = ROOT / "02_original_model" / "lenet5_best.pth"


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def check_structure(model: onnx.ModelProto) -> bool:
    print("\n1. Strukturna provera fajla (onnx.checker)")
    try:
        onnx.checker.check_model(model, full_check=True)
        print("Model je validan po ONNX specifikaciji")
        return True
    except onnx.checker.ValidationError as e:
        print("GRESKA ")
        print(e)
        return False

def check_outputs(model: onnx.ModelProto) -> bool:
    print("\n2. Provera izlaza (PyTorch referenca vs. ONNX Runtime)")

    torch_model = LeNet5()
    torch_model.load_state_dict(torch.load(PTH_PATH, map_location="cpu"))
    torch_model.eval()

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    torch.manual_seed(0)
    max_abs_diff = 0.0

    for batch_size in (1, 5, 16):
        x = torch.randn(batch_size, 1, 28, 28)

        with torch.no_grad():
            torch_out = torch_model(x).numpy()

        onnx_out = session.run(None, {input_name: x.numpy()})[0]

        diff = float(np.abs(torch_out - onnx_out).max())
        max_abs_diff = max(max_abs_diff, diff)
        print(f"batch_size={batch_size:>2}  max|diff|={diff:.3e}")

    print(f"najvece odstupanje (max|diff|) preko svih batch-eva: {max_abs_diff:.3e}")
    return True


def check_opset(model: onnx.ModelProto) -> bool:
    print("\n3. Provera okruzenja (opset)")
    for imp in model.opset_import:
        domain = imp.domain if imp.domain else "ai.onnx"
        print(f"model izvezen sa opset-om: domain='{domain}' version={imp.version}")

    print(f"onnx (biblioteka za citanje/validaciju) verzija: {onnx.__version__}")
    print(f"onnxruntime verzija: {ort.__version__}")

    try:
        ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
        print("OK - ONNX Runtime je uspesno ucitao model sa ovim opset-om.")
        return True
    except Exception as e:
        print("GRESKA - ONNX Runtime ne moze da ucita model (opset verovatno nije podrzan):")
        print(e)
        return False


def main():
    model = onnx.load(str(ONNX_PATH))

    check_structure(model)
    check_outputs(model)
    check_opset(model)


if __name__ == "__main__":
    main()
