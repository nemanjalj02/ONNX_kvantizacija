from pathlib import Path

import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]
QUANT_DIR = ROOT / "04_quantized_models"

MODELS = [
    ("Base", "FP32", ROOT / "02_original_model" / "lenet5_prep.onnx"),
    ("V1", "INT16", QUANT_DIR / "lenet5_weights_int16.onnx"),
    ("V2", "INT16", QUANT_DIR / "lenet5_weights_bias_int16.onnx"),
    ("V3", "INT16", QUANT_DIR / "lenet5_full_int16.onnx"),
    ("V1", "INT8", QUANT_DIR / "lenet5_weights_int8.onnx"),
    ("V2", "INT8", QUANT_DIR / "lenet5_weights_bias_int8.onnx"),
    ("V3", "INT8", QUANT_DIR / "lenet5_full_int8.onnx"),
]


def categorize(name: str) -> str:
    n = name.lower()
    if "zero_point" in n:
        return "zero_point"
    if "scale" in n:
        return "scale"
    if "bias" in n:
        return "bias"
    if "weight" in n:
        return "weight"
    return "ostalo (initializer)"


def tensor_bytes(init) -> int:
    arr = numpy_helper.to_array(init)
    return arr.nbytes


def main():
    header = (
        f"{'Model':<8}{'Bitska preciznost':<20}{'Ukupna velicina (KB)':>22}"
        f"{'Podaci za kvantizaciju (KB)':>30}{'Struktura grafa (KB)':>22}"
    )
    print(header)
    for model_label, precision, path in MODELS:
        model = onnx.load(str(path))
        file_size = path.stat().st_size

        totals = {"weight": 0, "bias": 0, "scale": 0, "zero_point": 0, "ostalo (initializer)": 0}
        for init in model.graph.initializer:
            cat = categorize(init.name)
            totals[cat] += tensor_bytes(init)

        init_total = sum(totals.values())
        structure = file_size - init_total

        print(
            f"{model_label:<8}{precision:<20}{file_size / 1024:>22.2f}"
            f"{init_total / 1024:>30.2f}{structure / 1024:>22.2f}"
        )


if __name__ == "__main__":
    main()
