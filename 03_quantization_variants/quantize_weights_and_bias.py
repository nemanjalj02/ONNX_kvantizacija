#V2: kvantizacija tezina i biasa
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
sys.path.insert(0, str(THIS_DIR))

from common_tools import quantize_per_channel, quantize_per_tensor, replace_with_dequantized

INPUT_PATH = ROOT / "02_original_model" / "lenet5_prep.onnx"
OUT_DIR = ROOT / "04_quantized_models"

WEIGHT_NAMES = [
    "conv1.weight",
    "conv2.weight",
    "fc1.weight",
    "fc2.weight",
    "fc3.weight",
]

BIAS_NAMES = [
    "conv1.bias",
    "conv2.bias",
    "fc1.bias",
    "fc2.bias",
    "fc3.bias",
]

QUANT_TYPES = {
    8: (TensorProto.INT8, np.int8, 127),
    16: (TensorProto.INT16, np.int16, 32767),
}

#tezine kao i u V1, a za biase se poziva fja za kvantizaciju po tenzoru
def build_quantized_model(bits: int) -> onnx.ModelProto:
    _, np_dtype, qmax = QUANT_TYPES[bits]

    model = onnx.load(str(INPUT_PATH))
    graph = model.graph

    for weight_name in WEIGHT_NAMES:
        init = next(i for i in graph.initializer if i.name == weight_name)
        weight = numpy_helper.to_array(init)
        q, scale = quantize_per_channel(weight, qmax)
        replace_with_dequantized(graph, weight_name, q, scale, np_dtype, axis=0)

    for bias_name in BIAS_NAMES:
        init = next(i for i in graph.initializer if i.name == bias_name)
        bias = numpy_helper.to_array(init)
        q, scale = quantize_per_tensor(bias, qmax)
        replace_with_dequantized(graph, bias_name, q, scale, np_dtype, axis=None)

    onnx.checker.check_model(model, full_check=True)
    return model


def main():
    for bits in (8, 16):
        model = build_quantized_model(bits)
        out_path = OUT_DIR / f"lenet5_weights_bias_int{bits}.onnx"
        onnx.save(model, str(out_path))
        print(f"sacuvano: {out_path.name}  velicina={out_path.stat().st_size} B")


if __name__ == "__main__":
    main()
