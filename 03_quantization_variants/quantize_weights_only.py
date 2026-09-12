#V1: kvantizuju se samo tezine
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
sys.path.insert(0, str(THIS_DIR))

from common_tools import quantize_per_channel, replace_with_dequantized

INPUT_PATH = ROOT / "02_original_model" / "lenet5_prep.onnx"
OUT_DIR = ROOT / "04_quantized_models"
#nazivi incijalizatora  koji se kvantizuju
WEIGHT_NAMES = [
    "conv1.weight",
    "conv2.weight",
    "fc1.weight",
    "fc2.weight",
    "fc3.weight",
]
#tipovi kvantizacije sa maks vrednostima
QUANT_TYPES = {
    8: (TensorProto.INT8, np.int8, 127),
    16: (TensorProto.INT16, np.int16, 32767),
}

#za svaki inicijalizator se poveziva fja za kvantizaciju po kanalu i umece se taj cvor u graf
def build_quantized_model(bits: int) -> onnx.ModelProto:
    _, np_dtype, qmax = QUANT_TYPES[bits]

    model = onnx.load(str(INPUT_PATH))
    graph = model.graph

    for weight_name in WEIGHT_NAMES:
        init = next(i for i in graph.initializer if i.name == weight_name)
        weight = numpy_helper.to_array(init)
        q, scale = quantize_per_channel(weight, qmax)
        replace_with_dequantized(graph, weight_name, q, scale, np_dtype, axis=0)

    onnx.checker.check_model(model, full_check=True)
    return model


def main():
    for bits in (8, 16):
        model = build_quantized_model(bits)
        out_path = OUT_DIR / f"lenet5_weights_int{bits}.onnx"
        onnx.save(model, str(out_path))
        print(f"sacuvano: {out_path.name}  velicina={out_path.stat().st_size} B")


if __name__ == "__main__":
    main()
