import tempfile
from collections import Counter
from pathlib import Path

import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
QUANT_DIR = ROOT / "04_quantized_models"

MODELS = [
    ("FP32 (baseline)", ROOT / "02_original_model" / "lenet5_prep.onnx"),
    ("V1 samo tezine - INT8", QUANT_DIR / "lenet5_weights_int8.onnx"),
    ("V1 samo tezine - INT16", QUANT_DIR / "lenet5_weights_int16.onnx"),
    ("V2 tezine+bias - INT8", QUANT_DIR / "lenet5_weights_bias_int8.onnx"),
    ("V2 tezine+bias - INT16", QUANT_DIR / "lenet5_weights_bias_int16.onnx"),
    ("V3 tezine+bias+akt. - INT8", QUANT_DIR / "lenet5_full_int8.onnx"),
    ("V3 tezine+bias+akt. - INT16", QUANT_DIR / "lenet5_full_int16.onnx"),
]

INTEGER_KERNEL_OPS = {
    "QLinearConv", "QGemm", "QLinearMatMul", "QLinearAdd",
    "ConvInteger", "MatMulInteger", "QLinearAveragePool", "QLinearGlobalAveragePool",
}


def optimized_op_counts(path: Path) -> Counter:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    so.optimized_model_filepath = str(tmp_path)
    ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])

    model = onnx.load(str(tmp_path))
    tmp_path.unlink(missing_ok=True)
    return Counter(n.op_type for n in model.graph.node)


def main():
    for label, path in MODELS:
        counts = optimized_op_counts(path)
        used_integer_kernels = sorted(op for op in counts if op in INTEGER_KERNEL_OPS)

        print(f"\n=== {label} ===")
        for op, n in counts.most_common():
            marker = "  <- celobrojni kernel" if op in INTEGER_KERNEL_OPS else ""
            print(f"  {op:<22}{n}{marker}")

        if used_integer_kernels:
            print(f"  => FUZIONISANO u celobrojne kernele: {', '.join(used_integer_kernels)}")
        else:
            print("  => NEMA fuzije u celobrojne kernele (racun ostaje u FP32)")


if __name__ == "__main__":
    main()
