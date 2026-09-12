#provera brzine zakljucivanja, sa 50 zagrevajucih prolaza i 1000 na kojima se meri
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

WARMUP = 50
REPEATS = 1000
BATCH_SIZE = 1

ROOT = Path(__file__).resolve().parents[1]
FP32_PATH = ROOT / "02_original_model" / "lenet5_prep.onnx"
QUANT_DIR = ROOT / "04_quantized_models"


def bench(onnx_path, save_optimized: bool = False, batch_size: int = BATCH_SIZE) -> dict:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if save_optimized:
        onnx_path = Path(onnx_path)
        so.optimized_model_filepath = str(onnx_path.with_name(onnx_path.stem + "_optimized.onnx"))

    session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    rng = np.random.default_rng(0)
    x = rng.standard_normal((batch_size, 1, 28, 28)).astype(np.float32)

    for _ in range(WARMUP):
        session.run(None, {input_name: x})

    times = np.empty(REPEATS)
    for i in range(REPEATS):
        start = time.perf_counter()
        session.run(None, {input_name: x})
        times[i] = time.perf_counter() - start

    times_ms = times * 1000
    return {
        "min": times_ms.min(),
        "median": np.median(times_ms),
        "mean": times_ms.mean(),
        "std": times_ms.std(),
    }


def main():
    args = sys.argv[1:]
    save_optimized = "--save-optimized" in args
    if save_optimized:
        args.remove("--save-optimized")

    batch_size = BATCH_SIZE
    if "--batch" in args:
        idx = args.index("--batch")
        batch_size = int(args[idx + 1])
        del args[idx : idx + 2]

    paths = [Path(p) for p in args] if args else [FP32_PATH] + sorted(QUANT_DIR.glob("*.onnx"))

    for path in paths:
        stats = bench(path, save_optimized=save_optimized, batch_size=batch_size)
        print(
            f"{path.name:<28} min={stats['min']:.4f}  median={stats['median']:.4f}  "
            f"mean={stats['mean']:.4f} +- {stats['std']:.4f} ms/inference "
            f"(batch={batch_size}, warmup={WARMUP}, repeats={REPEATS})"
        )


if __name__ == "__main__":
    main()
