#propusta ranije provereni model kroz fju za preprocess
from pathlib import Path

import onnx
from onnxruntime.quantization.shape_inference import quant_pre_process

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "02_original_model" / "lenet5.onnx"
OUTPUT_PATH = ROOT / "02_original_model" / "lenet5_prep.onnx"


def main():
    model_before = onnx.load(str(INPUT_PATH))
    print(f"ulaz:  {INPUT_PATH.name}  cvorova={len(model_before.graph.node)}  "
          f"velicina={INPUT_PATH.stat().st_size} B")
    quant_pre_process(
        input_model=str(INPUT_PATH),
        output_model_path=str(OUTPUT_PATH),
        skip_symbolic_shape=True,
    )

    model_after = onnx.load(str(OUTPUT_PATH))
    print(f"izlaz: {OUTPUT_PATH.name}  cvorova={len(model_after.graph.node)}  "
          f"velicina={OUTPUT_PATH.stat().st_size} B")

    onnx.checker.check_model(model_after, full_check=True)
    print("OK - pripremljeni model je validan (onnx.checker, full_check).")


if __name__ == "__main__":
    main()
