#V3:kvantizacija tezina, biasa i aktivacija
import sys
from pathlib import Path

import onnx
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
sys.path.insert(0, str(THIS_DIR))

from calibration import MNISTCalibrationDataReader

INPUT_PATH = ROOT / "02_original_model" / "lenet5_prep.onnx"
OUT_DIR = ROOT / "04_quantized_models"
NUM_CALIBRATION_SAMPLES = 200

QUANT_TYPES = {
    8: (QuantType.QInt8, QuantType.QUInt8),
    16: (QuantType.QInt16, QuantType.QUInt16),
}


def main():
    input_name = onnx.load(str(INPUT_PATH)).graph.input[0].name

    for bits in (8, 16):
        weight_type, activation_type = QUANT_TYPES[bits]
        calibration_reader = MNISTCalibrationDataReader(input_name, num_samples=NUM_CALIBRATION_SAMPLES)
        #preko ONNX fje quantize static i kalibracionog skupa
        out_path = OUT_DIR / f"lenet5_full_int{bits}.onnx"
        quantize_static(
            model_input=str(INPUT_PATH),
            model_output=str(out_path),
            calibration_data_reader=calibration_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=True, 
            weight_type=weight_type,
            activation_type=activation_type,
            calibrate_method=CalibrationMethod.MinMax, #kvantizuje se sa min i max vrednoscu iz kalibracionog skupa
        )

        onnx.checker.check_model(onnx.load(str(out_path)), full_check=True)
        print(f"sacuvano: {out_path.name}  velicina={out_path.stat().st_size} B")


if __name__ == "__main__":
    main()
