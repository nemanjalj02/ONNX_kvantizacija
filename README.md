# ONNX Kvantizacija

Kvantizacija LeNet5 CNN modela (MNIST) u ONNX formatu, tri varijante (samo tezine / tezine+bias / tezine+bias+aktivacije), svaka u INT8 i INT16, sa poredjenjem tacnosti, brzine i velicine u odnosu na FP32 original.

## Struktura foldera

```
ONNX_Kvantizacija/
│
├── 01_training_preparation/
│   ├── CNN_training.ipynb
│   ├── preprocess.py
│   └── verify_export.py
│
├── 02_original_model/
│   ├── lenet5.onnx
│   ├── lenet5.onnx.data
│   ├── lenet5_best.pth
│   └── lenet5_prep.onnx
│
├── 03_quantization_variants/
│   ├── calibration.py
│   ├── common_tools.py
│   ├── quantize_static_variant.py
│   ├── quantize_weights_and_bias.py
│   └── quantize_weights_only.py
│
├── 04_quantized_models/
│   ├── lenet5_full_int16.onnx
│   ├── lenet5_full_int16_optimized.onnx
│   ├── lenet5_full_int8.onnx
│   ├── lenet5_full_int8_optimized.onnx
│   ├── lenet5_weights_bias_int16.onnx
│   ├── lenet5_weights_bias_int8.onnx
│   ├── lenet5_weights_int16.onnx
│   └── lenet5_weights_int8.onnx
│
├── 05_compare_models/
│   ├── 01_accuracy.py
│   ├── 02_size_breakdown.py
│   ├── 03_inference.py
│   ├── check_fusion.py
│   └── diff_predictions.py
│
├── 06_plots/
│   ├── plot_training.py
│   └── training_curve.png
│
├── 07_documentation/
│   ├── Izvestaj.pdf
│   ├── Uputstvo za pokretanje.pdf
│   └── Uvodna prezentacija.pdf
│
├── requirements.txt
└── README.md
```

## Opis foldera

### 01_training_preparation/
- `CNN_training.ipynb` trening LeNet5 na MNIST-u
- `preprocess.py` ONNX shape inference, priprema modela za kvantizaciju
- `verify_export.py` provera ispavnosti strukture, izlaza i verzije

### 02_original_model/
- `lenet5_best.pth` PyTorch tezine posle treninga
- `lenet5.onnx`, `lenet5.onnx.data` izvezen ONNX model
- `lenet5_prep.onnx` model posle shape inference-a, ulaz za sve kvantizacije

### 03_quantization_variants/
- `quantize_weights_only.py` varijanta 1, kvantizacija samo tezina
- `quantize_weights_and_bias.py` varijanta 2, kvantizacija tezina i bias-a
- `quantize_static_variant.py` varijanta 3, staticka kvantizacija tezina, bias-a i aktivacija
- `calibration.py` kalibracioni data reader za varijantu 3
- `common_tools.py` deljene funkcije za varijante 1 i 2

### 04_quantized_models/
- `lenet5_weights_int8.onnx`, `lenet5_weights_int16.onnx` varijanta 1
- `lenet5_weights_bias_int8.onnx`, `lenet5_weights_bias_int16.onnx` varijanta 2
- `lenet5_full_int8.onnx`, `lenet5_full_int16.onnx` varijanta 3
- `*_optimized.onnx` verzije posle ONNX Runtime graph optimizacije

### 05_compare_models/
- `01_accuracy.py` top-1 tacnost, bez argumenata proverava FP32 + sve kvantizovane modele
- `02_size_breakdown.py` velicina modela po tenzorima
- `03_inference.py` brzina inferencije, bez argumenata meri FP32 + sve kvantizovane modele
- `check_fusion.py` provera da li se QDQ graf fuzionise u celobrojne kernele
- `diff_predictions.py` razlike u predikcijama po uzorku

### 06_plots/
- `plot_training.py` iscrtava krivu treninga
- `training_curve.png` generisan grafik

### 07_documentation/
- `Izvestaj.pdf` pisani izvestaj o projektu
- `Uputstvo za pokretanje.pdf` uputstvo za pokretanje koda korak po korak
- `Uvodna prezentacija.pdf` uvodna prezentacija projekta


