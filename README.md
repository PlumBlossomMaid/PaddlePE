# paddlePE: Unified Pitch Extraction Toolkit for PaddlePaddle

A unified pitch extraction (F0) toolkit for PaddlePaddle, collecting multiple
pitch estimation models under a single API.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from paddlepe import PE

# List available models
print(PE.list_models())

# Create a pitch extractor
pe = PE.create("fcpe")  # or "rmvpe"

# Extract pitch from audio
import paddle
wav = paddle.to_tensor(...)  # shape: (samples,) or (1, samples)
f0, confidence = pe.infer(wav, sr=16000)
```

## CLI

```bash
# Extract pitch from WAV
paddlepe input.wav -o out.f0
paddlepe input.wav -o out.csv -f csv

# Convert formats
paddlepe in.csv -o out.f0
paddlepe in.f0 -o out.csv

# List available models
paddlepe -l
```

## Supported Models

| Model | Type | Trainable | ONNX |
|-------|------|-----------|------|
| FCPE (MelConformerF0) | Conformer-based | ✅ | ✅ |
| RMVPE (RMVPEUNet) | UNet + BiGRU | ✅ | ✅ |
| Parselmouth | Praat wrapper | ❌ | ❌ |
| pyworld | WORLD vocoder | ❌ | ❌ |

## Binary Format (.f0)

Custom binary format with magic header `PADDLEF0`, cross-language readable (C/C++/Python/...).

## Project Structure

```
paddlePE/
├── paddlepe/           # Main package
│   ├── models/         # Model implementations
│   ├── postproc/       # Model-agnostic postprocessing
│   ├── io/             # Format I/O
│   ├── training/       # Training infrastructure
│   ├── export/         # ONNX/static graph export
│   └── cli/            # Command-line interface
├── ckpts/              # Pretrained weights
└── tests/              # Test suite
```
