# paddlePE — Project Context

Unified Pitch Extraction (F0) toolkit for PaddlePaddle.

## Quick Reference

```bash
# Install
pip install -e .

# List models
paddlepe -l

# Extract pitch
paddlepe input.wav -o output.f0

# Format convert
paddlepe in.csv -o out.f0

# Start inference server (Windows CUDA workaround)
paddlepe server --model fcpe --port 18560

# Run tests
python -m pytest tests/ -v --timeout=120

# Lint
ruff check paddlepe/ tests/
ruff format paddlepe/ tests/ --check
```

## Architecture

```
paddlepe/
├── __init__.py              # PE facade + try-import paddle (auto fallback)
├── registry.py              # Model registry (separate to avoid circular imports)
├── models/
│   ├── base.py              # BasePE abstract class
│   ├── fcpe/                # MelConformerF0 (Conformer-based)
│   ├── rmvpe/               # RMVPEUNet (UNet + BiGRU)
│   ├── crepe/               # CrepeBackbone (6-layer ConvNet)
│   └── wrappers/            # Parselmouth, pyworld (non-DL)
├── postproc/                # Model-agnostic postprocessing
├── io/                      # .f0 binary format + CSV + .pv/.tsv
├── remote.py                # Subprocess server client (Windows fallback)
├── server.py                # HTTP inference server
├── cli/main.py              # Single-command auto-detect
├── export/                  # ONNX + static graph
└── training/                # Training infrastructure
```

## Registered Models

| Name | Type | Backbone | Weights |
|------|------|----------|---------|
| `crepe` | DL | 6-layer ConvNet | ckpts/crepe.pdparams (167M) |
| `fcpe` | DL | MelConformerF0 | ckpts/fcpe.pdparams (42M) |
| `rmvpe` | DL | RMVPEUNet | ckpts/rmvpe.pdparams (355M) |
| `parselmouth` | Wrapper | Praat autocorrelation | N/A |
| `world` | Wrapper | WORLD Harvest/DIO | N/A |

## Precision Alignment

CREPE: Paddle output vs PyTorch reference: max diff < 3e-5 (fp64).
RMVPE: Weights converted and verified.
FCPE: Weights partially mapped (needs exact key alignment).

## Binary Format (.f0)

Magic: `PADDLEF0` (8 bytes), header: 37 bytes packed (cross-language).
Data: f0_hz (float32)[N] + confidence (float32)[N, optional].

## Remote Mode (Windows CUDA Workaround)

On Windows, PyTorch and PaddlePaddle cuDNN DLLs conflict.
`import paddlepe` auto-detects and falls back to subprocess server.
Use `PE.create("fcpe", force_remote=True)` to force remote mode.

## Key Design Decisions

- **Try-import split** — `__init__.py` tries `import paddle`; if DLL fails, client-only mode
- **Pre-converted weights** — `.pdparams` directly in `ckpts/`, no runtime conversion
- **Model-agnostic postproc** — `postproc/` modules work with any F0 model
- **Registry pattern** — models self-register via `@registry.register("name")`
- **Paddle code style** — ruff 0.15.0, 80 char line length, preserve quotes
