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
├── __init__.py              # PE facade
├── registry.py              # Model registry (separate to avoid circular imports)
├── models/
│   ├── base.py              # BasePE abstract class
│   ├── fcpe/                # MelConformerF0 (former CFNaiveMelPE)
│   └── rmvpe/               # RMVPEUNet (former E2E0)
├── postproc/                # Model-agnostic postprocessing
│   ├── decode.py            # argmax / weighted / viterbi / local_argmax
│   ├── ensemble.py          # DP TTA (from FCPE)
│   ├── threshold.py         # UV detection (from torchcrepe)
│   ├── filter.py            # NaN-aware smoothing
│   ├── periodicity.py       # entropy / max / sum
│   └── convert.py           # bin ↔ cent ↔ Hz ↔ MIDI
├── io/                      # .f0 binary format + CSV
├── remote.py                # Subprocess server client (Windows fallback)
├── server.py                # HTTP inference server
├── cli/main.py              # Single-command auto-detect
├── export/                  # ONNX + static graph
└── training/                # Training infrastructure
```

## Binary Format (.f0)

Magic: `PADDLEF0` (8 bytes), header: 37 bytes packed (cross-language).
Data: f0_hz (float32)[N] + confidence (float32)[N, optional].

## Remote Mode (Windows CUDA Workaround)

On Windows, PyTorch and PaddlePaddle cuDNN DLLs conflict.
Use `PE.create("fcpe", force_remote=True)` or let auto-fallback handle it.

## Key Design Decisions

- **No conversion scripts in repo** — all weights are pre-converted `.pdparams`
- **Model-agnostic postproc** — `postproc/` modules work with any F0 model
- **Registry pattern** — models self-register via `@PE.register("name")`
- **Paddle code style** — ruff 0.15.0, 80 char line length, preserve quotes
