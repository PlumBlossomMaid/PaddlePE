# paddlePE — Project Context

Unified Pitch Extraction (F0) toolkit for PaddlePaddle.

4 deep models (FCPE, RMVPE, CREPE, PENN) + 2 wrappers (Parselmouth, WORLD).

## Quick Reference

```bash
# Install
pip install -e .

# List models
paddlepe -l

# Extract pitch
paddlepe input.wav -o output.f0         # auto-detect model
paddlepe input.wav -o output.csv -f csv # CSV format

# Format convert
paddlepe in.f0 -o out.csv

# Start inference server (Windows CUDA workaround)
paddlepe server --model fcpe --port 18560

# Run CI tests
python -m pytest tests/test_io.py tests/test_base.py tests/test_postproc.py -v --timeout=120

# Lint (pre-commit before every commit!)
ruff check paddlepe/ tests/
ruff format paddlepe/ tests/ --check
pre-commit run --all-files

# Train
python scripts/train.py                          # default: rmvpe, mir1k+ptdb, 1 epoch
python scripts/train.py --config configs/train_default.yaml
python scripts/train.py --model crepe --datasets mir1k --training.epochs 5
```

## Architecture

```
paddlepe/
├── __init__.py              # PE facade + try-import paddle (auto fallback to remote)
├── pe.py                    # PE.create() + _auto_device
├── registry.py              # Model registry (@PE.register decorator)
├── models/
│   ├── base.py              # BasePE abstract class (forward + infer)
│   ├── fcpe/                # MelConformerF0 (Conformer-based, 42M)
│   ├── rmvpe/               # RMVPEUNet (UNet + BiGRU, 355M)
│   ├── crepe/               # CrepeBackbone (6-layer ConvNet, 167M)
│   ├── penn/                # PennBackbone (6 Conv1D + LayerNorm, 35M)
│   └── wrappers/            # Parselmouth, pyworld (non-DL)
├── postproc/                # Model-agnostic postprocessing (numpy + paddle)
│   ├── pipeline.py          # Unified pipeline (postprocess_f0)
│   ├── filter.py            # median_filter, interpolate_uv, nanmean...
│   ├── threshold.py         # threshold_at, hysteresis, silence_mask
│   ├── decode.py            # argmax, weighted_argmax, viterbi
│   ├── convert.py           # Hz↔cent↔MIDI↔bin conversion
│   ├── ensemble.py          # Multi-key-shift ensemble
│   └── periodicity.py       # Entropy/max/sum confidence estimation
├── io/                      # .f0 binary format + CSV + .pv/.tsv readers
│   ├── formats.py           # Binary header (PADDLEF0, 37 bytes)
│   ├── reader.py            # read_f0, read_csv, read_pv, read_tsv
│   └── writer.py            # write_f0, write_csv
├── remote.py                # Subprocess server client (Windows fallback)
├── server.py                # HTTP inference server
├── cli/main.py              # Single-command CLI
├── export/                  # ONNX + static graph export
└── training/                # Training infrastructure
    ├── hdf5_dataset.py      # HDF5Dataset (unified format reader)
    ├── pe_datamodule.py     # PEDataModule (N-dataset, auto-preprocess)
    ├── collators/           # Per-model collators (FCPE/RMVPE/CREPE/PENN)
    ├── samplers.py          # Multi-dataset sampling (proportional/balanced/round_robin)
    └── preprocess/          # Dataset-specific preprocessing → HDF5
```

## Registered Models

| Name | Type | Backbone | Weights |
|------|------|----------|---------|
| `crepe` | DL | 6-layer ConvNet | ckpts/crepe.pdparams (167 MB) |
| `fcpe` | DL | MelConformerF0 | ckpts/fcpe.pdparams (42 MB) |
| `penn` | DL | PennBackbone | ckpts/penn.pdparams (35 MB) |
| `rmvpe` | DL | RMVPEUNet | ckpts/rmvpe.pdparams (355 MB) |
| `parselmouth` | Wrapper | Praat autocorrelation | N/A |
| `world` | Wrapper | WORLD Harvest | N/A |

## Post-processing Pipeline

Each model's `infer()` calls `postprocess_f0()` with its own defaults.
Pipeline runs **entirely on GPU** (paddle tensors) — no CPU round-trip.

| Model | threshold | median_filter | interp_uv | decoder |
|-------|-----------|---------------|-----------|---------|
| FCPE | 0.05 | 0 | False | local_argmax |
| RMVPE | 0.03 | 0 | False | local_average |
| CREPE | 0.5 | **3** | **True** | weighted_argmax |
| PENN | 0.01 | 0 | False | argmax |

Users override any parameter: `pe.infer(wav, sr, interp_uv=True, median_filter=5)`

## Training

```yaml
# configs/train_default.yaml
model: rmvpe
datasets: [mir1k, ptdb]
training:
  epochs: 1
  batch_size: 2
  device: gpu:0
  val_split: 0.02
```

Training uses `ocean.Trainer` with `ocean.Model` subclass. Each model's
`_Model(ocean.Model)` defines `training_step` / `validation_step` in
`scripts/train.py` — this is the **correct design pattern** for ocean
(analogous to PyTorch Lightning, NOT like ppmat's BaseTrainer).

## Supported Datasets

| Dataset | Samples | Sample Rate | Auto-download | Source |
|---------|---------|-------------|---------------|--------|
| MIR-1K | 990 clips | 16 kHz | ✅ AI Studio | Right channel = vocals, .pv → Hz |
| PTDB-TUG | 4718 clips | 48→16 kHz | ✅ Direct URL | 4-column F0, EGG-derived |

Preprocessed to unified HDF5 format: waveform + f0 + sr + hop + name.
Frame rate: 10 ms hop (100 fps).

## Precision Alignment

- CREPE: Paddle vs PyTorch reference, max diff < 3e-5 (fp64)
- RMVPE: Weights converted and verified
- FCPE: einops 0.8.2+ required (supports Paddle natively)
- PENN: 220 Hz → 220.3 Hz, argmax matches torch exactly

## Binary Format (.f0)

- Magic: `PADDLEF0` (8 bytes)
- Header: 37 bytes packed (cross-language C struct)
- Data: f0_hz (float32)[N] + optional confidence (float32)[N]

## Remote Mode (Windows CUDA Workaround)

On Windows, PyTorch and PaddlePaddle cuDNN DLLs conflict when importing
in the same process. `import paddlepe` auto-detects and falls back to
subprocess server mode. Use `PE.create("fcpe", force_remote=True)`
to force remote mode.

## Key Design Decisions

- **Try-import split** — `__init__.py` tries `import paddle`; if DLL fails, client-only mode
- **Pre-converted weights** — `.pdparams` in `ckpts/`, no runtime conversion needed
- **Model-agnostic postproc** — `postproc/` modules work with any F0 model
- **Registry pattern** — models self-register via `@PE.register("name")`
- **Collator pattern** — train-time: HDF5Dataset (uniform) → Collator (model-specific) → DataModule
- **ocean.Trainer** — training uses paddleOcean's Trainer with Model hooks
- **Paddle code style** — ruff 0.15.0, 88 char line length, preserve quotes
- **pre-commit** — must run `pre-commit run --all-files` before every commit
- **CI** — lint + tests-cpu (3 OS × 3 Python); tests-cuda removed (self-hosted, not general)
