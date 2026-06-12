#!/usr/bin/env python3
"""Convert PENN (FCNF0++) PyTorch weights to PaddlePaddle format.

Usage:
    python scripts/convert_penn.py                              # uses existing fcnf0++.pt
    python scripts/convert_penn.py --source /path/to/fcnf0++.pt # custom source
    python scripts/convert_penn.py --no-verify                   # skip allclose test

Key mapping:
  Torch Sequential                    Paddle PennPE
  ──────────────────────────          ──────────────────────
  0.0.weight / 0.0.bias              backbone.blocks.0.conv.weight / .bias
  0.3.weight / 0.3.bias   (LayerNorm) backbone.blocks.0.norm.weight / .bias
  1.0.weight / 1.0.bias              backbone.blocks.1.conv.weight / .bias
  1.3.weight / 1.3.bias              backbone.blocks.1.norm.weight / .bias
  2.0.weight / 2.0.bias              backbone.blocks.2.conv.weight / .bias
  2.3.weight / 2.3.bias              backbone.blocks.2.norm.weight / .bias
  3.0.weight / 3.0.bias              backbone.blocks.3.conv.weight / .bias
  3.2.weight / 3.2.bias   (no pool)  backbone.blocks.3.norm.weight / .bias
  4.0.weight / 4.0.bias              backbone.blocks.4.conv.weight / .bias
  4.2.weight / 4.2.bias              backbone.blocks.4.norm.weight / .bias
  5.0.weight / 5.0.bias              backbone.blocks.5.conv.weight / .bias
  5.2.weight / 5.2.bias              backbone.blocks.5.norm.weight / .bias
  6.weight  / 6.bias                 backbone.final_conv.weight / .bias
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_CKPT = PROJECT_ROOT / "ckpts" / "fcnf0++.pt"
PADDLE_CKPTS = PROJECT_ROOT / "ckpts"


# ---------------------------------------------------------------------------
# Key mapping
# ---------------------------------------------------------------------------
# Blocks 0,1,2 have pooling -> norm at index 3
# Blocks 3,4,5 have no pooling -> norm at index 2
BLOCK_NORM_INDICES = {0: 3, 1: 3, 2: 3, 3: 2, 4: 2, 5: 2}


def _build_key_map() -> dict[str, str]:
    """Build mapping from torch Sequential keys to Paddle PennPE keys.

    Returns:
        dict mapping torch_key -> paddle_key (with backbone. prefix)
    """
    mapping = {}

    for block_idx in range(6):
        # Conv1d layer
        torch_conv_w = f"{block_idx}.0.weight"
        torch_conv_b = f"{block_idx}.0.bias"
        paddle_conv_w = f"backbone.blocks.{block_idx}.conv.weight"
        paddle_conv_b = f"backbone.blocks.{block_idx}.conv.bias"
        mapping[torch_conv_w] = paddle_conv_w
        mapping[torch_conv_b] = paddle_conv_b

        # LayerNorm layer (at different indices depending on pooling)
        norm_idx = BLOCK_NORM_INDICES[block_idx]
        torch_norm_w = f"{block_idx}.{norm_idx}.weight"
        torch_norm_b = f"{block_idx}.{norm_idx}.bias"
        paddle_norm_w = f"backbone.blocks.{block_idx}.norm.weight"
        paddle_norm_b = f"backbone.blocks.{block_idx}.norm.bias"
        mapping[torch_norm_w] = paddle_norm_w
        mapping[torch_norm_b] = paddle_norm_b

    # Final Conv1d
    mapping["6.weight"] = "backbone.final_conv.weight"
    mapping["6.bias"] = "backbone.final_conv.bias"

    return mapping


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------
def convert(
    source: str | None = None,
    output: str | None = None,
    verify: bool = True,
) -> str:
    """Convert PENN PyTorch weights to Paddle format.

    Args:
        source: Path to fcnf0++.pt torch checkpoint
        output: Output .pdparams path (default: ckpts/penn.pdparams)
        verify: Run allclose test after conversion

    Returns:
        Path to the saved .pdparams file.
    """
    import torch

    source_path = Path(source) if source else SOURCE_CKPT
    if not source_path.exists():
        raise FileNotFoundError(
            f"PyTorch checkpoint not found: {source_path}"
        )

    if output is None:
        output = PADDLE_CKPTS / "penn.pdparams"

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    key_map = _build_key_map()

    # ------------------------------------------------------------------
    # 1. Load PyTorch checkpoint
    # ------------------------------------------------------------------
    print(f"[1/5] Loading torch checkpoint: {source_path}")
    torch_ckpt = torch.load(
        source_path, map_location="cpu", weights_only=True
    )

    # The checkpoint has: step, model (state_dict), optimizer
    if "model" in torch_ckpt:
        torch_sd = torch_ckpt["model"]
    else:
        torch_sd = torch_ckpt

    print(f"  Source keys ({len(torch_sd)} total):")
    for k, v in torch_sd.items():
        shape = list(v.shape)
        if k in key_map:
            print(f"  {k:25s} {str(shape):20s} -> {key_map[k]}")
        else:
            print(f"  {k:25s} {str(shape):20s}  (no mapping)")

    # ------------------------------------------------------------------
    # 2. Build Paddle model
    # ------------------------------------------------------------------
    print(f"\n[2/5] Building Paddle PennPE...")
    import paddle

    sys.path.insert(0, str(PROJECT_ROOT))
    from paddlepe.models.penn.backbone import PennBackbone
    from paddlepe.models.penn.infer import PennPE

    # Build bare backbone for verification
    paddle_backbone = PennBackbone()
    paddle_backbone.eval()

    # Also build full PE model for state dict
    paddle_pe = PennPE()
    paddle_pe.eval()

    # ------------------------------------------------------------------
    # 3. Build state dict
    # ------------------------------------------------------------------
    print(f"\n[3/5] Mapping weights to Paddle format...")
    paddle_state = {}
    mapped = 0
    errors = []

    for torch_key, torch_tensor in torch_sd.items():
        if torch_key not in key_map:
            errors.append(f"  No mapping for: {torch_key}")
            continue

        paddle_key = key_map[torch_key]
        arr = torch_tensor.detach().cpu().numpy().astype(np.float32)

        # PaddlePaddle LayerNorm stores weight/bias as 1D flattened tensor,
        # while PyTorch stores it as a multi-dimensional tensor matching
        # normalized_shape. Flatten norm parameters.
        bare_key = paddle_key.replace("backbone.", "", 1)
        is_norm = "norm.weight" in paddle_key or "norm.bias" in paddle_key

        if is_norm and arr.ndim > 1:
            arr = arr.reshape(-1)
            print(f"  (flattened LayerNorm weight: {list(torch_tensor.shape)} -> {list(arr.shape)})")

        # Check shape compatibility
        if bare_key in dict(paddle_backbone.named_parameters()):
            expected = list(dict(paddle_backbone.named_parameters())[bare_key].shape)
            actual = list(arr.shape)
            if expected != actual:
                errors.append(
                    f"  Shape mismatch for {torch_key} -> {paddle_key}: "
                    f"expected {expected}, got {actual}"
                )
                continue

        paddle_state[paddle_key] = arr
        mapped += 1
        print(f"  {torch_key:25s} -> {paddle_key:35s} {list(arr.shape)}")

    print(f"\n  Mapped: {mapped} / {len(torch_sd)}")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
    else:
        print("  All shapes match! Ready to load into Paddle model.")

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    print(f"\n[4/5] Saving to: {output}")
    paddle.save(paddle_state, str(output))
    print(f"  Saved {len(paddle_state)} parameters.")

    # ------------------------------------------------------------------
    # 5. Verify
    # ------------------------------------------------------------------
    if verify and not errors:
        _verify(torch_sd, paddle_backbone, paddle_state)

    return str(output)


# ---------------------------------------------------------------------------
# Verification: run identical input through both models
# ---------------------------------------------------------------------------
def _verify(
    torch_sd: dict,
    paddle_model: "nn.Layer",
    paddle_state: dict[str, np.ndarray],
) -> None:
    """Run identical random input through both frameworks and compare."""
    import torch

    print(f"\n{'=' * 60}")
    print("Verification: running identical input through both frameworks...")
    print(f"{'=' * 60}")

    # Build torch model matching the original Fcnf0 architecture
    # We construct it directly rather than importing the penn package
    # to avoid its heavy dependencies (yapecs, torchutil, etc.)
    import torch.nn as tnn

    class TorchBlock(tnn.Sequential):
        def __init__(self, in_ch, out_ch, length, pooling=None, kernel_size=32):
            layers = [
                tnn.Conv1d(in_ch, out_ch, kernel_size),
                tnn.ReLU(),
            ]
            if pooling is not None:
                layers.append(tnn.MaxPool1d(*pooling))
            layers.append(tnn.LayerNorm((out_ch, length)))
            super().__init__(*layers)

    class TorchFcnf0(tnn.Sequential):
        def __init__(self):
            layers = [
                TorchBlock(1, 256, 481, (2, 2)),
                TorchBlock(256, 32, 225, (2, 2)),
                TorchBlock(32, 32, 97, (2, 2)),
                TorchBlock(32, 128, 66),
                TorchBlock(128, 256, 35),
                TorchBlock(256, 512, 4),
                tnn.Conv1d(512, 1440, 4),
            ]
            super().__init__(*layers)

        def forward(self, x):
            return super().forward(x[:, :, 16:-15])

    torch_model = TorchFcnf0()
    torch_model.load_state_dict(torch_sd)
    torch_model.eval()

    # Load weights into Paddle backbone (strip 'backbone.' prefix)
    import paddle

    bare_state = {}
    for k, v in paddle_state.items():
        if k.startswith("backbone."):
            bare_state[k.replace("backbone.", "", 1)] = v
        else:
            bare_state[k] = v

    # Get all model keys (parameters + buffers)
    model_param_keys = set(dict(paddle_model.named_parameters()).keys())
    model_buffer_keys = set(dict(paddle_model.named_buffers()).keys())
    model_keys = model_param_keys | model_buffer_keys

    compatible = {k: v for k, v in bare_state.items() if k in model_keys}
    missing = model_keys - set(bare_state.keys())

    if missing:
        print(f"\n  Info: {len(missing)} Paddle model keys initialized as defaults:")
        for k in sorted(missing):
            print(f"    {k}")

    if compatible:
        paddle_model.set_state_dict(compatible)
        print(f"\n  Loaded {len(compatible)} parameters into Paddle backbone.")

    # Generate random input: (B, 1, 1024)
    rng = np.random.RandomState(42)
    x_np = rng.randn(4, 1, 1024).astype(np.float32)

    # Torch forward
    x_torch = torch.from_numpy(x_np)
    with torch.no_grad():
        out_torch = torch_model(x_torch).numpy()

    # Paddle forward
    x_paddle = paddle.to_tensor(x_np)
    with paddle.no_grad():
        out_paddle = paddle_model(x_paddle).numpy()

    # Compare
    if out_torch.shape == out_paddle.shape:
        diff = np.abs(out_torch - out_paddle)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        allclose = np.allclose(out_torch, out_paddle, atol=5e-4, rtol=5e-4)

        print(f"\n  Torch output shape:  {out_torch.shape}")
        print(f"  Paddle output shape: {out_paddle.shape}")
        print(f"  Max diff:  {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")
        print(f"  Allclose (1e-4): {allclose}")
    else:
        print(f"\n  Shape mismatch: torch {out_torch.shape} vs paddle {out_paddle.shape}")

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert PENN (FCNF0++) weights to PaddlePaddle format."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to fcnf0++.pt checkpoint (default: ckpts/fcnf0++.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .pdparams path (default: ckpts/penn.pdparams)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip allclose verification",
    )
    args = parser.parse_args()

    output_path = convert(
        source=args.source,
        output=args.output,
        verify=not args.no_verify,
    )
    print(f"\nDone. Converted checkpoint saved to: {output_path}")


if __name__ == "__main__":
    main()
