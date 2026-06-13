#!/usr/bin/env python3
"""Convert torchcrepe PyTorch weights to PaddlePaddle format.

Usage:
    python scripts/convert_crepe.py                    # converts full model
    python scripts/convert_crepe.py --capacity tiny    # converts tiny model
    python scripts/convert_crepe.py --no-verify        # skip allclose test

The conversion script:
  1. Loads the PyTorch checkpoint from torchcrepe/assets/{capacity}.pth
  2. Builds a CrepeBackbone in Paddle (eval mode)
  3. Maps every torch parameter to its Paddle counterpart
  4. Saves the result to ckpts/crepe.pdparams
  5. Optionally verifies by running identical random input through both
     frameworks and checking allclose.
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
TORCH_ASSETS = (
    PROJECT_ROOT.parent / "pes" / "torchcrepe" / "torchcrepe" / "assets"
)
PADDLE_CKPTS = PROJECT_ROOT / "ckpts"


# ---------------------------------------------------------------------------
# Key mapping from original torchcrepe (Conv2d named layers) to Paddle
# (ModuleList naming).
#
# Original keys:
#   conv{N}.weight, conv{N}.bias
#   conv{N}_BN.weight, conv{N}_BN.bias
#   conv{N}_BN.running_mean, conv{N}_BN.running_var
#   conv{N}_BN.num_batches_tracked   (skip - not used in Paddle)
#   classifier.weight, classifier.bias
#
# Paddle keys:
#   convs.{I}.weight, convs.{I}.bias
#   bns.{I}.weight, bns.{I}.bias
#   bns.{I}._mean, bns.{I}._variance
#   fc2.weight, fc2.bias
# ---------------------------------------------------------------------------
def _build_key_map() -> dict[str, str]:
    """Build mapping from torch key names to Paddle key names."""
    mapping = {}

    conv_names = ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6"]
    for i, name in enumerate(conv_names):
        mapping[f"{name}.weight"] = f"convs.{i}.weight"
        mapping[f"{name}.bias"] = f"convs.{i}.bias"
        mapping[f"{name}_BN.weight"] = f"bns.{i}.weight"
        mapping[f"{name}_BN.bias"] = f"bns.{i}.bias"
        mapping[f"{name}_BN.running_mean"] = f"bns.{i}._mean"
        mapping[f"{name}_BN.running_var"] = f"bns.{i}._variance"

    mapping["classifier.weight"] = "fc2.weight"
    mapping["classifier.bias"] = "fc2.bias"

    return mapping


def _invert_key_map() -> dict[str, str]:
    """Build reverse mapping: Paddle key -> torch key."""
    fwd = _build_key_map()
    return {v: k for k, v in fwd.items()}


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------
def convert(
    capacity: str = "full",
    output: str | None = None,
    verify: bool = True,
) -> str:
    """Convert torchcrepe weights to Paddle format.

    Args:
        capacity: 'full' or 'tiny'
        output: output .pdparams path (default: ckpts/crepe.pdparams)
        verify: run allclose test after conversion

    Returns:
        Path to the saved .pdparams file.
    """
    import torch

    assert capacity in ("full", "tiny"), f"Unknown capacity: {capacity}"

    torch_ckpt_path = TORCH_ASSETS / f"{capacity}.pth"
    if not torch_ckpt_path.exists():
        raise FileNotFoundError(
            f"PyTorch checkpoint not found: {torch_ckpt_path}"
        )

    if output is None:
        # Default: full -> crepe.pdparams (primary), tiny -> crepe_tiny.pdparams
        suffix = "" if capacity == "full" else f"_{capacity}"
        output = PADDLE_CKPTS / f"crepe{suffix}.pdparams"

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    key_map = _build_key_map()
    keys_to_skip = {
        "conv1_BN.num_batches_tracked",
        "conv2_BN.num_batches_tracked",
        "conv3_BN.num_batches_tracked",
        "conv4_BN.num_batches_tracked",
        "conv5_BN.num_batches_tracked",
        "conv6_BN.num_batches_tracked",
    }

    # ------------------------------------------------------------------
    # 1. Load PyTorch checkpoint
    # ------------------------------------------------------------------
    print(f"[1/5] Loading torch checkpoint: {torch_ckpt_path}")
    torch_ckpt = torch.load(
        torch_ckpt_path, map_location="cpu", weights_only=True
    )

    print(f"  Source keys ({len(torch_ckpt)} total):")
    for k, v in torch_ckpt.items():
        shape = list(v.shape)
        if k in keys_to_skip:
            print(f"  {k:40s} {shape!s:20s}  (skipping)")
        elif k in key_map:
            print(f"  {k:40s} {shape!s:20s} -> {key_map[k]}")
        else:
            print(f"  {k:40s} {shape!s:20s}  (no mapping)")

    # ------------------------------------------------------------------
    # 2. Build Paddle model
    # ------------------------------------------------------------------
    print(f"\n[2/5] Building Paddle CrepeBackbone (capacity={capacity})...")
    import paddle

    sys.path.insert(0, str(PROJECT_ROOT))
    from paddlepe.models.crepe.backbone import CrepeBackbone

    paddle_model = CrepeBackbone(capacity=capacity)
    paddle_model.eval()

    # ------------------------------------------------------------------
    # 3. Build state dict
    # ------------------------------------------------------------------
    print("\n[3/5] Mapping weights to Paddle format...")
    paddle_state = {}
    mapped = 0
    skipped = 0
    errors = []

    for torch_key, torch_tensor in torch_ckpt.items():
        if torch_key in keys_to_skip:
            skipped += 1
            continue

        if torch_key not in key_map:
            errors.append(f"  No mapping for: {torch_key}")
            continue

        paddle_key = key_map[torch_key]
        arr = torch_tensor.detach().cpu().numpy().astype(np.float32)

        # Paddle Linear weight uses [in_features, out_features] layout,
        # so we need to transpose from PyTorch's [out_features, in_features].
        if torch_key == "classifier.weight":
            arr = arr.T  # [360, 2048] -> [2048, 360]
            print("  (transposed for Paddle Linear weight format)")

        # Check shape compatibility
        if paddle_key in dict(paddle_model.named_parameters()):
            expected = list(
                dict(paddle_model.named_parameters())[paddle_key].shape
            )
            actual = list(arr.shape)
            if expected != actual:
                errors.append(
                    f"  Shape mismatch for {torch_key} -> {paddle_key}: "
                    f"expected {expected}, got {actual}"
                )
                continue

        paddle_state[paddle_key] = arr
        mapped += 1
        print(f"  {torch_key:40s} -> {paddle_key:20s} {list(arr.shape)}")

    # Prefix with 'backbone.' so the state dict can be loaded into CrepePE
    # (which wraps CrepeBackbone as self.backbone)
    paddle_state_pe = {f"backbone.{k}": v for k, v in paddle_state.items()}
    # Also keep the bare version for loading into CrepeBackbone directly
    # (it will be saved as the primary state dict; the PE version stored separately)

    print(f"\n  Mapped: {mapped}, Skipped (num_batches_tracked): {skipped}")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")

    if not errors:
        print("  All shapes match! Ready to load into Paddle model.")
    else:
        print(
            f"  WARNING: {len(errors)} error(s) occurred. "
            f"Output may be incomplete."
        )

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    print(f"\n[4/5] Saving to: {output}")
    paddle.save(
        paddle_state_pe, str(output)
    )  # Save with 'backbone.' prefix for CrepePE
    print(
        f"  Saved {len(paddle_state_pe)} parameters (with 'backbone.' prefix)."
    )

    # Also save a bare version for CrepeBackbone directly
    bare_output = output.parent / output.name.replace(
        ".pdparams", "_backbone.pdparams"
    )
    paddle.save(paddle_state, str(bare_output))
    print(f"  Also saved bare backbone keys -> {bare_output}")
    print(f"  ({len(paddle_state)} parameters, no prefix)")

    # ------------------------------------------------------------------
    # 5. Verify
    # ------------------------------------------------------------------
    if verify and not errors:
        _verify(
            torch_ckpt_path,
            paddle_model,
            paddle_state_pe,
            paddle_state,
            capacity,
        )

    return str(output)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _verify(
    torch_ckpt_path: Path,
    paddle_model: nn.Layer,
    paddle_state_pe: dict[str, np.ndarray],
    paddle_state_bare: dict[str, np.ndarray],
    capacity: str,
) -> None:
    """Run identical random input through both frameworks and compare."""
    import torch

    print(f"\n{'=' * 60}")
    print("Verification: running identical input through both frameworks...")
    print(f"{'=' * 60}")

    # Build torch model
    torchcrepe_dir = str(TORCH_ASSETS.parent.parent)  # torchcrepe package root
    if torchcrepe_dir not in sys.path:
        sys.path.insert(0, torchcrepe_dir)

    from torchcrepe.model import Crepe as TorchCrepe

    torch_model = TorchCrepe(model=capacity)
    torch_ckpt = torch.load(
        torch_ckpt_path, map_location="cpu", weights_only=True
    )
    torch_model.load_state_dict(torch_ckpt)
    torch_model.eval()

    # Load weights into Paddle model
    import paddle

    paddle_params = {
        k: paddle.to_tensor(v) for k, v in paddle_state_bare.items()
    }

    # Identify compatible keys
    model_param_keys = set(dict(paddle_model.named_parameters()).keys())
    model_buffer_keys = set(dict(paddle_model.named_buffers()).keys())
    model_keys = model_param_keys | model_buffer_keys

    compatible = {k: v for k, v in paddle_params.items() if k in model_keys}
    missing = model_keys - set(paddle_params.keys())

    if missing:
        print(
            f"\n  Info: {len(missing)} Paddle model keys initialized as defaults "
            f"(running stats may differ):"
        )
        for k in sorted(missing):
            print(f"    {k}")

    if compatible:
        paddle_model.set_state_dict(compatible)
        print(f"\n  Loaded {len(compatible)} parameters into Paddle model.")

    # Generate random input
    rng = np.random.RandomState(42)
    x_np = rng.randn(4, 1024).astype(np.float32)

    # Torch forward: torch model expects (B, 1024) and adds dims internally
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
        allclose = np.allclose(out_torch, out_paddle, atol=1e-4, rtol=1e-4)

        print(f"\n  Torch output shape:  {out_torch.shape}")
        print(f"  Paddle output shape: {out_paddle.shape}")
        print(f"  Max diff:  {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")
        print(f"  Allclose (1e-4): {allclose}")
    else:
        print(
            f"\n  Shape mismatch: torch {out_torch.shape} vs paddle {out_paddle.shape}"
        )

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert torchcrepe weights to PaddlePaddle format."
    )
    parser.add_argument(
        "--capacity",
        choices=["full", "tiny"],
        default="full",
        help="Model capacity (default: full)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .pdparams path (default: ckpts/crepe.pdparams)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip allclose verification",
    )
    args = parser.parse_args()

    output_path = convert(
        capacity=args.capacity,
        output=args.output,
        verify=not args.no_verify,
    )
    print(f"\nDone. Converted checkpoint saved to: {output_path}")


if __name__ == "__main__":
    main()
