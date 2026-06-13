"""Convert RMVPE checkpoint: add backbone. prefix for wrapped model loading."""

from __future__ import annotations

from pathlib import Path

import paddle


def convert_rmvpe_ckpt(src: str | Path, dst: str | Path):
    """Convert RMVPE checkpoint to match paddlePE's backbone-wrapped naming."""
    src, dst = Path(src), Path(dst)
    ckpt = paddle.load(str(src))

    new_ckpt = {}
    for key, val in ckpt.items():
        # Add backbone. prefix since RMVPEPE wraps the backbone
        new_key = f"backbone.{key}"
        new_ckpt[new_key] = val

    paddle.save(new_ckpt, str(dst))
    print(f"Converted {len(new_ckpt)} keys from {src} → {dst}")

    # Verify
    from paddlepe.models.rmvpe.infer import RMVPEPE

    model = RMVPEPE(n_blocks=4, n_gru=1)
    model.set_state_dict(new_ckpt)
    print("Verification: model loaded OK (no warnings = clean)")


if __name__ == "__main__":
    src = "/home/aistudio/plum/ocean_project/如来唱/ckpts/rmvpe/rmvpe.pdparams"
    dst = "/home/aistudio/plum/ocean_project/paddlePE/ckpts/rmvpe.pdparams"
    convert_rmvpe_ckpt(src, dst)
