"""Convert FCPE checkpoint to paddlePE format."""

from __future__ import annotations

from pathlib import Path

import paddle


def convert_fcpe_ckpt(src: str | Path, dst: str | Path):
    """Convert FCPE checkpoint to match MelConformerF0 key naming."""
    src, dst = Path(src), Path(dst)

    raw = paddle.load(str(src))
    weights = raw["model"]  # actual model state dict

    new_ckpt = {}
    for old_key, val in weights.items():
        new_key = old_key

        # input_stack → input_proj
        if new_key.startswith("input_stack"):
            new_key = new_key.replace("input_stack", "input_proj", 1)

        # net.encoder_layers → encoder.layers
        if new_key.startswith("net.encoder_layers"):
            new_key = new_key.replace("net.encoder_layers", "encoder.layers", 1)

        # Handle conformer key remapping
        parts = new_key.split(".")
        if "conformer" in parts:
            # e.g. encoder.layers.0.conformer.net.0.weight
            idx = parts.index("conformer")
            if idx > 0 and idx + 2 < len(parts):
                prefix = parts[:idx]
                # net.0 → ff1.0 (LayerNorm)
                # net.2 → ff1.1 (Linear)
                # net.4 → conv.net (ConformerConvModule)
                # net.6 → ff1.2 (Linear)
                # net.8 → ff1.3 (Dropout)
                seq_idx = int(parts[idx + 2])
                sub_key = parts[idx + 3 :]
                if seq_idx == 0:
                    new_parts = prefix + ["ff1", "0"] + sub_key
                elif seq_idx == 1:
                    new_parts = prefix + ["ff1", "1"] + sub_key
                elif seq_idx == 2:
                    new_parts = prefix + ["ff1", "2"] + sub_key
                elif seq_idx == 3:
                    new_parts = prefix + ["ff1", "3"] + sub_key
                elif seq_idx == 4:
                    new_parts = prefix + ["conv", "net"] + sub_key
                elif seq_idx == 5:
                    new_parts = prefix + ["conv", "net", "4"] + sub_key
                elif seq_idx == 6:
                    new_parts = prefix + ["ff1", "4"] + sub_key
                elif seq_idx == 7:
                    new_parts = prefix + ["ff1", "5"] + sub_key
                else:
                    new_parts = prefix + sub_key
                new_key = ".".join(new_parts)

        # Also check for GroupNorm naming differences
        # GroupNorm uses weight/bias not weight/bias
        # This should be fine as-is

        new_ckpt[f"backbone.{new_key}"] = val

    paddle.save(new_ckpt, str(dst))
    print(f"Converted {len(new_ckpt)} keys from {src} → {dst}")

    from paddlepe.models.fcpe.infer import FCPEPE

    pe = FCPEPE()
    missing, unexpected = pe.set_state_dict(new_ckpt)
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    if not missing and not unexpected:
        print("FCPE weight load OK: all keys matched!")


if __name__ == "__main__":
    src = "/home/aistudio/plum/ocean_project/pes/FCPE_paddle/train/paddlefcpe/assets/fcpe_c_v001.pdparams"
    dst = "/home/aistudio/plum/ocean_project/paddlePE/ckpts/fcpe.pdparams"
    convert_fcpe_ckpt(src, dst)
