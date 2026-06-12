"""CLI for paddlePE: single command with auto-detect mode.

Usage:
    paddlepe input.wav -o out.f0                # extract + binary
    paddlepe input.wav -o out.csv -f csv         # extract + csv
    paddlepe in.csv -o out.f0                    # convert csv→f0
    paddlepe in.f0 -o out.csv                    # convert f0→csv
    paddlepe -l                                  # list models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _list_models() -> list[str]:
    """List available models without importing all backends."""
    from paddlepe import PE

    return PE.list_models()


def _do_extract(args):
    """Extract pitch from WAV file."""
    from paddlepe import PE

    # Load audio
    try:
        import soundfile as sf
    except ImportError:
        print(
            "Error: 'soundfile' required for WAV I/O. "
            "Install with: pip install soundfile"
        )
        sys.exit(1)

    wav, sr = sf.read(args.input, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)  # mono

    import paddle

    wav_t = paddle.to_tensor(wav)

    # Create model
    model_name = args.model or "fcpe"
    ckpt = args.ckpt
    pe = PE.create(model_name, ckpt=ckpt)

    # Infer
    f0, confidence = pe.infer(wav_t, sr, interp_uv=args.interp_uv)
    f0_np = f0.numpy()
    conf_np = confidence.numpy() if confidence is not None else None

    f0_min = float(np.nanmin(f0_np[f0_np > 0])) if (f0_np > 0).any() else 32.0
    f0_max = float(np.nanmax(f0_np)) if f0_np.any() else 2100.0

    output_format = args.format or Path(args.output).suffix.lstrip(".") or "f0"
    output_path = Path(args.output)

    if output_format == "f0":
        from paddlepe.io import write_f0

        write_f0(
            output_path, f0_np, conf_np, int(sr), int(sr / 100), f0_min, f0_max
        )
    elif output_format == "csv":
        from paddlepe.io import write_csv

        write_csv(output_path, f0_np, conf_np, int(sr), int(sr / 100))
    else:
        print(f"Error: Unknown format: {output_format}")
        sys.exit(1)

    print(f"Wrote {len(f0_np)} frames to {output_path}")


def _do_convert(args):
    """Convert between formats."""
    from paddlepe.io import read

    f0, confidence, sr, hop = read(args.input)

    output_format = args.format or Path(args.output).suffix.lstrip(".") or "f0"
    output_path = Path(args.output)

    f0_min = float(np.nanmin(f0[f0 > 0])) if (f0 > 0).any() else 32.0
    f0_max = float(np.nanmax(f0)) if f0.any() else 2100.0

    if output_format == "f0":
        from paddlepe.io import write_f0

        write_f0(output_path, f0, confidence, sr, hop, f0_min, f0_max)
    elif output_format == "csv":
        from paddlepe.io import write_csv

        write_csv(output_path, f0, confidence, sr, hop)
    else:
        print(f"Error: Unknown format: {output_format}")
        sys.exit(1)

    print(f"Converted {len(f0)} frames to {output_path}")


def main():
    # Check for server mode FIRST, before argparse, to avoid
    # positional arguments being misinterpreted as subparser commands.
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        server_parser = argparse.ArgumentParser(
            description="Start inference server"
        )
        server_parser.add_argument("--model", default="fcpe", help="Model name")
        server_parser.add_argument(
            "--port", type=int, default=18560, help="Server port"
        )
        server_parser.add_argument(
            "--ckpt", default=None, help="Checkpoint path"
        )
        server_parser.add_argument(
            "--no-preload",
            action="store_true",
            help="Start without loading model (lazy load on first request)",
        )
        args = server_parser.parse_args(sys.argv[2:])
        from paddlepe.server import run_server

        run_server(
            model=args.model,
            port=args.port,
            ckpt=args.ckpt,
            no_preload=args.no_preload,
        )
        return

    parser = argparse.ArgumentParser(
        description="paddlePE: Unified Pitch Extraction Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  paddlepe input.wav -o out.f0                # extract pitch
  paddlepe input.wav -o out.csv -f csv         # extract as CSV
  paddlepe in.csv -o out.f0                    # convert to f0
  paddlepe in.f0 -o out.csv                    # convert to CSV
  paddlepe -l                                  # list models
  paddlepe server --model fcpe --port 18560    # start server
        """,
    )

    # Global options
    parser.add_argument(
        "input", nargs="?", type=str, help="Input file (.wav, .f0, .csv)"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=False, help="Output file"
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["f0", "csv"],
        default=None,
        help="Output format",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Model name (default: fcpe)",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Path to checkpoint"
    )
    parser.add_argument(
        "--interp-uv", action="store_true", help="Interpolate unvoiced frames"
    )
    parser.add_argument(
        "-l", "--list", action="store_true", help="List available models"
    )

    args = parser.parse_args()

    if args.list:
        models = _list_models()
        print("Available models:")
        for m in models:
            print(f"  - {m}")
        return

    if args.input is None:
        parser.print_help()
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)

    # Auto-detect: extract or convert
    suffix = input_path.suffix.lower()
    if suffix == ".wav":
        _do_extract(args)
    elif suffix in (".f0", ".csv", ".pv", ".tsv"):
        _do_convert(args)
    else:
        print(f"Error: unknown input format: {suffix}")
        sys.exit(1)


if __name__ == "__main__":
    main()
