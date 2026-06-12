"""ONNX and static graph export for paddlePE models.

Only available when PaddlePaddle can be loaded directly.
"""

from __future__ import annotations

from pathlib import Path

from paddlepe._compat import require_paddle

require_paddle("export")

import paddle  # noqa: E402


def export_onnx(
    model: paddle.nn.Layer,
    input_spec: list[paddle.static.InputSpec],
    output_path: str | Path,
    model_name: str = "model",
):
    """Export a model to ONNX format.

    Args:
        model: model to export (must support ONNX export)
        input_spec: list of InputSpec describing model inputs
        output_path: where to save the .onnx file
        model_name: name for the ONNX graph
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    # Export with dynamic batch
    paddle.onnx.export(
        model,
        str(output_path.with_suffix("")),
        input_spec=input_spec,
        opset_version=11,
        enable_onnx_checker=True,
    )
    onnx_path = output_path.with_suffix(".onnx")
    print(f"ONNX model exported to: {onnx_path}")


def export_static_graph(
    model: paddle.nn.Layer,
    input_spec: list[paddle.static.InputSpec],
    output_path: str | Path,
):
    """Export a model to Paddle static graph (inference-only).

    Args:
        model: model to export
        input_spec: list of InputSpec
        output_path: where to save the .pdmodel / .pdiparams
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    paddle.jit.save(
        layer=model,
        path=str(output_path.with_suffix("")),
        input_spec=input_spec,
    )
    print(f"Static graph exported to: {output_path.with_suffix('.pdmodel')}")


def export_fcpe_to_onnx(
    model: paddle.nn.Layer,
    output_path: str | Path,
    mel_bins: int = 128,
    max_frames: int = 1000,
):
    """Export FCPE model to ONNX with typical input shape."""
    input_spec = [
        paddle.static.InputSpec(
            shape=[None, None, mel_bins],
            dtype="float32",
            name="mel",
        ),
    ]
    export_onnx(model, input_spec, output_path, model_name="mel_conformer_f0")


def export_rmvpe_to_onnx(
    model: paddle.nn.Layer,
    output_path: str | Path,
    mel_bins: int = 128,
    max_frames: int = 1000,
):
    """Export RMVPE model to ONNX with typical input shape."""
    input_spec = [
        paddle.static.InputSpec(
            shape=[None, mel_bins, max_frames],
            dtype="float32",
            name="mel",
        ),
    ]
    export_onnx(model, input_spec, output_path, model_name="rmvpe_unet")
