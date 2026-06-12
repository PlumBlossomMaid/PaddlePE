"""PaddlePE inference server.

Serves pitch extraction models via HTTP, running Paddle in an isolated process.

Usage:
    python -m paddlepe.server --model fcpe --port 18560
    paddlepe server --model fcpe --port 18560
    python -m paddlepe.server --no-preload --port 18560   # headless, for /models only

API endpoints:
    GET /health          → {"status": "ok"}
    GET /models          → {"models": ["fcpe", "rmvpe", ...]}
    POST /load           → load model: {"model": "fcpe"}
    POST /infer          → infer pitch: binary WAV → binary .f0 response
"""

from __future__ import annotations

import argparse
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

# Paddle is imported lazily — only inside request handlers
_MODEL: object | None = None
_MODEL_NAME: str = ""
_SERVER: HTTPServer | None = None


def _shutdown_after_delay():
    """Shutdown server after a short delay (response must be sent first)."""
    import time as _t

    _t.sleep(0.1)
    if _SERVER:
        _SERVER.shutdown()


class _PitchHandler(BaseHTTPRequestHandler):
    """HTTP handler for pitch extraction requests."""

    # Silence default HTTP log messages
    def log_message(self, format, *args):
        pass

    def _send_json(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_f0(
        self,
        code: int,
        f0: np.ndarray,
        confidence: np.ndarray | None,
        sr: int,
        hop: int,
    ):
        """Send F0 data as .f0 binary format."""
        from paddlepe.io.writer import write_f0

        buf = io.BytesIO()
        write_f0(buf, f0, confidence, sr, hop)
        body = buf.getvalue()
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    _TEXT_TYPES = {".wav", ".mp3", ".flac", ".ogg"}

    def do_GET(self):
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "model": _MODEL_NAME or None,
                    "loaded": _MODEL is not None,
                },
            )
        elif self.path == "/models":
            # Lazy import: triggers PaddlePaddle only when queried
            from paddlepe import PE

            self._send_json(200, {"models": PE.list_models()})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        global _MODEL, _MODEL_NAME, _SERVER

        if self.path == "/shutdown":
            self._send_json(200, {"status": "shutting down"})
            import threading as _t

            _t.Thread(target=_shutdown_after_delay, daemon=True).start()
            return

        if self.path == "/load":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body)
            name = data.get("model", "fcpe")
            ckpt = data.get("ckpt")
            try:
                _load_model(name, ckpt)
                _MODEL_NAME = name
                self._send_json(200, {"status": "loaded", "model": name})
            except Exception as e:
                self._send_json(500, {"error": str(e)})

        elif self.path == "/infer":
            if _MODEL is None:
                self._send_json(
                    400, {"error": "no model loaded, call /load first"}
                )
                return

            # Read raw WAV bytes
            length = int(self.headers.get("Content-Length", 0))
            wav_bytes = self.rfile.read(length)

            # Parse WAV file properly
            import wave as _wave

            try:
                with _wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                    sr = wf.getframerate()
                    n_channels = wf.getnchannels()
                    n_frames = wf.getnframes()
                    raw = wf.readframes(n_frames)
                    wav_np = (
                        np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    if n_channels > 1:
                        wav_np = wav_np.reshape(-1, n_channels).mean(
                            axis=1
                        )  # mono
            except Exception:
                # Fallback: treat as raw PCM
                wav_np = (
                    np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
                sr = 16000

            # Read inference params from headers
            params = {}
            if self.headers.get("X-Decoder"):
                params["decoder"] = self.headers["X-Decoder"]
            if self.headers.get("X-Threshold"):
                params["threshold"] = float(self.headers["X-Threshold"])
            if self.headers.get("X-Interp-UV"):
                params["interp_uv"] = (
                    self.headers["X-Interp-UV"].lower() == "true"
                )

            import paddle

            wav_t = paddle.to_tensor(wav_np)
            f0_t, conf_t = _MODEL.infer(wav_t, sr, **params)
            f0_np = f0_t.numpy()
            conf_np = conf_t.numpy() if conf_t is not None else None

            hop = getattr(_MODEL, "hop_length", sr // 100)
            self._send_f0(200, f0_np, conf_np, sr, hop)

        else:
            self._send_json(404, {"error": "not found"})


def _load_model(name: str, ckpt: str | None = None):
    """Lazy-import Paddle and load model."""
    global _MODEL
    import paddle  # noqa: F401 — triggers Paddle import only here

    from paddlepe import PE

    _MODEL = PE.create(name, ckpt=ckpt)
    _MODEL.eval()


def run_server(
    model: str | None = "fcpe",
    port: int = 18560,
    ckpt: str | None = None,
    no_preload: bool = False,
):
    """Start the inference server.

    This is meant to run in a subprocess. Paddle is imported lazily
    so that the parent process (e.g. PyTorch) never conflicts.

    Args:
        model: Model name to preload, or None if no_preload=True.
        port: TCP port to listen on.
        ckpt: Optional checkpoint path.
        no_preload: If True, start server without loading any model.
            Client must call /load before /infer. Useful for listing
            models without loading an actual model.
    """
    global _SERVER

    if no_preload:
        model = None
    if model is not None:
        _load_model(model, ckpt)

    _SERVER = HTTPServer(("127.0.0.1", port), _PitchHandler)
    _SERVER.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="paddlePE inference server")
    parser.add_argument("--model", default="fcpe", help="Model name")
    parser.add_argument("--port", type=int, default=18560, help="Server port")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path")
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Start without preloading a model. Use /load later.",
    )
    args = parser.parse_args()
    run_server(
        args.model, args.port, args.ckpt, no_preload=args.no_preload
    )


if __name__ == "__main__":
    main()
