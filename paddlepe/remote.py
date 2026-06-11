"""Remote pitch extraction client.

Communicates with a paddlePE server via HTTP.
Used as a transparent fallback when Paddle cannot be loaded
in the same process (e.g. when PyTorch already owns CUDA).

API is identical to BasePE so it can be used as a drop-in replacement.
"""

from __future__ import annotations

import io
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from paddlepe.io.formats import HEADER_SIZE, decode_header
from paddlepe.models.base import BasePE


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Wait for server to be ready."""
    import urllib.request

    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(f"{url}/health", timeout=2)
            if resp.status == 200:
                return True
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.5)
    return False


def _server_script_path() -> str:
    """Get path to server.py."""
    return str(Path(__file__).parent / "server.py")


class RemotePE(BasePE):
    """PE wrapper that delegates to a subprocess HTTP server.

    The server runs Paddle in an isolated process, avoiding CUDA context
    conflicts with other frameworks (PyTorch, TensorFlow, etc.).
    """

    trainable = False
    support_onnx = False

    def __init__(
        self,
        model: str = "fcpe",
        ckpt: str | None = None,
        port: int | None = None,
        url: str | None = None,
        auto_shutdown: bool = True,
    ):
        super().__init__()
        self._model_name = model
        self._ckpt = ckpt
        self._auto_shutdown = auto_shutdown
        self._process: subprocess.Popen | None = None

        if url:
            self._base_url = url.rstrip("/")
        else:
            port = port or _find_free_port()
            self._base_url = f"http://127.0.0.1:{port}"
            self._start_server(port)

        # Load model on server
        self._load_model()

    def _start_server(self, port: int):
        """Start paddlePE server as a subprocess."""
        cmd = [
            sys.executable,
            _server_script_path(),
            "--model",
            self._model_name,
            "--port",
            str(port),
        ]
        if self._ckpt:
            cmd += ["--ckpt", self._ckpt]

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,  # Capture stderr for debugging
        )

        if not _wait_for_server(self._base_url, timeout=30.0):
            stderr_output = (
                self._process.stderr.read().decode("utf-8", errors="replace")
                if self._process.stderr
                else ""
            )
            self._process.kill()
            raise RuntimeError(
                f"paddlePE server failed to start on port {port}.\n"
                f"Server stderr:\n{stderr_output}"
            )

    def _load_model(self):
        """Load model on server."""
        import urllib.request

        data = json.dumps(
            {"model": self._model_name, "ckpt": self._ckpt}
        ).encode()
        req = urllib.request.Request(
            f"{self._base_url}/load",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read())
        if resp.status != 200:
            raise RuntimeError(
                f"Failed to load model: {result.get('error', 'unknown')}"
            )

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError(
            "RemotePE does not support forward() for training. "
            "Use direct PE.create() in a Paddle process."
        )

    def infer(
        self,
        wav: np.ndarray,
        sr: int = 16000,
        decoder: str = "local_argmax",
        threshold: float = 0.05,
        interp_uv: bool = False,
        **kwargs,
    ) -> tuple:
        """Infer pitch via remote server.

        Accepts either paddle.Tensor (converted to numpy) or numpy array.
        Returns numpy arrays regardless of input type.
        """
        import urllib.request

        # Convert to numpy int16 WAV
        if hasattr(wav, "numpy"):
            wav_np = wav.numpy()
        else:
            wav_np = np.asarray(wav, dtype=np.float32)

        # Normalize to int16 range if in float
        if wav_np.dtype == np.float32 or wav_np.dtype == np.float64:
            max_val = np.max(np.abs(wav_np))
            if max_val > 1.0:
                wav_int16 = wav_np.astype(np.int16)
            else:
                wav_int16 = (wav_np * 32767).astype(np.int16)
        else:
            wav_int16 = wav_np.astype(np.int16)

        # Build WAV bytes
        wav_bytes = self._wav_to_bytes(wav_int16, sr)

        # Send request
        headers = {
            "Content-Type": "audio/wav",
            "X-Decoder": decoder,
            "X-Threshold": str(threshold),
            "X-Interp-UV": str(interp_uv).lower(),
        }
        req = urllib.request.Request(
            f"{self._base_url}/infer",
            data=wav_bytes,
            headers=headers,
        )
        resp = urllib.request.urlopen(req)
        f0_bytes = resp.read()

        # Parse .f0 response
        header = decode_header(f0_bytes[:HEADER_SIZE])
        f0_offset = header.header_size
        f0 = np.frombuffer(
            f0_bytes,
            dtype=np.float32,
            count=header.num_frames,
            offset=f0_offset,
        )
        conf: np.ndarray | None = None
        if header.flags & 0x01:
            conf = np.frombuffer(
                f0_bytes,
                dtype=np.float32,
                count=header.num_frames,
                offset=f0_offset + header.num_frames * 4,
            )

        return f0.copy(), conf.copy() if conf is not None else None

    def _wav_to_bytes(self, samples: np.ndarray, sr: int) -> bytes:
        """Convert numpy int16 array to WAV bytes."""
        import struct as _struct

        n_channels = 1
        bytes_per_sample = 2
        data_size = len(samples) * bytes_per_sample
        buf = io.BytesIO()
        # RIFF header
        buf.write(b"RIFF")
        buf.write(_struct.pack("<I", 36 + data_size))
        buf.write(b"WAVE")
        # fmt chunk
        buf.write(b"fmt ")
        buf.write(
            _struct.pack(
                "<IHHIIHH",
                16,
                1,
                n_channels,
                sr,
                sr * bytes_per_sample,
                bytes_per_sample,
                16,
            )
        )
        # data chunk
        buf.write(b"data")
        buf.write(_struct.pack("<I", data_size))
        buf.write(samples.tobytes())
        return buf.getvalue()

    def __del__(self):
        """Shutdown server on cleanup."""
        if self._auto_shutdown and self._process is not None:
            self._process.terminate()
            self._process.wait(timeout=5)
