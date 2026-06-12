"""Remote pitch extraction client.

Communicates with a paddlePE server via HTTP.
Used as a transparent fallback when Paddle cannot be loaded
in the same process (e.g. when PyTorch already owns CUDA).

API mirrors BasePE so it can be used as a drop-in replacement.
"""

from __future__ import annotations

import io
import json
import socket
import sys
import time
from typing import Any

import numpy as np

from paddlepe._compat import to_numpy
from paddlepe.io.formats import HEADER_SIZE, decode_header


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
    """Get module path for server.

    Uses ``-m paddlepe.server`` instead of file path to avoid
    issues with Chinese characters in the installation path.
    """
    return "-m"


class RemotePE:
    """PE wrapper that delegates to a subprocess HTTP server.

    The server runs Paddle in an isolated process, avoiding CUDA context
    conflicts with other frameworks (PyTorch, TensorFlow, etc.).

    Does NOT inherit from BasePE so it can be imported without PaddlePaddle.
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
            "-m",
            "paddlepe.server",
            "--model",
            self._model_name,
            "--port",
            str(port),
        ]
        if self._ckpt:
            cmd += ["--ckpt", self._ckpt]

        import subprocess as _sp

        # On Windows, subprocess.Popen can hang when PaddlePaddle is
        # loaded in the parent process. Use start /B via cmd.exe as
        # a workaround when Paddle is present.
        if "paddle" in sys.modules:
            self._process = None
            shell_cmd = (
                f'start /B {sys.executable} -m paddlepe.server '
                f'--model {self._model_name} --port {port}'
            )
            if self._ckpt:
                shell_cmd += f" --ckpt {self._ckpt}"
            _sp.Popen(
                ["cmd.exe", "/d", "/s", "/c", shell_cmd],
                stdout=_sp.DEVNULL,
                stderr=_sp.DEVNULL,
            )
        else:
            self._process = _sp.Popen(
                cmd,
                stdout=_sp.DEVNULL,
                stderr=_sp.DEVNULL,
            )

        if not _wait_for_server(self._base_url, timeout=60.0):
            self._process.kill()
            raise RuntimeError(
                f"paddlePE server failed to start on port {port} "
                f"within 60 seconds."
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

    def forward(self, x: Any, *args, **kwargs) -> Any:
        raise NotImplementedError(
            "RemotePE does not support forward() for training. "
            "Use direct PE.create() in a process with PaddlePaddle."
        )

    def get_pitch(
        self,
        wav: Any,
        sr: int = 16000,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Alias for infer()."""
        return self.infer(wav, sr, **kwargs)

    def infer(
        self,
        wav: Any,
        sr: int = 16000,
        decoder: str = "local_argmax",
        threshold: float = 0.05,
        interp_uv: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Infer pitch via remote server.

        Accepts numpy.ndarray or any object with a ``.numpy()`` method
        (e.g. torch.Tensor). Returns numpy arrays regardless of input type.
        """
        import urllib.request

        # Normalize input to numpy array (client mode — no paddle import)
        wav_np = to_numpy(wav)

        # Normalize to int16 range if in float
        max_val = np.max(np.abs(wav_np))
        if max_val > 1.0:
            wav_int16 = wav_np.astype(np.int16)
        else:
            wav_int16 = (wav_np * 32767).astype(np.int16)

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
        if not self._auto_shutdown:
            return
        # Preferred: graceful HTTP shutdown
        try:
            import urllib.request as _ur

            _ur.urlopen(
                f"{self._base_url}/shutdown",
                data=b"{}",
                timeout=3,
            )
            return
        except Exception:
            pass
        # Fallback: kill process directly
        try:
            if self._process is not None:
                self._process.terminate()
                self._process.wait(timeout=3)
        except Exception:
            pass
