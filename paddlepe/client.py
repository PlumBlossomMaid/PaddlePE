"""Client-only PE facade.

Used when PaddlePaddle cannot be imported in the current process
(e.g., due to cuDNN DLL conflict with PyTorch on Windows CUDA).

In this mode:
  - Only inference is supported (via subprocess server)
  - Training and export are unavailable
  - Model registration is not supported

Usage:
    from paddlepe import PE

    pe = PE.create("fcpe")
    f0, confidence = pe.infer(wav, sr)
"""

from __future__ import annotations

import json
import urllib.request


class ClientPE:
    """PE facade for client-only mode (no PaddlePaddle available).

    All operations delegate to a remote paddlePE server via HTTP.
    """

    @staticmethod
    def create(name: str, ckpt: str | None = None, **kwargs):
        """Create a pitch extraction model via remote server.

        Starts a Paddle subprocess server and returns a RemotePE
        client that communicates with it.
        """
        from paddlepe.remote import RemotePE

        return RemotePE(model=name, ckpt=ckpt, **kwargs)

    @staticmethod
    def list_models() -> list[str]:
        """Query available models by starting a temporary server.

        Note: This starts a PaddlePaddle subprocess and may take
        a few seconds on first call. Results are not cached.
        """
        from paddlepe.remote import RemotePE

        temp = RemotePE(model="fcpe", auto_shutdown=True)
        try:
            resp = urllib.request.urlopen(f"{temp._base_url}/models", timeout=10)
            return list(json.loads(resp.read()).get("models", []))
        finally:
            temp.__del__()

    @staticmethod
    def register(name: str):
        """Not supported in client-only mode."""
        raise NotImplementedError(
            "Client-only mode (PaddlePaddle unavailable) does not "
            "support model registration. Use in a PaddlePaddle process."
        )
