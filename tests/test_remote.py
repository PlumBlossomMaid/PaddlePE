"""Tests for remote (subprocess server) mode."""

from __future__ import annotations

import numpy as np
import pytest

from paddlepe.remote import RemotePE


class TestRemotePE:
    """Test RemotePE subprocess server mode."""

    def test_create_and_infer(self):
        """RemotePE can start server, load model, and infer."""
        pe = RemotePE(model="fcpe", auto_shutdown=True)
        # Remove unused attributes
        assert pe._model_name == "fcpe"
        assert pe._process is not None
        assert pe._process.poll() is None  # still running

        # Create synthetic audio
        sr = 16000
        t = np.linspace(0, 0.3, int(sr * 0.3), dtype=np.float32)
        wav = np.sin(2 * np.pi * 440 * t)

        f0, conf = pe.infer(wav, sr)
        assert len(f0) > 0
        assert f0.dtype == np.float32
        if conf is not None:
            assert len(conf) > 0

        # Cleanup
        pe.__del__()

    def test_infer_with_paddle_tensor(self):
        """RemotePE accepts paddle.Tensor input."""
        import paddle

        pe = RemotePE(model="fcpe", auto_shutdown=True)
        sr = 16000
        t = paddle.linspace(0, 0.2, int(sr * 0.2))
        wav = paddle.sin(2 * paddle.pi * 440 * t)

        f0, conf = pe.infer(wav, sr)
        assert len(f0) > 0
        pe.__del__()

    def test_list_models(self):
        """Server reports available models."""
        import urllib.request

        pe = RemotePE(model="fcpe", auto_shutdown=True)
        resp = urllib.request.urlopen(f"{pe._base_url}/models")
        import json

        data = json.loads(resp.read())
        assert "models" in data
        assert "fcpe" in data["models"]
        pe.__del__()

    def test_health(self):
        """Server responds to health check."""
        import urllib.request

        pe = RemotePE(model="fcpe", auto_shutdown=True)
        resp = urllib.request.urlopen(f"{pe._base_url}/health")
        import json

        data = json.loads(resp.read())
        assert data["status"] == "ok"
        pe.__del__()

    @pytest.mark.skipif(
        not hasattr(__import__("paddle"), "is_compiled_with_cuda")
        or not __import__("paddle").is_compiled_with_cuda(),
        reason="CUDA not available",
    )
    def test_force_remote_creates_remote(self):
        """PE.create with force_remote=True returns RemotePE."""
        from paddlepe import PE

        pe = PE.create("fcpe", force_remote=True)
        from paddlepe.remote import RemotePE

        assert isinstance(pe, RemotePE)
        f0, conf = pe.infer(
            np.sin(np.linspace(0, 2 * np.pi * 440, 16000 // 100)).astype(np.float32),
            16000,
        )
        assert len(f0) > 0
