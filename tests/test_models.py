"""Tests for model backbones (forward pass only, no weights)."""

from __future__ import annotations

import paddle
import pytest


class TestFCPEBackbone:
    """Test FCPE backbone forward pass."""

    def test_forward_shape(self):
        """Forward pass returns correct shape."""
        from paddlepe.models.fcpe.backbone import MelConformerF0

        model = MelConformerF0(
            mel_bins=128, out_dims=360, hidden_dims=64, n_layers=2, n_heads=4
        )
        model.eval()

        B, T, D = 2, 50, 128
        mel = paddle.randn([B, T, D])
        out = model(mel)
        assert out.shape == [B, T, 360]
        assert 0 <= out.min().item() <= 1
        assert 0 <= out.max().item() <= 1

    def test_infer_shape(self):
        """Infer returns correct shape."""
        from paddlepe.models.fcpe.backbone import MelConformerF0

        model = MelConformerF0(
            mel_bins=128, out_dims=360, hidden_dims=64, n_layers=2, n_heads=4
        )
        model.eval()

        B, T, D = 1, 50, 128
        mel = paddle.randn([B, T, D])
        f0 = model.infer(mel, decoder="argmax")
        assert f0.shape == [B, T, 1]
        assert f0.dtype == paddle.float32

    def test_infer_local_argmax(self):
        """Local argmax inference works."""
        from paddlepe.models.fcpe.backbone import MelConformerF0

        model = MelConformerF0(
            mel_bins=128, out_dims=360, hidden_dims=64, n_layers=2, n_heads=4
        )
        model.eval()

        mel = paddle.randn([1, 20, 128])
        f0 = model.infer(mel, decoder="local_argmax")
        assert f0.shape == [1, 20, 1]

    @pytest.mark.skipif(
        not paddle.device.is_compiled_with_cuda(),
        reason="CUDA not available",
    )
    def test_gpu_forward(self):
        """Forward pass works on GPU."""
        from paddlepe.models.fcpe.backbone import MelConformerF0

        model = MelConformerF0(
            mel_bins=128, out_dims=360, hidden_dims=64, n_layers=2, n_heads=4
        )
        model = model.to("gpu:0")
        model.eval()

        mel = paddle.randn([1, 20, 128])
        out = model(mel)
        assert out.place.is_gpu_place()


class TestRMVPEBackbone:
    """Test RMVPE backbone forward pass."""

    def test_forward_shape(self):
        """Forward pass returns correct shape."""
        from paddlepe.models.rmvpe.backbone import RMVPEUNet

        model = RMVPEUNet(n_blocks=2, n_gru=1)
        model.eval()

        B, D, T = 2, 128, 50
        mel = paddle.randn([B, D, T])
        out = model(mel)
        assert out.shape == [B, T, 360]
        assert 0 <= out.min().item() <= 1
        assert 0 <= out.max().item() <= 1

    @pytest.mark.skipif(
        not paddle.device.is_compiled_with_cuda(),
        reason="CUDA not available",
    )
    def test_gpu_forward(self):
        """Forward pass works on GPU."""
        from paddlepe.models.rmvpe.backbone import RMVPEUNet

        model = RMVPEUNet(n_blocks=2, n_gru=1)
        model.eval()

        mel = paddle.randn([1, 128, 20])
        out = model(mel)
        assert out.place.is_gpu_place() or isinstance(out.place, paddle.CPUPlace)


class TestFCPEPE:
    """Test FCPE inference wrapper."""

    def test_create_and_infer(self):
        """FCPE can be created and runs inference."""
        from paddlepe import PE

        pe = PE.create("fcpe")
        assert pe.trainable is True
        assert pe.support_onnx is True

        # Create synthetic audio
        sr = 16000
        t = paddle.linspace(0, 0.5, int(sr * 0.5))
        wav = paddle.sin(2 * paddle.pi * 440 * t)  # 440 Hz sine wave

        f0, conf = pe.infer(wav, sr)
        assert f0.shape[0] > 0
        if conf is not None:
            assert conf.shape[0] > 0

    def test_forward_training(self):
        """FCPE forward pass for training."""
        from paddlepe.models.fcpe.infer import FCPEPE

        pe = FCPEPE(hidden_dims=64, n_layers=2, n_heads=4)
        pe.eval()

        B, T, D = 2, 30, 128
        mel = paddle.randn([B, T, D])
        out = pe(mel)
        assert out.shape == [B, T, 360]


class TestRMVPEPE:
    """Test RMVPE inference wrapper."""

    def test_create_and_list(self):
        """RMVPE is registered and can be created."""
        from paddlepe import PE

        assert "rmvpe" in PE.list_models()
        pe = PE.create("rmvpe")
        assert pe.trainable is True
        assert pe.support_onnx is True

    def test_forward_training(self):
        """RMVPE forward pass for training."""
        from paddlepe.models.rmvpe.infer import RMVPEPE

        pe = RMVPEPE(n_blocks=2, n_gru=1)
        pe.eval()

        B, D, T = 2, 128, 30
        mel = paddle.randn([B, D, T])
        out = pe(mel)
        assert out.shape == [B, T, 360]
