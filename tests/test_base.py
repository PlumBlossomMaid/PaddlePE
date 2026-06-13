"""Tests for BasePE and PE facade."""

from __future__ import annotations

import paddle
import paddle.base.libpaddle as lp
import pytest

from paddlepe import PE
from paddlepe.models.base import BasePE


# Register a lightweight test model at module level so tests that need
# a real model instance (but not a checkpoint) work without ckpt files.
@PE.register("_test_model")
class _TestModel(BasePE):
    trainable = True

    def forward(self, x):
        return x

    def infer(self, wav, sr, **kwargs):
        return paddle.zeros([100]), paddle.ones([100])


class TestBasePE:
    """Test BasePE interface."""

    def test_base_pe_cannot_instantiate(self):
        """BasePE is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePE()  # type: ignore

    def test_model_registration(self):
        """Test that models are registered."""
        models = PE.list_models()
        assert "fcpe" in models
        assert "rmvpe" in models

    def test_list_models_returns_list_of_strings(self):
        """list_models() returns a list of strings."""
        models = PE.list_models()
        assert isinstance(models, list)
        assert all(isinstance(m, str) for m in models)

    def test_create_unknown_model_raises(self):
        """Creating an unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            PE.create("nonexistent_model")

    def test_custom_registration(self):
        """Custom model registration works."""

        @PE.register("test_model")
        class TestModel(BasePE):
            trainable = True

            def forward(self, x):
                return x

            def infer(self, wav, sr, **kwargs):
                return paddle.zeros([100]), paddle.ones([100])

        assert "test_model" in PE.list_models()
        model = PE.create("test_model")
        assert isinstance(model, BasePE)

    def test_pe_create_returns_basepe_instance(self):
        """PE.create returns a BasePE instance."""
        pe = PE.create("_test_model")
        assert isinstance(pe, BasePE)

    def test_device_property(self):
        """BasePE has a device property."""
        pe = PE.create("_test_model")
        device = pe.device
        # Should return a valid PaddlePaddle place
        assert isinstance(device, lp.Place)
