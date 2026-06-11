"""Registry for PE models. Kept separate to avoid circular imports."""

from __future__ import annotations

from typing import Optional

import paddle


class _Registry:
    """Internal registry for PE models."""

    def __init__(self):
        self._models: dict[str, type] = {}

    def register(self, name: str):
        """Decorator to register a model class."""

        def decorator(cls):
            self._models[name] = cls
            return cls

        return decorator

    def create(self, name: str, ckpt: Optional[str] = None, **kwargs):
        """Create a model instance by name."""
        if name not in self._models:
            raise ValueError(
                f"Unknown model: {name}. Available: {list(self._models.keys())}"
            )
        from paddlepe.models.base import BasePE
        model = self._models[name](**kwargs)
        if ckpt is not None:
            state = paddle.load(ckpt)
            model.set_state_dict(state)
        return model

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return sorted(self._models.keys())


# Global registry instance
registry = _Registry()
