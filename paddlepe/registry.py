"""Registry for PE models. Kept separate to avoid circular imports."""

from __future__ import annotations

from pathlib import Path

import paddle


def _default_ckpt_path(name: str) -> str | None:
    """Resolve default checkpoint path for a model."""
    ckpt = Path(__file__).parent.parent / "ckpts" / f"{name}.pdparams"
    return str(ckpt) if ckpt.exists() else None


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

    def create(self, name: str, ckpt: str | None = None, **kwargs):
        """Create a model instance by name."""
        if name not in self._models:
            raise ValueError(
                f"Unknown model: {name}. Available: {list(self._models.keys())}"
            )
        model = self._models[name](**kwargs)
        resolved = ckpt or _default_ckpt_path(name)
        if resolved:
            state = paddle.load(resolved)
            model.set_state_dict(state)
        return model

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return sorted(self._models.keys())


# Global registry instance
registry = _Registry()
