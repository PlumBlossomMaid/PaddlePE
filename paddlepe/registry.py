"""Registry for PE models. Kept separate to avoid circular imports."""

from __future__ import annotations

from pathlib import Path

import paddle

# AI Studio repo ID for model weight distribution
AI_STUDIO_REPO = "PlumBlossom/PaddlePE"


def _default_ckpt_path(name: str) -> str | None:
    """Resolve default checkpoint path for a model.

    Checks local ``ckpts/{name}.pdparams`` first. If not found,
    attempts to download from AI Studio via ``aistudio-sdk``.
    """
    ckpt = Path(__file__).parent.parent / "ckpts" / f"{name}.pdparams"
    if ckpt.exists():
        return str(ckpt)

    # Not found locally — try downloading from AI Studio
    print(f"  Model weight {ckpt.name} not found locally. Downloading from AI Studio...")
    _download_from_ai_studio(name, ckpt)
    return str(ckpt) if ckpt.exists() else None


def _download_from_ai_studio(name: str, dest: Path) -> None:
    """Download model weight from AI Studio using ``aistudio-sdk``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    filename = f"{name}.pdparams"
    try:
        from aistudio_sdk.snapshot_download import snapshot_download
    except ImportError:
        raise RuntimeError(
            "aistudio-sdk not installed. Install with: pip install aistudio-sdk\n"
            "Or place the weight file manually at: {dest}"
        )
    try:
        snapshot_download(
            repo_id=AI_STUDIO_REPO,
            revision="master",
            local_dir=str(dest.parent),
            allow_patterns=[filename],
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {filename} from AI Studio.\n"
            f"  Set AISTUDIO_ACCESS_TOKEN if the repo is private.\n"
            f"  Or place the file manually at: {dest}\n"
            f"  Error: {e}"
        )
    if not dest.exists():
        raise RuntimeError(f"Download reported success but {dest} not found.")


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
