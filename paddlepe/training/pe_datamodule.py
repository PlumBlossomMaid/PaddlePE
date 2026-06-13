"""PEDataModule — ocean.DataModule for pitch estimation training.

Manages multi-dataset loading, automatic preprocessing, and
model-specific collation.

Usage::

    from paddlepe.training import PEDataModule
    from paddlepe.training.collators import FCPECollator

    dm = PEDataModule(
        datasets={
            "mir1k": "data/mir1k",
            "mdb": {"path": "data/mdb", "weight": 2.0},
        },
        collator=FCPECollator(),
        sampler="balanced",
        batch_size=16,
        val_split=0.1,
    )

    trainer.fit(model, datamodule=dm)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import paddle
from paddle.io import ConcatDataset, DataLoader

from paddlepe.training.hdf5_dataset import HDF5Dataset

logger = logging.getLogger(__name__)

DatasetConfig = Union[
    str,  # just a path
    dict[str, Any],  # {"path": ..., "weight": ...}
]


def _resolve_config(
    datasets: dict[str, DatasetConfig],
) -> tuple[list[str], list[float]]:
    """Normalize dataset config to (paths, weights)."""
    paths, weights = [], []
    for name, cfg in datasets.items():
        if isinstance(cfg, str):
            paths.append(cfg)
            weights.append(1.0)
        elif isinstance(cfg, dict):
            paths.append(cfg["path"])
            weights.append(cfg.get("weight", 1.0))
        else:
            raise TypeError(f"Unexpected dataset config for '{name}': {cfg}")
    return paths, weights


class PEDataModule:
    """DataModule for pitch estimation model training.

    Args:
        datasets: dict mapping dataset name → path or config dict.
            Config dict may contain ``path``, ``weight``, ``min_f0_hz``.
        collator: model-specific collator instance.
        sampler: sampling strategy (``'proportional'``, ``'balanced'``,
            ``'round_robin'``).
        batch_size: training batch size.
        val_split: fraction of data to hold out for validation (0.0~1.0).
        num_workers: DataLoader workers.
        preprocess_root: where to find/store preprocessed HDF5 files.
            Default: ``<dataset_path>/../h5/``.
        seed: random seed for reproducibility.
    """

    def __init__(
        self,
        datasets: dict[str, DatasetConfig],
        collator: Any,
        sampler: str = "proportional",
        batch_size: int = 16,
        val_split: float = 0.0,
        num_workers: int = 0,
        preprocess_root: str | None = None,
        seed: int = 42,
    ):
        self.datasets = datasets
        self.collator = collator
        self.sampler = sampler
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.preprocess_root = preprocess_root
        self.seed = seed

        self._train_ds: ConcatDataset | None = None
        self._val_ds: ConcatDataset | None = None

    # ------------------------------------------------------------------
    # Data lifecycle
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        """Ensure preprocessed HDF5 files exist.

        Called once before training (on rank 0 in distributed mode).
        """
        paths, _ = _resolve_config(self.datasets)

        for ds_path in paths:
            h5_path = self._h5_path(ds_path)
            if h5_path.exists():
                logger.info("Found existing HDF5: %s", h5_path)
                continue

            logger.info("Preprocessing %s → %s ...", ds_path, h5_path)
            self._preprocess(ds_path, h5_path)

    def setup(self, stage: str = "fit") -> None:
        """Load datasets and create train/val splits."""
        paddle.seed(self.seed)

        paths, weights = _resolve_config(self.datasets)
        all_datasets: list[HDF5Dataset] = []

        for ds_path, w in zip(paths, weights):
            h5_path = self._h5_path(ds_path)
            if not h5_path.exists():
                raise FileNotFoundError(
                    f"HDF5 not found at {h5_path}. "
                    "Call prepare_data() first or run the preprocess script."
                )
            min_f0 = 40.0  # default
            # Check if per-dataset config has min_f0_hz override
            cfg = self.datasets.get(
                [
                    k
                    for k, v in self.datasets.items()
                    if isinstance(v, dict)
                    and v.get("path") == ds_path
                    or v == ds_path
                ][0]
            )
            if isinstance(cfg, dict) and "min_f0_hz" in cfg:
                min_f0 = cfg["min_f0_hz"]

            ds = HDF5Dataset(str(h5_path), min_f0_hz=min_f0)
            all_datasets.append(ds)

        if self.val_split > 0.0:
            # Hold out last val_split fraction from each dataset
            train_subs, val_subs = [], []
            for ds in all_datasets:
                n = len(ds)
                n_val = max(1, int(n * self.val_split))
                train_subs.append(paddle.io.Subset(ds, list(range(n - n_val))))
                val_subs.append(paddle.io.Subset(ds, list(range(n - n_val, n))))
            self._train_ds = (
                ConcatDataset(train_subs)
                if len(train_subs) > 1
                else train_subs[0]
            )
            self._val_ds = (
                ConcatDataset(val_subs) if len(val_subs) > 1 else val_subs[0]
            )
        else:
            self._train_ds = (
                ConcatDataset(all_datasets)
                if len(all_datasets) > 1
                else all_datasets[0]
            )
            self._val_ds = None

        logger.info(
            "Train: %d samples | Val: %s",
            len(self._train_ds),
            len(self._val_ds) if self._val_ds else "N/A",
        )

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader with model-specific collator."""
        assert self._train_ds is not None, "Call setup() first"

        # Paddle DataLoader doesn't support raw Sampler, use shuffle=True
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Return validation DataLoader, or None."""
        if self._val_ds is None:
            return None
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.num_workers,
        )

    def teardown(self, stage: str = "fit") -> None:
        """Cleanup after training."""
        self._train_ds = None
        self._val_ds = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _h5_path(self, dataset_path: str) -> Path:
        """Resolve the expected HDF5 path for a dataset."""
        name = Path(dataset_path).name
        if self.preprocess_root:
            return Path(self.preprocess_root) / f"{name}.h5"
        return Path(dataset_path).parent / "h5" / f"{name}.h5"

    def _preprocess(self, dataset_path: str, h5_path: Path) -> None:
        """Auto-discover and run the appropriate preprocess script."""
        # Try to import preprocess module by dataset name
        ds_name = Path(dataset_path).name.lower()
        try:
            mod = __import__(
                f"paddlepe.training.preprocess.preprocess_{ds_name}",
                fromlist=["preprocess"],
            )
            mod.preprocess(dataset_path, str(h5_path))
        except ImportError:
            raise RuntimeError(
                f"No preprocess script found for dataset '{ds_name}'. "
                f"Expected paddlepe/training/preprocess/preprocess_{ds_name}.py. "
                f"Write one or manually preprocess the data."
            )
