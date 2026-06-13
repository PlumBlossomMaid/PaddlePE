"""Multi-dataset samplers for mixed training.

Strategies:
  - proportional (default): sample proportionally to dataset size
  - balanced: equal probability per dataset regardless of size
  - round_robin: alternate between datasets per-batch
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import paddle


def build_sampler(
    datasets: Sequence[paddle.io.Dataset],
    strategy: str = "proportional",
    weights: Sequence[float] | None = None,
    shuffle: bool = True,
) -> paddle.io.Sampler:
    """Build a sampler for concatenated multi-dataset training.

    Args:
        datasets: list of datasets to combine
        strategy: ``'proportional'``, ``'balanced'``, or ``'round_robin'``
        weights: per-dataset weights (used only for balanced mode)
        shuffle: whether to shuffle indices

    Returns:
        Sampler that yields global indices into ``ConcatDataset(datasets)``
    """
    lengths = [len(d) for d in datasets]
    offsets = np.cumsum([0] + lengths[:-1])
    total = sum(lengths)

    if strategy == "proportional":
        # Global shuffle over ConcatDataset — natural proportion
        indices = np.arange(total)
        if shuffle:
            np.random.shuffle(indices)
        return _ListSampler(indices.tolist())

    elif strategy == "balanced":
        # Equal probability per dataset
        if weights is not None:
            assert len(weights) == len(datasets), (
                f"weights ({len(weights)}) must match datasets ({len(datasets)})"
            )
            w = np.array(weights, dtype=np.float64)
        else:
            w = np.ones(len(datasets), dtype=np.float64)
        w /= w.sum()

        # Assign probabilities: each dataset gets w_i / len_i per sample
        per_sample_weights = np.concatenate(
            [np.full(n, w[i] / n) for i, n in enumerate(lengths)]
        )
        sampler = paddle.io.WeightedRandomSampler(
            weights=per_sample_weights.tolist(),
            num_samples=total,
            replacement=True,
        )
        return sampler

    elif strategy == "round_robin":
        return _RoundRobinSampler(lengths, offsets, shuffle=shuffle)

    else:
        raise ValueError(f"Unknown sampler strategy: {strategy}")


class _ListSampler(paddle.io.Sampler):
    """Simple sampler from an index list."""

    def __init__(self, indices: list[int]):
        super().__init__(data_source=None)
        self._indices = indices

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)


class _RoundRobinSampler(paddle.io.Sampler):
    """Round-robin: one batch from each dataset in turn."""

    def __init__(
        self,
        lengths: list[int],
        offsets: np.ndarray,
        shuffle: bool = True,
    ):
        super().__init__(data_source=None)
        self.lengths = lengths
        self.offsets = offsets
        self.shuffle = shuffle
        self._len = max(lengths) * len(lengths)

    def __iter__(self):
        # Per-dataset permutation
        per_ds = [
            np.random.permutation(n).tolist()
            if self.shuffle
            else list(range(n))
            for n in self.lengths
        ]
        # Interleave
        indices = []
        for i in range(max(self.lengths)):
            for d in range(len(self.lengths)):
                if i < self.lengths[d]:
                    indices.append(self.offsets[d] + per_ds[d][i])
        return iter(indices)

    def __len__(self):
        return self._len
