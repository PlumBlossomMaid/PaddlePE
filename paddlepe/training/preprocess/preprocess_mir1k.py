"""Preprocess MIR-1K (RMVPE variant) → unified HDF5.

Auto-download from AI Studio if data is missing (public dataset,
no token required).

Input format:
  - <root>/train/*.wav    — 16kHz mono, right channel = vocals
  - <root>/train/*.pv     — text file, one MIDI note per line (0=unvoiced)

Output:
  HDF5 file with one group per sample, each containing
  ``waveform`` (right channel), ``f0`` (Hz), ``sr``, ``hop``, ``name``.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import h5py
import numpy as np
import soundfile as sf
from tqdm import tqdm

logger = logging.getLogger(__name__)

# AI Studio dataset (public, no token needed)
AI_STUDIO_REPO = "PlumBlossom/3yD0GV0L"


def _download(root: str) -> None:
    """Download MIR-1K from AI Studio if missing; extracts zip if needed."""
    root = Path(root)

    # Already has extracted data
    if root.exists() and list(root.rglob("*.wav")):
        return

    zip_path = root / "MIR-1K_4_RMVPE.zip"

    # Already has zip but not extracted
    if zip_path.exists():
        _extract_zip(zip_path, root)
        return

    # Download from AI Studio (public dataset)
    try:
        from aistudio_sdk.snapshot_download import snapshot_download

        logger.info("Downloading MIR-1K from AI Studio (%s) ...", AI_STUDIO_REPO)
        root.mkdir(parents=True, exist_ok=True)
        res = snapshot_download(
            repo_id=AI_STUDIO_REPO,
            revision="master",
            local_dir=str(root),
            repo_type="dataset",
        )
        logger.info("Download complete: %s", res)
    except ImportError:
        logger.error(
            "aistudio-sdk not installed. Run: pip install --upgrade aistudio-sdk"
        )
        raise
    except Exception as e:
        logger.error("Download failed: %s", e)
        raise

    # Extract if zip was downloaded
    if zip_path.exists():
        _extract_zip(zip_path, root)


def _extract_zip(zip_path: Path, root: Path) -> None:
    """Extract zip and clean up."""
    logger.info("Extracting %s ...", zip_path.name)
    with zipfile.ZipFile(str(zip_path)) as z:
        z.extractall(str(root))
    zip_path.unlink()
    logger.info("Extracted to %s", root)


def preprocess(
    root: str,
    output_path: str,
    sr: int = 16000,
    min_f0_hz: float = 40.0,
    overwrite: bool = False,
    auto_download: bool = True,
) -> None:
    """Preprocess MIR-1K RMVPE variant to unified HDF5.

    Args:
        root: path to ``Hybrid/train`` or ``Hybrid`` directory
        output_path: where to write the .h5 file
        sr: target sample rate
        min_f0_hz: frames with F0 below this are treated as unvoiced
        overwrite: overwrite existing HDF5
        auto_download: download from AI Studio if data is missing
    """
    root = Path(root)
    output_path = Path(output_path)

    # Auto-download if needed
    if auto_download:
        _download(root)

    # Locate train directory — supports both flat and nested layouts
    candidates = [
        root / "train",
        root / "Hybrid" / "train",
        root,
    ]
    train_dir = None
    for c in candidates:
        if c.exists() and list(c.glob("*.wav")):
            train_dir = c
            break
    if train_dir is None:
        raise FileNotFoundError(
            f"No .wav files found under {root}. "
            "Make sure the dataset is downloaded or place data manually."
        )

    wav_files = sorted(train_dir.glob("*.wav"))

    # Skip if HDF5 exists and overwrite is False
    if output_path.exists() and not overwrite:
        logger.info("HDF5 already exists: %s (use overwrite=True to redo)", output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = len(wav_files)
    n_skipped = 0

    with h5py.File(str(output_path), "w") as f:
        f.attrs["dataset"] = "MIR-1K (RMVPE variant)"
        f.attrs["n_samples"] = n_total
        f.attrs["sr"] = sr

        for idx, wav_path in enumerate(tqdm(wav_files, desc="Preprocessing MIR-1K")):
            pv_path = wav_path.with_suffix(".pv")
            if not pv_path.exists():
                logger.warning("Missing .pv for %s, skipping", wav_path.name)
                n_skipped += 1
                continue

            # Read audio (right channel = vocals)
            wav, file_sr = sf.read(str(wav_path))
            if wav.ndim == 2:
                audio = wav[:, 1]  # right channel
            else:
                audio = wav

            # Read F0 annotations (.pv = MIDI notes, text file)
            with open(str(pv_path)) as pf:
                lines = pf.readlines()
            midi = np.array([float(line.strip()) for line in lines], dtype=np.float32)

            # MIDI → Hz
            f0 = np.where(
                midi > 0,
                55.0 * (2.0 ** ((midi - 33.0) / 12.0)),
                0.0,
            )

            # Filter low values
            f0[f0 < min_f0_hz] = 0.0

            # Resample audio if needed
            if file_sr != sr:
                import scipy.signal  # lazy import

                ratio = sr / file_sr
                new_len = int(len(audio) * ratio)
                audio = scipy.signal.resample(audio, new_len)

            # Create HDF5 group
            grp = f.create_group(f"sample_{idx:05d}")
            grp.create_dataset(
                "waveform",
                data=audio.astype(np.float32),
                compression="gzip",
                compression_opts=3,
            )
            grp.create_dataset("f0", data=f0, compression="gzip", compression_opts=3)
            grp.create_dataset("sr", data=sr)
            grp.create_dataset("hop", data=160)  # MIR-1K: 10ms @ 16kHz
            grp.attrs["name"] = wav_path.stem

    logger.info(
        "Done: %d samples → %s (%d skipped)",
        n_total - n_skipped,
        output_path,
        n_skipped,
    )
