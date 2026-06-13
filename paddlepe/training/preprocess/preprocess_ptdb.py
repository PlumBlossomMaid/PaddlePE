"""Preprocess PTDB-TUG → unified HDF5.

Auto-download 3.9G zip from Graz University if data is missing.

Input format (extracted zip):
  SPEECH_DATA/{F01..F10,M01..M10}/{utterance}.wav  — 48kHz 16-bit PCM
  SPEECH_DATA/{F01..F10,M01..M10}/{utterance}.f0   — F0 (Hz) from EGG,
                                                       one value per line,
                                                       1 ms intervals.
                                                       0 = unvoiced.

Output:
  HDF5 file with one group per sample, each containing
  ``waveform`` (16kHz mono), ``f0`` (Hz, 10ms hop),
  ``sr``, ``hop``, ``name``.
"""

from __future__ import annotations

import logging
import urllib.request
import zipfile
from pathlib import Path

import h5py
import numpy as np
import soundfile as sf
from tqdm import tqdm

logger = logging.getLogger(__name__)

PTDB_ZIP_URL = (
    "https://www2.spsc.tugraz.at/databases/PTDB-TUG/SPEECH_DATA_ZIPPED.zip"
)
TARGET_SR = 16000  # unified sample rate
TARGET_HOP = 160  # 10 ms @ 16 kHz


def _download_and_extract(root: str) -> Path:
    """Download SPEECH_DATA_ZIPPED.zip and extract it.

    Returns:
        Path to the extracted SPEECH_DATA directory.
    """
    root = Path(root)
    zip_path = root / "SPEECH_DATA_ZIPPED.zip"
    extract_dir = root / "SPEECH_DATA"

    # Already extracted
    if extract_dir.exists() and any(extract_dir.iterdir()):
        logger.info("Found extracted PTDB-TUG at %s", extract_dir)
        return extract_dir

    # Download if missing
    if not zip_path.exists():
        logger.info("Downloading PTDB-TUG (3.9G) ...")
        root.mkdir(parents=True, exist_ok=True)

        def _reporthook(count, block_size, total_size):
            downloaded = count * block_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            if count % 50 == 0:
                logger.info(
                    "  %.0f / %.0f MB (%.0f%%)",
                    downloaded,
                    total_mb,
                    100 * downloaded / total_mb if total_mb > 0 else 0,
                )

        urllib.request.urlretrieve(
            PTDB_ZIP_URL,
            str(zip_path),
            reporthook=_reporthook,
        )
        logger.info(
            "Download complete: %s (%.1f GB)",
            zip_path,
            zip_path.stat().st_size / (1024**3),
        )

    # Extract
    logger.info("Extracting %s ...", zip_path.name)
    with zipfile.ZipFile(str(zip_path)) as zf:
        zf.extractall(str(root))
    logger.info("Extracted to %s", extract_dir)

    return extract_dir


def preprocess(
    root: str,
    output_path: str,
    sr: int = TARGET_SR,
    min_f0_hz: float = 40.0,
    overwrite: bool = False,
    auto_download: bool = True,
) -> None:
    """Preprocess PTDB-TUG to unified HDF5.

    Args:
        root: directory for download/extraction (or existing SPEECH_DATA dir)
        output_path: where to write the .h5 file
        sr: target sample rate (16000)
        min_f0_hz: frames with F0 below this are treated as unvoiced
        overwrite: overwrite existing HDF5
        auto_download: download from Graz University if data is missing
    """
    root = Path(root)
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        logger.info(
            "HDF5 already exists: %s (use overwrite=True to redo)", output_path
        )
        return

    # Download / locate data
    if auto_download:
        speech_dir = _download_and_extract(root)
    else:
        speech_dir = root / "SPEECH_DATA"
        if not speech_dir.exists():
            raise FileNotFoundError(
                f"SPEECH_DATA not found at {speech_dir}. "
                "Set auto_download=True or place data manually."
            )

    # Discover all .wav files
    wav_files = sorted(speech_dir.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(
            f"No .wav files found under {speech_dir}. "
            "Is the dataset extracted correctly?"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = len(wav_files)
    n_skipped = 0

    with h5py.File(str(output_path), "w") as f:
        f.attrs["dataset"] = "PTDB-TUG"
        f.attrs["n_samples"] = n_total
        f.attrs["sr"] = sr

        for idx, wav_path in enumerate(
            tqdm(wav_files, desc="Preprocessing PTDB-TUG")
        ):
            f0_path = wav_path.with_suffix(".f0")
            if not f0_path.exists():
                logger.warning("Missing .f0 for %s, skipping", wav_path.name)
                n_skipped += 1
                continue

            # Read audio
            audio, file_sr = sf.read(str(wav_path))
            if audio.ndim > 1:
                audio = audio.mean(axis=-1)  # mono

            # Read F0 from .f0 file (Hz, one per line, 1 ms intervals)
            f0_raw = np.loadtxt(str(f0_path), dtype=np.float32)

            # Resample audio: 48 kHz → 16 kHz
            if file_sr != sr:
                import scipy.signal  # lazy import

                ratio = sr / file_sr
                new_len = int(len(audio) * ratio)
                audio = scipy.signal.resample(audio, new_len)

            # Downsample F0: 1 ms → 10 ms (take every 10th value)
            # .f0 has 1 value per ms at original 48 kHz
            target_frames = len(audio) // TARGET_HOP
            f0 = np.zeros(target_frames, dtype=np.float32)
            for t in range(target_frames):
                src_idx = t * 10
                if src_idx < len(f0_raw):
                    val = f0_raw[src_idx]
                    f0[t] = val if val >= min_f0_hz else 0.0

            # Create HDF5 group
            grp = f.create_group(f"sample_{idx:05d}")
            grp.create_dataset(
                "waveform",
                data=audio.astype(np.float32),
                compression="gzip",
                compression_opts=3,
            )
            grp.create_dataset(
                "f0",
                data=f0.astype(np.float32),
                compression="gzip",
                compression_opts=3,
            )
            grp.create_dataset("sr", data=sr)
            grp.create_dataset("hop", data=TARGET_HOP)
            grp.attrs["name"] = f"{wav_path.parent.name}_{wav_path.stem}"

    logger.info(
        "Done: %d samples → %s (%d skipped)",
        n_total - n_skipped,
        output_path,
        n_skipped,
    )
