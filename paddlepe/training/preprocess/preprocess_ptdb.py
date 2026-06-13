"""Preprocess PTDB-TUG → unified HDF5.

Input format (from SPEECH_DATA_ZIPPED.zip, 3.9G):
  SPEECH DATA/{FEMALE,MALE}/LAR/{F01..M10}/lar_{spk}_{utt}.wav   — 48kHz 16-bit PCM
  SPEECH DATA/{FEMALE,MALE}/REF/{F01..M10}/ref_{spk}_{utt}.f0   — EGG-derived F0,
                                                                  4 columns per line:
                                                                  col 1-2: other signals
                                                                  col 3: F0 (Hz)
                                                                  col 4: confidence
                                                                  0 = unvoiced.

  Only LAR/ (headset microphone) has clean speech for training.
  Each LAR/*.wav has a corresponding REF/*.f0 with EGG-derived pitch.

  All files are paired by speaker + utterance name
  (e.g., lar_F01_sa1.wav ↔ ref_F01_sa1.f0).

Output:
  HDF5 file with one group per sample, each containing
  ``waveform`` (16kHz mono, LAR channel), ``f0`` (Hz, 10ms hop),
  ``sr``, ``hop``, ``name``.
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

TARGET_SR = 16000
TARGET_HOP = 160  # 10 ms @ 16 kHz


def _find_speech_dir(root: Path) -> Path:
    """Locate the extracted ``SPEECH DATA`` directory (note the space)."""
    # The zip extracts to a directory named "SPEECH DATA" (with space)
    candidates = [
        root / "SPEECH DATA",
        root / "SPEECH_DATA",
        root,
    ]
    for c in candidates:
        if c.exists() and any(c.iterdir()):
            return c
    raise FileNotFoundError(
        f"Cannot find extracted SPEECH DATA directory under {root}. "
        "Try extracting SPEECH_DATA_ZIPPED.zip manually."
    )


def _collect_pairs(speech_dir: Path) -> list[tuple[Path, Path]]:
    """Collect (wav_path, f0_path) pairs from LAR/ and REF/ subdirs.

    Matches lar_{spk}_{utt}.wav ↔ ref_{spk}_{utt}.f0 by speaker and utterance.
    """
    pairs = []

    # Find all LAR .wav files
    lar_wavs = list(speech_dir.rglob("**/LAR/**/*.wav"))

    for wav_path in lar_wavs:
        stem = wav_path.stem  # e.g. "lar_F01_sa1"
        # Build expected .f0 path: replace "lar_" with "ref_", REF not LAR
        f0_stem = stem.replace("lar_", "ref_", 1)
        # Navigate: go up to speaker dir, then up to LAR/, sibling REF/, then speaker dir  # noqa: E501
        # Path: .../LAR/F01/lar_F01_sa1.wav → .../REF/F01/ref_F01_sa1.f0
        speaker_dir = wav_path.parent  # .../LAR/F01
        ref_dir = speaker_dir.parent.parent / "REF" / speaker_dir.name
        f0_path = ref_dir / f"{f0_stem}.f0"

        if f0_path.exists():
            pairs.append((wav_path, f0_path))
        else:
            logger.warning("Missing .f0 for %s (expected %s)", stem, f0_path)

    logger.info(
        "Found %d wav-f0 pairs from %d LAR wav files", len(pairs), len(lar_wavs)
    )
    return pairs


def preprocess(
    root: str,
    output_path: str,
    sr: int = TARGET_SR,
    min_f0_hz: float = 40.0,
    overwrite: bool = False,
    auto_extract: bool = True,
) -> None:
    """Preprocess PTDB-TUG to unified HDF5.

    Args:
        root: directory containing SPEECH_DATA_ZIPPED.zip or extracted SPEECH DATA/
        output_path: where to write the .h5 file
        sr: target sample rate (16000)
        min_f0_hz: frames with F0 below this are treated as unvoiced
        overwrite: overwrite existing HDF5
        auto_extract: extract zip if SPEECH DATA directory doesn't exist
    """
    root = Path(root)
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        logger.info("HDF5 already exists: %s (use overwrite=True to redo)", output_path)
        return

    # Extract zip if needed
    zip_path = root / "SPEECH_DATA_ZIPPED.zip"
    speech_dir_candidates = [
        root / "SPEECH DATA",
        root / "SPEECH_DATA",
        root,
    ]
    is_extracted = any(c.exists() and any(c.iterdir()) for c in speech_dir_candidates)

    if not is_extracted and zip_path.exists() and auto_extract:
        logger.info("Extracting %s ...", zip_path.name)
        with zipfile.ZipFile(str(zip_path)) as zf:
            zf.extractall(str(root))
        logger.info("Extraction complete.")

    speech_dir = _find_speech_dir(root)

    # Collect (wav, f0) pairs
    pairs = _collect_pairs(speech_dir)
    if not pairs:
        raise FileNotFoundError(f"No valid wav-f0 pairs found under {speech_dir}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_skipped = 0

    with h5py.File(str(output_path), "w") as f:
        f.attrs["dataset"] = "PTDB-TUG"
        f.attrs["n_samples"] = len(pairs)
        f.attrs["sr"] = sr

        for idx, (wav_path, f0_path) in enumerate(
            tqdm(pairs, desc="Preprocessing PTDB-TUG")
        ):
            try:
                # ── Read audio ──
                audio, file_sr = sf.read(str(wav_path))
                if audio.ndim > 1:
                    audio = audio.mean(axis=-1)

                # ── Read F0 from .f0 file (4 columns, col 3 = F0 in Hz) ──
                f0_raw = np.loadtxt(str(f0_path), dtype=np.float32)
                if f0_raw.ndim == 2 and f0_raw.shape[1] >= 3:
                    f0_hz = f0_raw[:, 2]  # column 3 (0-indexed: 2)
                elif f0_raw.ndim == 1:
                    f0_hz = f0_raw
                else:
                    logger.warning("Unexpected F0 format in %s, skipping", f0_path)
                    n_skipped += 1
                    continue

                # ── Resample audio: 48 kHz → 16 kHz ──
                if file_sr != sr:
                    import scipy.signal

                    ratio = sr / file_sr
                    new_len = int(len(audio) * ratio)
                    audio = scipy.signal.resample(audio, new_len)

                # ── Downsample F0: 1 ms → 10 ms (take every 10th value) ──
                target_frames = len(audio) // TARGET_HOP
                f0 = np.zeros(target_frames, dtype=np.float32)
                for t in range(target_frames):
                    src_idx = t * 10
                    if src_idx < len(f0_hz):
                        val = f0_hz[src_idx]
                        f0[t] = val if val >= min_f0_hz else 0.0

                # ── Write HDF5 group ──
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
                grp.attrs["name"] = wav_path.stem

            except Exception as e:
                logger.warning("Error processing %s: %s, skipping", wav_path, e)
                n_skipped += 1
                continue

    logger.info(
        "Done: %d samples → %s (%d skipped)",
        len(pairs) - n_skipped,
        output_path,
        n_skipped,
    )
