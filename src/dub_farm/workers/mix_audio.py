from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from dub_farm.config import AppConfig

logger = logging.getLogger(__name__)


def run_mix(
    background_path: Path,
    speech_path: Path,
    work_dir: Path,
    config: AppConfig,
) -> Path:
    """Module 10: Mix generated speech with background track."""
    output = work_dir / "final_mix.wav"
    if output.exists():
        logger.info("final_mix.wav already exists, skipping mix")
        return output

    logger.info("Mixing speech and background")
    bg, bg_sr = sf.read(str(background_path))
    speech, speech_sr = sf.read(str(speech_path))

    if bg_sr != speech_sr:
        raise ValueError(f"Sample rate mismatch: bg={bg_sr}, speech={speech_sr}")

    bg = _to_mono(bg)
    speech = _to_mono(speech)

    max_len = max(len(bg), len(speech))
    bg = _pad(bg, max_len)
    speech = _pad(speech, max_len)

    bg_gain = 10 ** (config.mix.background_gain_db / 20)
    speech_gain = 10 ** (config.mix.speech_gain_db / 20)

    mixed = bg * bg_gain + speech * speech_gain

    if config.mix.normalize:
        mixed = _normalize(mixed, target_peak=0.95)

    sf.write(str(output), mixed, bg_sr)
    return output


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return audio.mean(axis=1).astype(np.float32)


def _pad(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio)))


def _normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return audio
    scaled = audio * (target_peak / peak)
    return np.clip(scaled, -1.0, 1.0)
