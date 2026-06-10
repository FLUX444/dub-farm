from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from dub_farm.config import AppConfig
from dub_farm.models.schemas import TranscriptSegment, TranslationSegment
from dub_farm.utils.gpu import clear_gpu_memory
from dub_farm.workers.qwen_tts_engine import (
    resample_audio,
    synthesize_segment,
    unload_qwen_model,
)

logger = logging.getLogger(__name__)


def run_tts(
    work_dir: Path,
    config: AppConfig,
    translations: list[TranslationSegment],
    sample_rate: int,
) -> Path:
    """Module 9: Generate dubbed speech with Qwen3-TTS."""
    output = work_dir / "generated_speech.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        logger.info("generated_speech.wav already exists, skipping TTS")
        return output

    if config.tts.backend != "qwen3-tts":
        raise ValueError(
            f"Unsupported TTS backend: {config.tts.backend!r}. "
            "Only qwen3-tts is supported."
        )

    if not translations:
        logger.warning("No translations — creating silent speech track")
        _create_silent_wav(output, duration_sec=1.0, sample_rate=sample_rate)
        return output

    speech_path = work_dir / "speech.wav"
    source_segments = _load_source_segments(work_dir)

    total_duration = max(seg.end for seg in translations) + 1.0
    canvas = np.zeros(int(total_duration * sample_rate), dtype=np.float32)

    try:
        for i, seg in enumerate(translations, start=1):
            if not seg.translated_text.strip():
                continue
            logger.info(
                "TTS [%d/%d] %s: %.1fs",
                i,
                len(translations),
                seg.speaker,
                seg.end - seg.start,
            )
            clip, clip_sr = synthesize_segment(
                seg,
                config,
                speech_path=speech_path if speech_path.exists() else None,
                source_segments=source_segments,
            )
            clip = resample_audio(clip, clip_sr, sample_rate)
            clip = _fit_to_duration(clip, seg.end - seg.start, sample_rate)

            start_idx = int(seg.start * sample_rate)
            end_idx = min(start_idx + len(clip), len(canvas))
            canvas[start_idx:end_idx] += clip[: end_idx - start_idx]
    finally:
        unload_qwen_model()
        if config.pipeline.unload_models:
            clear_gpu_memory()

    peak = np.max(np.abs(canvas))
    if peak > 1.0:
        canvas = canvas / peak

    sf.write(str(output), canvas, sample_rate)
    return output


def _load_source_segments(work_dir: Path) -> list[TranscriptSegment]:
    transcript_path = work_dir / "transcript.json"
    if not transcript_path.exists():
        return []
    from dub_farm.utils.io import load_json

    return load_json(transcript_path, TranscriptSegment)


def _fit_to_duration(
    audio: np.ndarray, target_duration: float, sample_rate: int
) -> np.ndarray:
    target_samples = max(1, int(target_duration * sample_rate))
    if len(audio) == target_samples:
        return audio
    if len(audio) > target_samples:
        return audio[:target_samples]
    pad = np.zeros(target_samples - len(audio), dtype=np.float32)
    return np.concatenate([audio, pad])


def _create_silent_wav(path: Path, duration_sec: float, sample_rate: int) -> None:
    silence = np.zeros(int(duration_sec * sample_rate), dtype=np.float32)
    sf.write(str(path), silence, sample_rate)
