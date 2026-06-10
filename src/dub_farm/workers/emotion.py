from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from dub_farm.config import AppConfig
from dub_farm.models.schemas import EmotionSegment, TranscriptSegment
from dub_farm.utils.io import load_json, save_json

logger = logging.getLogger(__name__)

EMOTIONS = ("fear", "anger", "joy", "sadness", "panic", "sarcasm", "neutral")


def run_emotion_analysis(
    speech_path: Path,
    work_dir: Path,
    config: AppConfig,
    segments: list[TranscriptSegment],
) -> list[EmotionSegment]:
    """Module 4: Analyze emotional tone of each segment."""
    emotions_path = work_dir / "emotions.json"
    if emotions_path.exists():
        return load_json(emotions_path, EmotionSegment)

    if not config.emotion.enabled:
        logger.info("Emotion analysis disabled — using neutral defaults")
        results = [
            EmotionSegment(segment_id=seg.id, emotion="neutral", intensity=0.5)
            for seg in segments
        ]
        save_json(emotions_path, results)
        return results

    if config.emotion.backend == "heuristic":
        results = _heuristic_emotion(speech_path, segments)
    else:
        results = [
            EmotionSegment(segment_id=seg.id, emotion="neutral", intensity=0.5)
            for seg in segments
        ]

    save_json(emotions_path, results)
    return results


def _heuristic_emotion(
    speech_path: Path, segments: list[TranscriptSegment]
) -> list[EmotionSegment]:
    """Lightweight prosody-based heuristic until a dedicated model is plugged in."""
    audio, sample_rate = sf.read(str(speech_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    results: list[EmotionSegment] = []
    for seg in segments:
        start_idx = max(0, int(seg.start * sample_rate))
        end_idx = min(len(audio), int(seg.end * sample_rate))
        chunk = audio[start_idx:end_idx]

        if len(chunk) == 0:
            results.append(EmotionSegment(segment_id=seg.id))
            continue

        rms = float(np.sqrt(np.mean(chunk**2)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(chunk)))) / 2)
        duration = max(seg.end - seg.start, 0.01)
        word_count = max(len(seg.text.split()), 1)
        speech_rate = word_count / duration

        emotion = "neutral"
        intensity = 0.5

        if rms > 0.15 and speech_rate > 3.5:
            emotion, intensity = "anger", min(1.0, rms * 3)
        elif rms > 0.12 and zcr > 0.1:
            emotion, intensity = "fear", min(1.0, rms * 2.5)
        elif speech_rate > 4.0:
            emotion, intensity = "panic", min(1.0, speech_rate / 6)
        elif rms < 0.03:
            emotion, intensity = "sadness", 0.6

        results.append(
            EmotionSegment(
                segment_id=seg.id,
                emotion=emotion,
                intensity=round(intensity, 2),
                speech_rate=round(speech_rate, 2),
                volume=round(min(1.0, rms * 5), 2),
            )
        )

    return results
