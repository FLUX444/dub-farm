from __future__ import annotations

import logging
from pathlib import Path

from dub_farm.config import AppConfig
from dub_farm.utils.ffmpeg import run_ffmpeg

logger = logging.getLogger(__name__)


def run_extract_audio(video_path: Path, work_dir: Path, config: AppConfig) -> Path:
    """Module 1: Extract and normalize audio from video."""
    output = work_dir / "audio.wav"
    if output.exists():
        logger.info("audio.wav already exists, skipping extraction")
        return output

    logger.info("Extracting audio from %s", video_path.name)
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(config.audio.sample_rate),
            "-ac",
            str(config.audio.channels),
            str(output),
        ]
    )
    return output
