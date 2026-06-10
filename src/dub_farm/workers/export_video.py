from __future__ import annotations

import logging
from pathlib import Path

from dub_farm.config import AppConfig
from dub_farm.utils.ffmpeg import run_ffmpeg

logger = logging.getLogger(__name__)


def run_export(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    config: AppConfig,
) -> Path:
    """Module 11: Mux dubbed audio with original video."""
    if output_path.exists():
        logger.info("%s already exists, skipping export", output_path.name)
        return output_path

    logger.info("Exporting dubbed video to %s", output_path.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_ffmpeg(
        [
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            config.export.video_codec,
            "-crf",
            str(config.export.crf),
            "-preset",
            config.export.preset,
            "-c:a",
            config.export.audio_codec,
            "-shortest",
            str(output_path),
        ]
    )
    return output_path
