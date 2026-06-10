from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from dub_farm.config import AppConfig
from dub_farm.utils.ffmpeg import find_ffmpeg, run_ffmpeg
from dub_farm.utils.gpu import clear_gpu_memory

logger = logging.getLogger(__name__)


def run_separate_audio(audio_path: Path, work_dir: Path, config: AppConfig) -> tuple[Path, Path]:
    """Module 2: Separate speech from background."""
    speech_path = work_dir / "speech.wav"
    background_path = work_dir / "background.wav"

    if speech_path.exists() and background_path.exists():
        logger.info("Separated audio already exists, skipping")
        return speech_path, background_path

    if config.separation.backend == "skip":
        logger.warning("Separation skipped — copying original audio as speech track")
        shutil.copy2(audio_path, speech_path)
        _create_silent_background(audio_path, background_path, config)
        return speech_path, background_path

    logger.info("Separating audio with Demucs (%s)", config.separation.model)
    try:
        import demucs  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Demucs is required for audio separation. "
            "Install with: pip install dub-farm[ml]"
        ) from exc

    demucs_out = work_dir / "demucs_out"
    demucs_out.mkdir(parents=True, exist_ok=True)

    import subprocess
    import sys

    env = os.environ.copy()
    ffmpeg_dir = str(Path(find_ffmpeg()).parent)
    env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "demucs",
            "-n",
            config.separation.model,
            "--two-stems",
            config.separation.two_stems,
            "-o",
            str(demucs_out),
            str(audio_path),
        ],
        check=True,
        env=env,
    )

    stem_dir = demucs_out / config.separation.model / audio_path.stem
    vocals = stem_dir / "vocals.wav"
    no_vocals = stem_dir / "no_vocals.wav"

    if not vocals.exists() or not no_vocals.exists():
        raise FileNotFoundError(f"Demucs output not found in {stem_dir}")

    shutil.copy2(vocals, speech_path)
    shutil.copy2(no_vocals, background_path)

    if config.pipeline.unload_models:
        clear_gpu_memory()

    return speech_path, background_path


def _create_silent_background(
    audio_path: Path, background_path: Path, config: AppConfig
) -> None:
    run_ffmpeg(
        [
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={config.audio.sample_rate}:cl=stereo",
            "-t",
            _probe_duration(audio_path),
            "-acodec",
            "pcm_s16le",
            str(background_path),
        ]
    )


def _probe_duration(audio_path: Path) -> str:
    from dub_farm.utils.ffmpeg import get_duration

    return f"{get_duration(audio_path):.3f}"
