from __future__ import annotations

import logging
import os
import queue
import traceback
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from dub_farm.config import AppConfig
from dub_farm.models.schemas import (
    EmotionSegment,
    SceneAnalysis,
    SceneBoundary,
    TranscriptSegment,
    TranslationSegment,
)
from dub_farm.utils.gpu import clear_gpu_memory
from dub_farm.utils.io import load_json
from dub_farm.utils.proxy import apply_proxy_env

logger = logging.getLogger(__name__)


def run_step_isolated(
    step: str,
    *,
    video_path: Path,
    output_path: Path,
    work_dir: Path,
    config: AppConfig,
) -> None:
    """Run a pipeline step in a fresh Python process.

    This is intentionally process-based, not thread-based: CUDA contexts and
    large model allocators can survive Python object deletion in the parent.
    Exiting the child process is the reliable cleanup boundary on Windows/Linux.
    """
    ctx = get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_isolated_entrypoint,
        args=(
            step,
            str(video_path),
            str(output_path),
            str(work_dir),
            config.model_dump(mode="json"),
            result_queue,
        ),
        name=f"dub-farm-{step}",
    )
    process.start()
    process.join(config.pipeline.worker_timeout_sec)

    if process.is_alive():
        process.terminate()
        process.join(10)
        raise TimeoutError(
            f"Worker {step!r} exceeded timeout "
            f"{config.pipeline.worker_timeout_sec} seconds"
        )

    try:
        status, payload = result_queue.get_nowait()
    except queue.Empty:
        status, payload = ("ok", None) if process.exitcode == 0 else (
            "error",
            f"Worker {step!r} exited with code {process.exitcode}",
        )

    if status == "error":
        raise RuntimeError(payload)
    if process.exitcode not in (0, None):
        raise RuntimeError(f"Worker {step!r} exited with code {process.exitcode}")


def _isolated_entrypoint(
    step: str,
    video_path_raw: str,
    output_path_raw: str,
    work_dir_raw: str,
    config_data: dict[str, Any],
    result_queue,
) -> None:
    try:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        _prepend_ffmpeg_to_path()
        config = AppConfig.model_validate(config_data)
        apply_proxy_env(config)
        _run_step(
            step,
            video_path=Path(video_path_raw),
            output_path=Path(output_path_raw),
            work_dir=Path(work_dir_raw),
            config=config,
        )
        if config.pipeline.unload_models:
            clear_gpu_memory()
        result_queue.put(("ok", None))
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))
        raise


def _prepend_ffmpeg_to_path() -> None:
    try:
        from dub_farm.utils.ffmpeg import find_ffmpeg

        ffmpeg_dir = str(Path(find_ffmpeg()).parent)
    except Exception:
        return
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")


def _run_step(
    step: str,
    *,
    video_path: Path,
    output_path: Path,
    work_dir: Path,
    config: AppConfig,
) -> None:
    from dub_farm.workers import (
        run_export,
        run_extract_audio,
        run_mix,
        run_scene_analysis,
        run_separate_audio,
        run_transcribe,
        run_tts,
    )

    if step == "extract":
        run_extract_audio(video_path, work_dir, config)
    elif step == "separate":
        run_separate_audio(work_dir / "audio.wav", work_dir, config)
    elif step == "transcribe":
        run_transcribe(work_dir / "speech.wav", work_dir, config)
    elif step == "scene_analysis":
        scenes = load_json(work_dir / "scenes.json", SceneBoundary)
        run_scene_analysis(video_path, work_dir, config, scenes)
    elif step == "tts":
        translations = load_json(work_dir / "translation.json", TranslationSegment)
        run_tts(work_dir, config, translations, config.audio.sample_rate)
    elif step == "mix":
        run_mix(work_dir / "background.wav", work_dir / "generated_speech.wav", work_dir, config)
    elif step == "export":
        run_export(video_path, work_dir / "final_mix.wav", output_path, config)
    elif step == "emotion":
        from dub_farm.workers import run_emotion_analysis

        segments = load_json(work_dir / "transcript.json", TranscriptSegment)
        run_emotion_analysis(work_dir / "speech.wav", work_dir, config, segments)
    elif step == "translate":
        from dub_farm.workers import run_translate

        segments = load_json(work_dir / "transcript.json", TranscriptSegment)
        emotions = load_json(work_dir / "emotions.json", EmotionSegment)
        scene_analysis = load_json(work_dir / "scene_analysis.json", SceneAnalysis)
        run_translate(work_dir, config, segments, emotions, scene_analysis)
    elif step == "scene_detect":
        from dub_farm.workers import run_scene_detect

        run_scene_detect(video_path, work_dir, config)
    else:
        raise ValueError(f"Unknown isolated step: {step}")
