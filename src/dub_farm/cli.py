from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.logging import RichHandler

from dub_farm.config import AppConfig
from dub_farm.download_models import download_all
from dub_farm.pipeline import DubPipeline
from dub_farm.utils.proxy import apply_proxy_env

app = typer.Typer(
    name="dub-farm",
    help="Automatic video dubbing with context-aware translation and Qwen3-TTS.",
    no_args_is_help=True,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


@app.command()
def run(
    input_video: Path = typer.Argument(..., help="Input video file (mp4, mkv, avi, mov, webm)"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output video path (default: <name>.dubbed.mp4)"
    ),
    target_lang: str = typer.Option("ru", "--target-lang", "-t", help="Target language code"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to YAML config file"
    ),
    work_dir: Optional[str] = typer.Option(None, "--work-dir", help="Working directory for artifacts"),
    translation_backend: Optional[str] = typer.Option(
        None, "--translation-backend", help="ollama | openai | passthrough"
    ),
    transcription_backend: Optional[str] = typer.Option(
        None,
        "--transcription-backend",
        help="faster-whisper | whisperx | vllm-openai | transformers | skip",
    ),
    vllm_url: Optional[str] = typer.Option(
        None,
        "--vllm-url",
        help="OpenAI-compatible vLLM URL, for example http://127.0.0.1:8000/v1",
    ),
    vllm_model: Optional[str] = typer.Option(
        None,
        "--vllm-model",
        help="Model name/path served by vLLM for audio transcriptions",
    ),
    tts_mode: Optional[str] = typer.Option(
        None,
        "--tts-mode",
        help="custom_voice | voice_design | voice_clone",
    ),
    tts_speaker: Optional[str] = typer.Option(
        None, "--tts-speaker", help="Qwen3-TTS speaker (Ryan, Vivian, ...)"
    ),
    from_step: Optional[str] = typer.Option(
        None, "--from-step", help="Resume from a specific pipeline step"
    ),
    no_resume: bool = typer.Option(False, "--no-resume", help="Do not skip completed steps"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Run the full dubbing pipeline on a video file."""
    _setup_logging(verbose)

    if not input_video.exists():
        raise typer.BadParameter(f"File not found: {input_video}")

    suffix = input_video.suffix.lower()
    if suffix not in {".mp4", ".mkv", ".avi", ".mov", ".webm"}:
        typer.echo(f"Warning: uncommon video format '{suffix}' — FFmpeg may still handle it.")

    app_config = AppConfig.load(config)
    overrides: dict = {"translation.target_lang": target_lang}
    if work_dir:
        overrides["pipeline.work_dir"] = work_dir
    if translation_backend:
        overrides["translation.backend"] = translation_backend
    if transcription_backend:
        overrides["transcription.backend"] = transcription_backend
    if vllm_url:
        overrides["transcription.vllm_url"] = vllm_url
    if vllm_model:
        overrides["transcription.vllm_model"] = vllm_model
    if tts_mode:
        overrides["tts.mode"] = tts_mode
    if tts_speaker:
        overrides["tts.speaker"] = tts_speaker
    if no_resume:
        overrides["pipeline.resume"] = False

    app_config = app_config.merge_cli_overrides(**overrides)
    apply_proxy_env(app_config)

    output_path = output or input_video.with_name(f"{input_video.stem}.dubbed.mp4")

    pipeline = DubPipeline(input_video, output_path, app_config)
    pipeline.run(from_step=from_step)


@app.command("download-models")
def download_models(
    weights_dir: Path = typer.Option(
        Path("weights"),
        "--weights-dir",
        "-d",
        help="Base directory (creates weights/tts, weights/whisper, ...)",
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="YAML config file (for proxy settings)"
    ),
    tts_only: bool = typer.Option(False, "--tts-only", help="Download only Qwen3-TTS models"),
    with_vlm: bool = typer.Option(False, "--with-vlm", help="Also download Qwen2-VL for scenes"),
    no_whisper: bool = typer.Option(False, "--no-whisper", help="Skip faster-whisper model"),
) -> None:
    """Download Hugging Face models into weights/tts, weights/whisper, etc."""
    _setup_logging(verbose=False)
    app_config = AppConfig.load(config)
    apply_proxy_env(app_config)
    download_all(
        weights_dir,
        include_vlm=with_vlm,
        include_whisper=not no_whisper,
        tts_only=tts_only,
        config=app_config,
    )


@app.command()
def steps() -> None:
    """List available pipeline steps."""
    from dub_farm.models.schemas import PipelineStep

    for step in PipelineStep:
        typer.echo(step.value)


@app.command()
def speakers() -> None:
    """List Qwen3-TTS CustomVoice speakers."""
    rows = [
        ("Vivian", "Bright young female", "Chinese"),
        ("Serena", "Warm gentle female", "Chinese"),
        ("Uncle_Fu", "Low mellow male", "Chinese"),
        ("Dylan", "Beijing male", "Chinese (dialect)"),
        ("Eric", "Chengdu male", "Chinese (dialect)"),
        ("Ryan", "Dynamic male", "English"),
        ("Aiden", "Clear American male", "English"),
        ("Ono_Anna", "Playful female", "Japanese"),
        ("Sohee", "Warm emotional female", "Korean"),
    ]
    typer.echo("Speaker          Description              Native language")
    typer.echo("-" * 60)
    for name, desc, lang in rows:
        typer.echo(f"{name:<16} {desc:<24} {lang}")
    typer.echo(
        "\nAll speakers can speak Russian and other supported languages.\n"
        "For dubbing with original voices use: --tts-mode voice_clone"
    )


if __name__ == "__main__":
    app()
