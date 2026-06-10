from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    work_dir: str = ".dub-farm"
    resume: bool = True
    unload_models: bool = True
    isolate_heavy_workers: bool = True
    isolated_steps: list[str] = Field(
        default_factory=lambda: ["separate", "transcribe", "scene_analysis", "tts"]
    )
    worker_timeout_sec: int | None = None


class AudioConfig(BaseModel):
    sample_rate: int = 44100
    channels: int = 2


class SeparationConfig(BaseModel):
    backend: str = "demucs"
    model: str = "htdemucs"
    two_stems: str = "vocals"


class TranscriptionConfig(BaseModel):
    backend: str = "faster-whisper"
    model: str = "large-v3"
    fallback_backend: str = "transformers"
    vllm_url: str = "http://127.0.0.1:8000/v1"
    vllm_api_key_env: str | None = None
    vllm_model: str = "/models/whisper-v3"
    language: str | None = None
    batch_size: int = 16
    compute_type: str = "float16"
    diarize: bool = True
    min_speakers: int = 1
    max_speakers: int = 10


class EmotionConfig(BaseModel):
    enabled: bool = False
    backend: str = "heuristic"


class SceneConfig(BaseModel):
    detect_enabled: bool = True
    vlm_enabled: bool = False
    vlm_frame_interval_sec: float = 4.0


class TranslationConfig(BaseModel):
    backend: str = "ollama"
    fallback_backend: str | None = "transformers"
    source_lang: str = "auto"
    target_lang: str = "ru"
    model: str = "qwen2.5:14b"
    transformers_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    ollama_url: str = "http://localhost:11434"
    openai_url: str = "https://api.openai.com/v1"
    openai_api_key_env: str = "OPENAI_API_KEY"
    timing_tolerance_percent: int = 10
    max_timing_iterations: int = 3


class TTSConfig(BaseModel):
    backend: str = "qwen3-tts"
    mode: str = "custom_voice"  # custom_voice | voice_design | voice_clone
    model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    model_path: str | None = None  # local path override
    models_dir: str = "weights/tts"
    speaker: str = "Ryan"
    speaker_map: dict[str, str] = Field(default_factory=dict)
    voice_design_instruct: str = (
        "Natural adult voice for film dubbing, clear articulation, expressive."
    )
    use_flash_attn: bool = False
    x_vector_only_clone: bool = False


class MixConfig(BaseModel):
    speech_gain_db: float = 0.0
    background_gain_db: float = 0.0
    normalize: bool = True
    target_lufs: float = -16.0


class ExportConfig(BaseModel):
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 23
    preset: str = "medium"


class ProxyConfig(BaseModel):
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 2080
    scheme: str = "http"
    no_proxy: list[str] = Field(
        default_factory=lambda: ["127.0.0.1", "localhost", "::1"]
    )


class AppConfig(BaseModel):
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    separation: SeparationConfig = Field(default_factory=SeparationConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    emotion: EmotionConfig = Field(default_factory=EmotionConfig)
    scene: SceneConfig = Field(default_factory=SceneConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    mix: MixConfig = Field(default_factory=MixConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> AppConfig:
        if path is None:
            path = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
        if not path.exists():
            return cls()
        data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def merge_cli_overrides(self, **overrides: Any) -> AppConfig:
        data = self.model_dump()
        for key, value in overrides.items():
            if value is None:
                continue
            parts = key.split(".")
            target = data
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
        return AppConfig.model_validate(data)
