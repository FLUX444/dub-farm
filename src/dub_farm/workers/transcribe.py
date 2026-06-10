from __future__ import annotations

import logging
import os
import platform
from pathlib import Path

from dub_farm.config import AppConfig
from dub_farm.models.schemas import TranscriptSegment
from dub_farm.utils.gpu import clear_gpu_memory
from dub_farm.utils.io import save_json
from dub_farm.utils.proxy import apply_proxy_env, httpx_proxy_for

logger = logging.getLogger(__name__)


def run_transcribe(
    speech_path: Path, work_dir: Path, config: AppConfig
) -> list[TranscriptSegment]:
    """Module 3: Transcribe speech with word-level alignment."""
    transcript_path = work_dir / "transcript.json"
    if transcript_path.exists():
        from dub_farm.utils.io import load_json

        logger.info("transcript.json already exists, loading")
        return load_json(transcript_path, TranscriptSegment)

    if config.transcription.backend == "skip":
        logger.warning("Transcription skipped — no segments produced")
        segments: list[TranscriptSegment] = []
        save_json(transcript_path, segments)
        return segments

    backend = config.transcription.backend
    if backend == "faster-whisper":
        logger.info("Transcribing with faster-whisper (%s)", config.transcription.model)
        segments = _transcribe_faster_whisper(speech_path, config)
    elif backend == "whisperx":
        logger.info("Transcribing with WhisperX (%s)", config.transcription.model)
        segments = _transcribe_whisperx(speech_path, config)
    elif backend == "vllm-openai":
        logger.info("Transcribing with vLLM OpenAI API (%s)", config.transcription.vllm_url)
        try:
            segments = _transcribe_vllm_openai(speech_path, config)
        except Exception:
            fallback = config.transcription.fallback_backend
            if not fallback or fallback == "vllm-openai":
                raise
            logger.exception(
                "vLLM transcription failed, falling back to %s", fallback
            )
            fallback_config = config.model_copy(
                update={
                    "transcription": config.transcription.model_copy(
                        update={"backend": fallback}
                    )
                }
            )
            return run_transcribe(speech_path, work_dir, fallback_config)
    elif backend == "transformers":
        logger.info("Transcribing with Transformers (%s)", config.transcription.model)
        segments = _transcribe_transformers(speech_path, config)
    else:
        raise ValueError(
            f"Unknown transcription backend: {backend!r}. "
            "Use faster-whisper, whisperx, vllm-openai, transformers, or skip."
        )

    save_json(transcript_path, segments)
    if config.pipeline.unload_models:
        clear_gpu_memory()
    return segments


def _transcribe_faster_whisper(
    speech_path: Path, config: AppConfig
) -> list[TranscriptSegment]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError(
            "faster-whisper is required for transcription. "
            "Install with: pip install dub-farm[ml]"
        ) from exc

    apply_proxy_env(config)
    device = _resolve_device()
    compute_type = config.transcription.compute_type
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"

    model = WhisperModel(
        config.transcription.model,
        device=device,
        compute_type=compute_type,
    )

    segments_iter, info = model.transcribe(
        str(speech_path),
        language=config.transcription.language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
    )

    segments: list[TranscriptSegment] = []
    for idx, seg in enumerate(segments_iter, start=1):
        if not seg.text.strip():
            continue
        words = [
            {
                "word": word.word,
                "start": word.start,
                "end": word.end,
                "score": word.probability,
            }
            for word in (seg.words or [])
        ]
        segments.append(
            TranscriptSegment(
                id=idx,
                speaker="speaker_1",
                start=float(seg.start),
                end=float(seg.end),
                text=seg.text.strip(),
                words=words,
            )
        )

    if config.transcription.diarize and segments:
        _apply_simple_speaker_labels(segments)

    logger.debug("Detected language: %s (prob %.2f)", info.language, info.language_probability)
    del model
    return segments


def _transcribe_vllm_openai(
    speech_path: Path, config: AppConfig
) -> list[TranscriptSegment]:
    if platform.system().lower() == "windows":
        logger.info(
            "Windows detected: using vLLM only through Docker/WSL2 HTTP endpoint, "
            "not local vLLM import"
        )

    try:
        import httpx
    except ImportError as exc:
        raise ImportError("httpx is required for vllm-openai transcription") from exc

    headers = {}
    if config.transcription.vllm_api_key_env:
        api_key = os.environ.get(config.transcription.vllm_api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    url = f"{config.transcription.vllm_url.rstrip('/')}/audio/transcriptions"
    data = {
        "model": config.transcription.vllm_model,
        "response_format": "verbose_json",
    }
    if config.transcription.language:
        data["language"] = config.transcription.language

    with speech_path.open("rb") as audio_file:
        files = {"file": (speech_path.name, audio_file, "audio/wav")}
        with httpx.Client(timeout=None, proxy=httpx_proxy_for(config, url)) as client:
            response = client.post(url, data=data, files=files, headers=headers)
            response.raise_for_status()
            payload = response.json()

    raw_segments = payload.get("segments") or []
    if raw_segments:
        return [
            TranscriptSegment(
                id=idx,
                speaker="speaker_1",
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=str(seg.get("text", "")).strip(),
                words=seg.get("words") or [],
            )
            for idx, seg in enumerate(raw_segments, start=1)
            if str(seg.get("text", "")).strip()
        ]

    text = str(payload.get("text", "")).strip()
    if not text:
        return []
    duration = _probe_audio_duration(speech_path)
    return [
        TranscriptSegment(
            id=1,
            speaker="speaker_1",
            start=0.0,
            end=duration,
            text=text,
            words=[],
        )
    ]


def _transcribe_transformers(
    speech_path: Path, config: AppConfig
) -> list[TranscriptSegment]:
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "Transformers + PyTorch are required for transcription fallback. "
            "Install with: pip install dub-farm[ml]"
        ) from exc

    apply_proxy_env(config)
    device = 0 if torch.cuda.is_available() else -1
    model_id = config.transcription.model
    if model_id == "large-v3":
        model_id = "openai/whisper-large-v3"

    asr = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        torch_dtype=torch.float16 if device == 0 else torch.float32,
        return_timestamps=True,
    )
    generate_kwargs = {}
    if config.transcription.language:
        generate_kwargs["language"] = config.transcription.language

    result = asr(str(speech_path), generate_kwargs=generate_kwargs)
    chunks = result.get("chunks") or []
    segments: list[TranscriptSegment] = []
    if chunks:
        for idx, chunk in enumerate(chunks, start=1):
            timestamp = chunk.get("timestamp") or (0.0, 0.0)
            start, end = timestamp
            segments.append(
                TranscriptSegment(
                    id=idx,
                    speaker="speaker_1",
                    start=float(start or 0.0),
                    end=float(end or start or 0.0),
                    text=str(chunk.get("text", "")).strip(),
                    words=[],
                )
            )
    else:
        segments.append(
            TranscriptSegment(
                id=1,
                speaker="speaker_1",
                start=0.0,
                end=_probe_audio_duration(speech_path),
                text=str(result.get("text", "")).strip(),
                words=[],
            )
        )

    del asr
    return [segment for segment in segments if segment.text]


def _transcribe_whisperx(speech_path: Path, config: AppConfig) -> list[TranscriptSegment]:
    try:
        import torch
        import whisperx
    except ImportError as exc:
        raise ImportError(
            "WhisperX is required for this backend. "
            "Install with: pip install dub-farm[ml-whisperx] "
            "(Python 3.11–3.13 only; on 3.14 use faster-whisper)"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = config.transcription.compute_type
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"

    model = whisperx.load_model(
        config.transcription.model,
        device,
        compute_type=compute_type,
    )

    audio = whisperx.load_audio(str(speech_path))
    result = model.transcribe(
        audio,
        batch_size=config.transcription.batch_size,
        language=config.transcription.language,
    )

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device,
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    if config.transcription.diarize:
        try:
            from whisperx.diarize import DiarizationPipeline

            diarize_model = DiarizationPipeline(use_auth_token=None, device=device)
            diarize_segments = diarize_model(
                audio,
                min_speakers=config.transcription.min_speakers,
                max_speakers=config.transcription.max_speakers,
            )
            result = whisperx.assign_word_speakers(diarize_segments, result)
        except Exception as exc:
            logger.warning("Diarization failed, using default speaker: %s", exc)

    segments: list[TranscriptSegment] = []
    for idx, seg in enumerate(result.get("segments", []), start=1):
        if not seg.get("text", "").strip():
            continue
        segments.append(
            TranscriptSegment(
                id=idx,
                speaker=seg.get("speaker", "speaker_1"),
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=seg["text"].strip(),
                words=seg.get("words", []),
            )
        )

    del model, model_a
    return segments


def _resolve_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _probe_audio_duration(audio_path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        return float(info.frames / info.samplerate)
    except Exception:
        return 0.0


def _apply_simple_speaker_labels(segments: list[TranscriptSegment]) -> None:
    """Heuristic speaker split when pyannote/WhisperX diarization is unavailable."""
    speaker_idx = 1
    prev_end = 0.0
    for seg in segments:
        gap = seg.start - prev_end
        if gap > 1.5:
            speaker_idx += 1
        seg.speaker = f"speaker_{speaker_idx}"
        prev_end = seg.end
