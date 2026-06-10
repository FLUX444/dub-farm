from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from dub_farm.config import AppConfig
from dub_farm.models.schemas import TranslationSegment
from dub_farm.utils.proxy import apply_proxy_env

logger = logging.getLogger(__name__)

# Lazy singleton — loaded once per TTS worker run
_model = None
_clone_prompts: dict[str, object] = {}

LANG_MAP = {
    "ru": "Russian",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "zh": "Chinese",
}

EMOTION_INSTRUCT = {
    "fear": "Speak with fear and tension in the voice.",
    "anger": "Speak with anger and intensity.",
    "joy": "Speak cheerfully with warmth and energy.",
    "sadness": "Speak softly with sadness and melancholy.",
    "panic": "Speak in a panicked, rushed, breathless tone.",
    "sarcasm": "Speak with dry sarcasm and ironic tone.",
    "neutral": "",
}


def unload_qwen_model() -> None:
    global _model, _clone_prompts
    _model = None
    _clone_prompts = {}


def get_qwen_model(config: AppConfig):
    global _model
    if _model is not None:
        return _model

    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise ImportError(
            "qwen-tts is required. Install with:\n"
            "  py -m pip install -e \".[ml]\"\n"
            "  py -m dub_farm.cli download-models"
        ) from exc

    model_id = _resolve_model_path(config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    apply_proxy_env(config)
    load_kwargs: dict = {
        "device_map": device,
        "dtype": dtype,
    }
    if config.tts.use_flash_attn and torch.cuda.is_available():
        load_kwargs["attn_implementation"] = "flash_attention_2"

    logger.info("Loading Qwen3-TTS model: %s on %s", model_id, device)
    _model = Qwen3TTSModel.from_pretrained(model_id, **load_kwargs)
    return _model


def synthesize_segment(
    seg: TranslationSegment,
    config: AppConfig,
    *,
    speech_path: Path | None = None,
    source_segments: list | None = None,
) -> tuple[np.ndarray, int]:
    model = get_qwen_model(config)
    language = LANG_MAP.get(config.translation.target_lang, "Auto")
    instruct = _build_instruct(seg)

    mode = config.tts.mode
    if mode == "voice_clone":
        prompt = _get_clone_prompt(model, seg, config, speech_path, source_segments)
        wavs, sr = model.generate_voice_clone(
            text=seg.translated_text,
            language=language,
            voice_clone_prompt=prompt,
        )
    elif mode == "voice_design":
        wavs, sr = model.generate_voice_design(
            text=seg.translated_text,
            language=language,
            instruct=instruct or config.tts.voice_design_instruct,
        )
    else:  # custom_voice
        speaker = config.tts.speaker_map.get(seg.speaker, config.tts.speaker)
        wavs, sr = model.generate_custom_voice(
            text=seg.translated_text,
            language=language,
            speaker=speaker,
            instruct=instruct,
        )

    audio = np.asarray(wavs[0], dtype=np.float32)
    return audio, sr


MODE_DEFAULT_MODEL = {
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "voice_clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}


def _resolve_model_path(config: AppConfig) -> str:
    if config.tts.model_path:
        return config.tts.model_path

    model_id = config.tts.model
    expected = MODE_DEFAULT_MODEL.get(config.tts.mode)
    if expected and not _model_matches_mode(model_id, config.tts.mode):
        logger.warning(
            "TTS model %s does not match mode %s, using %s",
            model_id,
            config.tts.mode,
            expected,
        )
        model_id = expected

    models_dir = Path(config.tts.models_dir)
    local_name = model_id.replace("/", "--")
    local_path = models_dir / local_name
    if local_path.exists():
        return str(local_path)
    return model_id


def _model_matches_mode(model_id: str, mode: str) -> bool:
    markers = {
        "custom_voice": "CustomVoice",
        "voice_design": "VoiceDesign",
        "voice_clone": "Base",
    }
    marker = markers.get(mode, "")
    return marker in model_id


def _build_instruct(seg: TranslationSegment) -> str:
    base = EMOTION_INSTRUCT.get(seg.emotion, "")
    if seg.emotion_intensity > 0.75 and base:
        return f"{base} High emotional intensity."
    return base


def _get_clone_prompt(
    model,
    seg: TranslationSegment,
    config: AppConfig,
    speech_path: Path | None,
    source_segments: list | None,
):
    global _clone_prompts
    speaker = seg.speaker
    if speaker in _clone_prompts:
        return _clone_prompts[speaker]

    if speech_path is None or not speech_path.exists():
        raise FileNotFoundError(
            "voice_clone mode requires speech.wav for reference clips. "
            "Run the separate step first."
        )

    ref_audio, ref_text = _extract_speaker_reference(
        speech_path, speaker, source_segments or []
    )
    logger.info("Creating voice clone prompt for %s", speaker)
    prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=config.tts.x_vector_only_clone,
    )
    _clone_prompts[speaker] = prompt
    return prompt


def _extract_speaker_reference(
    speech_path: Path,
    speaker: str,
    source_segments: list,
) -> tuple[str, str]:
    """Pick a 3–15 s reference clip from the original speech track."""
    import soundfile as sf

    speaker_segs = [s for s in source_segments if getattr(s, "speaker", None) == speaker]
    if not speaker_segs:
        speaker_segs = source_segments[:3]

    ref_text_parts: list[str] = []
    start = speaker_segs[0].start
    end = speaker_segs[0].end

    for s in speaker_segs:
        if end - start > 15.0:
            break
        ref_text_parts.append(s.text)
        end = s.end

    audio, sr = sf.read(str(speech_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    start_idx = int(start * sr)
    end_idx = min(int(end * sr), len(audio))
    clip = audio[start_idx:end_idx]

    ref_path = speech_path.parent / f"ref_{speaker}.wav"
    sf.write(str(ref_path), clip, sr)

    ref_text = " ".join(ref_text_parts).strip() or "Reference speech."
    return str(ref_path), ref_text


def resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    try:
        import torch
        import torchaudio

        tensor = torch.from_numpy(audio).unsqueeze(0).float()
        resampled = torchaudio.functional.resample(tensor, src_sr, dst_sr)
        return resampled.squeeze(0).numpy()
    except Exception:
        duration = len(audio) / src_sr
        target_len = int(duration * dst_sr)
        x_old = np.linspace(0, 1, len(audio))
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, audio).astype(np.float32)
