from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import httpx

from dub_farm.config import AppConfig
from dub_farm.models.schemas import (
    EmotionSegment,
    SceneAnalysis,
    TranscriptSegment,
    TranslationSegment,
)
from dub_farm.utils.io import save_json
from dub_farm.utils.proxy import apply_proxy_env, httpx_proxy_for
from dub_farm.utils.timeline import assign_scenes_to_segments

logger = logging.getLogger(__name__)

LANG_NAMES = {
    "ru": "Russian",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "ja": "Japanese",
    "zh": "Chinese",
}


def run_translate(
    work_dir: Path,
    config: AppConfig,
    segments: list[TranscriptSegment],
    emotions: list[EmotionSegment],
    scene_analysis: list[SceneAnalysis],
) -> list[TranslationSegment]:
    """Modules 7–8: Context-aware translation with timing adaptation."""
    translation_path = work_dir / "translation.json"
    if translation_path.exists():
        from dub_farm.utils.io import load_json

        logger.info("translation.json already exists, loading")
        return load_json(translation_path, TranslationSegment)

    emotion_map = {e.segment_id: e for e in emotions}
    scene_map = {s.scene_id: s for s in scene_analysis}

    assign_scenes_to_segments(segments, _scenes_from_analysis(scene_analysis))

    translations: list[TranslationSegment] = []
    for i, seg in enumerate(segments):
        prev_text = segments[i - 1].text if i > 0 else ""
        next_text = segments[i + 1].text if i < len(segments) - 1 else ""
        emotion = emotion_map.get(seg.id)
        scene = scene_map.get(seg.scene_id) if seg.scene_id else None

        translated, iterations = _translate_segment(
            seg=seg,
            prev_text=prev_text,
            next_text=next_text,
            emotion=emotion,
            scene=scene,
            config=config,
        )

        translations.append(
            TranslationSegment(
                id=seg.id,
                speaker=seg.speaker,
                start=seg.start,
                end=seg.end,
                source_text=seg.text,
                translated_text=translated,
                emotion=emotion.emotion if emotion else "neutral",
                emotion_intensity=emotion.intensity if emotion else 0.5,
                speech_rate=emotion.speech_rate if emotion else 1.0,
                scene_id=seg.scene_id,
                scene_context=scene.model_dump() if scene else {},
                timing_iterations=iterations,
            )
        )

    save_json(translation_path, translations)
    return translations


def _scenes_from_analysis(scene_analysis: list[SceneAnalysis]):
    from dub_farm.models.schemas import SceneBoundary

    return [
        SceneBoundary(scene_id=s.scene_id, start=s.start, end=s.end)
        for s in scene_analysis
    ]


def _translate_segment(
    *,
    seg: TranscriptSegment,
    prev_text: str,
    next_text: str,
    emotion: EmotionSegment | None,
    scene: SceneAnalysis | None,
    config: AppConfig,
) -> tuple[str, int]:
    target_lang = LANG_NAMES.get(config.translation.target_lang, config.translation.target_lang)
    duration = max(seg.end - seg.start, 0.1)
    target_words = _estimate_word_count(duration, config.translation.target_lang)

    system_prompt = (
        f"You are a professional dubbing translator. Translate dialogue into {target_lang} "
        "for lip-sync dubbing. Write natural spoken dialogue, not literal translation. "
        f"Keep the line speakable in about {target_words} words (±{config.translation.timing_tolerance_percent}%). "
        "Preserve emotion, character voice, and scene context."
    )

    user_prompt = _build_user_prompt(seg, prev_text, next_text, emotion, scene)

    if config.translation.backend == "passthrough":
        return seg.text, 0

    translated = _call_llm(system_prompt, user_prompt, config)

    iterations = 0
    for _ in range(config.translation.max_timing_iterations):
        word_count = len(translated.split())
        deviation = abs(word_count - target_words) / max(target_words, 1) * 100
        if deviation <= config.translation.timing_tolerance_percent:
            break
        iterations += 1
        direction = "shorter" if word_count > target_words else "longer"
        refine_prompt = (
            f"The translation has {word_count} words but needs ~{target_words}. "
            f"Rewrite it to be {direction} while keeping meaning and natural speech:\n{translated}"
        )
        translated = _call_llm(system_prompt, refine_prompt, config)

    return translated.strip(), iterations


def _build_user_prompt(
    seg: TranscriptSegment,
    prev_text: str,
    next_text: str,
    emotion: EmotionSegment | None,
    scene: SceneAnalysis | None,
) -> str:
    parts = [f"Line: {seg.text}"]
    if prev_text:
        parts.append(f"Previous line: {prev_text}")
    if next_text:
        parts.append(f"Next line: {next_text}")
    if emotion:
        parts.append(
            f"Emotion: {emotion.emotion} (intensity {emotion.intensity}, rate {emotion.speech_rate})"
        )
    if scene:
        parts.append(
            f"Scene: {scene.scene_type}, {scene.location}, {scene.time_of_day}, "
            f"mood={scene.mood}, danger={scene.danger_level}. {scene.description}"
        )
    parts.append("Return only the translated line, no quotes or explanation.")
    return "\n".join(parts)


def _estimate_word_count(duration_sec: float, lang: str) -> int:
    # Average speaking rate varies by language
    wps = {"ru": 2.5, "en": 2.8, "de": 2.3, "ja": 4.0, "zh": 3.5}.get(lang, 2.5)
    return max(1, round(duration_sec * wps))


def _call_llm(system_prompt: str, user_prompt: str, config: AppConfig) -> str:
    backend = config.translation.backend
    try:
        return _call_with_backend(backend, system_prompt, user_prompt, config)
    except Exception:
        fallback = config.translation.fallback_backend
        if fallback and fallback != backend:
            logger.exception(
                "Translation backend %s failed, falling back to %s",
                backend,
                fallback,
            )
            return _call_with_backend(fallback, system_prompt, user_prompt, config)
        raise


def _call_with_backend(
    backend: str, system_prompt: str, user_prompt: str, config: AppConfig
) -> str:
    if backend == "ollama":
        return _call_ollama(system_prompt, user_prompt, config)
    if backend == "openai":
        return _call_openai(system_prompt, user_prompt, config)
    if backend == "transformers":
        return _call_transformers(system_prompt, user_prompt, config)
    raise ValueError(f"Unknown translation backend: {backend}")


def _call_ollama(system_prompt: str, user_prompt: str, config: AppConfig) -> str:
    url = f"{config.translation.ollama_url.rstrip('/')}/api/chat"
    payload = {
        "model": config.translation.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    with httpx.Client(timeout=120.0, proxy=httpx_proxy_for(config, url)) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        content = response.json()["message"]["content"]
    return _clean_llm_output(content)


def _call_openai(system_prompt: str, user_prompt: str, config: AppConfig) -> str:
    api_key = os.environ.get(config.translation.openai_api_key_env)
    url = f"{config.translation.openai_url.rstrip('/')}/chat/completions"
    local_endpoint = "127.0.0.1" in url or "localhost" in url
    if not api_key and not local_endpoint:
        raise EnvironmentError(
            f"Set {config.translation.openai_api_key_env} for OpenAI translation backend"
        )
    headers = {"Authorization": f"Bearer {api_key or 'local-vllm'}"}
    payload = {
        "model": config.translation.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    with httpx.Client(timeout=120.0, proxy=httpx_proxy_for(config, url)) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    return _clean_llm_output(content)


def _clean_llm_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^["\'«»]|["\'«»]$', "", text)
    return text.strip()


_transformers_model = None
_transformers_tokenizer = None


def _call_transformers(system_prompt: str, user_prompt: str, config: AppConfig) -> str:
    global _transformers_model, _transformers_tokenizer
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Transformers + PyTorch are required for translation fallback. "
            "Install with: pip install dub-farm[ml]"
        ) from exc

    model_id = config.translation.transformers_model
    if _transformers_model is None:
        apply_proxy_env(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        logger.info("Loading translation model %s on %s", model_id, device)
        _transformers_tokenizer = AutoTokenizer.from_pretrained(model_id)
        _transformers_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            _transformers_model = _transformers_model.to(device)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(_transformers_tokenizer, "apply_chat_template"):
        prompt = _transformers_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"

    inputs = _transformers_tokenizer(prompt, return_tensors="pt")
    device = next(_transformers_model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = _transformers_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    generated = output_ids[0, inputs["input_ids"].shape[-1] :]
    content = _transformers_tokenizer.decode(generated, skip_special_tokens=True)
    return _clean_llm_output(content)


def unload_transformers_model() -> None:
    global _transformers_model, _transformers_tokenizer
    _transformers_model = None
    _transformers_tokenizer = None
