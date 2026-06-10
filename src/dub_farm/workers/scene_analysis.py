from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from dub_farm.config import AppConfig
from dub_farm.models.schemas import SceneAnalysis, SceneBoundary
from dub_farm.utils.gpu import clear_gpu_memory
from dub_farm.utils.io import save_json
from dub_farm.utils.proxy import apply_proxy_env

logger = logging.getLogger(__name__)


def run_scene_analysis(
    video_path: Path,
    work_dir: Path,
    config: AppConfig,
    scenes: list[SceneBoundary],
) -> list[SceneAnalysis]:
    """Module 5: Analyze scene context with VLM (optional)."""
    analysis_path = work_dir / "scene_analysis.json"
    if analysis_path.exists():
        from dub_farm.utils.io import load_json

        return load_json(analysis_path, SceneAnalysis)

    if not config.scene.vlm_enabled:
        logger.info("VLM scene analysis disabled — using placeholder context")
        results = [
            SceneAnalysis(
                scene_id=scene.scene_id,
                start=scene.start,
                end=scene.end,
                description="Scene context not analyzed (VLM disabled).",
            )
            for scene in scenes
        ]
        save_json(analysis_path, results)
        return results

    logger.info("Analyzing scenes with VLM")
    results = _analyze_with_vlm(video_path, scenes, config)
    save_json(analysis_path, results)

    if config.pipeline.unload_models:
        clear_gpu_memory()

    return results


def _analyze_with_vlm(
    video_path: Path,
    scenes: list[SceneBoundary],
    config: AppConfig,
) -> list[SceneAnalysis]:
    """Placeholder VLM integration — extend with Qwen-VL when GPU is available."""
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        import torch
    except ImportError as exc:
        raise ImportError(
            "Transformers + torch required for VLM. Install with: pip install dub-farm[ml]"
        ) from exc

    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    apply_proxy_env(config)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )

    results: list[SceneAnalysis] = []
    interval = config.scene.vlm_frame_interval_sec

    for scene in scenes:
        frame_path = _extract_frame(video_path, scene.start + (scene.end - scene.start) / 2)
        description = _describe_frame(model, processor, frame_path, device)
        frame_path.unlink(missing_ok=True)

        results.append(
            SceneAnalysis(
                scene_id=scene.scene_id,
                start=scene.start,
                end=scene.end,
                description=description,
                scene_type=_infer_scene_type(description),
                mood=_infer_mood(description),
            )
        )

        _ = interval  # reserved for multi-frame sampling in future versions

    del model, processor
    return results


def _extract_frame(video_path: Path, timestamp: float) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    frame_path = Path(tmp.name)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(frame_path),
        ],
        check=True,
        capture_output=True,
    )
    return frame_path


def _describe_frame(model, processor, frame_path: Path, device: str) -> str:
    from PIL import Image

    image = Image.open(frame_path).convert("RGB")
    prompt = (
        "Describe this movie scene briefly: location, time of day, "
        "what characters are doing, mood, and danger level."
    )
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    output_ids = model.generate(**inputs, max_new_tokens=128)
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def _infer_scene_type(description: str) -> str:
    lowered = description.lower()
    for keyword, scene_type in (
        ("chase", "chase"),
        ("fight", "fight"),
        ("conversation", "dialogue"),
        ("talk", "dialogue"),
        ("forest", "outdoor"),
        ("city", "urban"),
    ):
        if keyword in lowered:
            return scene_type
    return "unknown"


def _infer_mood(description: str) -> str:
    lowered = description.lower()
    for keyword, mood in (
        ("tense", "tense"),
        ("dark", "dark"),
        ("happy", "light"),
        ("calm", "calm"),
        ("danger", "tense"),
    ):
        if keyword in lowered:
            return mood
    return "neutral"
