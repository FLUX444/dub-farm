from __future__ import annotations

import logging
from pathlib import Path

from dub_farm.config import AppConfig
from dub_farm.models.schemas import SceneBoundary
from dub_farm.utils.ffmpeg import get_duration
from dub_farm.utils.io import save_json

logger = logging.getLogger(__name__)


def run_scene_detect(
    video_path: Path, work_dir: Path, config: AppConfig
) -> list[SceneBoundary]:
    """Module 6: Detect scene boundaries."""
    scenes_path = work_dir / "scenes.json"
    if scenes_path.exists():
        from dub_farm.utils.io import load_json

        return load_json(scenes_path, SceneBoundary)

    if not config.scene.detect_enabled:
        duration = get_duration(video_path)
        scenes = [SceneBoundary(scene_id=1, start=0.0, end=duration)]
        save_json(scenes_path, scenes)
        return scenes

    logger.info("Detecting scene boundaries")
    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except ImportError as exc:
        raise ImportError(
            "PySceneDetect is required. Install with: pip install scenedetect[opencv]"
        ) from exc

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video)
    detected = scene_manager.get_scene_list()

    scenes: list[SceneBoundary] = []
    for idx, (start, end) in enumerate(detected, start=1):
        scenes.append(
            SceneBoundary(
                scene_id=idx,
                start=start.get_seconds(),
                end=end.get_seconds(),
            )
        )

    if not scenes:
        duration = get_duration(video_path)
        scenes = [SceneBoundary(scene_id=1, start=0.0, end=duration)]

    save_json(scenes_path, scenes)
    return scenes
