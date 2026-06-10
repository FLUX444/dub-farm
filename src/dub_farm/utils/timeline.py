from __future__ import annotations

from dub_farm.models.schemas import SceneBoundary, TranscriptSegment, TranslationSegment


def assign_scenes_to_segments(
    segments: list[TranscriptSegment] | list[TranslationSegment],
    scenes: list[SceneBoundary],
) -> None:
    """Assign scene_id based on segment midpoint."""
    for segment in segments:
        midpoint = (segment.start + segment.end) / 2
        for scene in scenes:
            if scene.start <= midpoint < scene.end:
                segment.scene_id = scene.scene_id
                break
