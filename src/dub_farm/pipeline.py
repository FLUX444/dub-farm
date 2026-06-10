from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, TextColumn

from dub_farm.config import AppConfig
from dub_farm.models.schemas import PipelineState, PipelineStep
from dub_farm.utils.io import load_json, save_json
from dub_farm.worker_process import run_step_isolated
from dub_farm.workers import (
    run_emotion_analysis,
    run_export,
    run_extract_audio,
    run_mix,
    run_scene_analysis,
    run_scene_detect,
    run_separate_audio,
    run_transcribe,
    run_translate,
    run_tts,
)

logger = logging.getLogger(__name__)
console = Console()


class DubPipeline:
    def __init__(self, video_path: Path, output_path: Path, config: AppConfig):
        self.video_path = video_path.resolve()
        self.output_path = output_path.resolve()
        self.config = config
        self.work_dir = (Path(config.pipeline.work_dir) / self.video_path.stem).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.work_dir / "pipeline_state.json"

    def run(self, *, from_step: str | None = None) -> Path:
        state = self._load_state()

        steps = [
            (PipelineStep.EXTRACT, self._step_extract),
            (PipelineStep.SEPARATE, self._step_separate),
            (PipelineStep.TRANSCRIBE, self._step_transcribe),
            (PipelineStep.EMOTION, self._step_emotion),
            (PipelineStep.SCENE_DETECT, self._step_scene_detect),
            (PipelineStep.SCENE_ANALYSIS, self._step_scene_analysis),
            (PipelineStep.TRANSLATE, self._step_translate),
            (PipelineStep.TTS, self._step_tts),
            (PipelineStep.MIX, self._step_mix),
            (PipelineStep.EXPORT, self._step_export),
        ]

        skip_until_found = from_step is not None
        for step, handler in steps:
            if skip_until_found:
                if step.value != from_step:
                    continue
                skip_until_found = False

            if self.config.pipeline.resume and step.value in state.completed_steps:
                logger.info("Skipping completed step: %s", step.value)
                continue

            state.current_step = step.value
            self._save_state(state)

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"[cyan]{step.value}...", total=None)
                try:
                    if self._should_isolate_step(step):
                        logger.info("Running %s in an isolated worker process", step.value)
                        run_step_isolated(
                            step.value,
                            video_path=self.video_path,
                            output_path=self.output_path,
                            work_dir=self.work_dir,
                            config=self.config,
                        )
                    else:
                        handler()
                except Exception as exc:
                    state.error = str(exc)
                    self._save_state(state)
                    raise
                progress.update(task, description=f"[green]{step.value} done")

            state.completed_steps.append(step.value)
            state.current_step = None
            self._save_state(state)

        console.print(f"[bold green]Done![/] Output: {self.output_path}")
        return self.output_path

    def _load_state(self) -> PipelineState:
        if self.state_path.exists() and self.config.pipeline.resume:
            return load_json(self.state_path, PipelineState)
        state = PipelineState(
            input_video=str(self.video_path),
            work_dir=str(self.work_dir),
            target_lang=self.config.translation.target_lang,
        )
        self._save_state(state)
        return state

    def _save_state(self, state: PipelineState) -> None:
        save_json(self.state_path, state)

    def _should_isolate_step(self, step: PipelineStep) -> bool:
        return (
            self.config.pipeline.isolate_heavy_workers
            and step.value in set(self.config.pipeline.isolated_steps)
        )

    def _step_extract(self) -> None:
        run_extract_audio(self.video_path, self.work_dir, self.config)

    def _step_separate(self) -> None:
        audio_path = self.work_dir / "audio.wav"
        run_separate_audio(audio_path, self.work_dir, self.config)

    def _step_transcribe(self) -> None:
        speech_path = self.work_dir / "speech.wav"
        run_transcribe(speech_path, self.work_dir, self.config)

    def _step_emotion(self) -> None:
        speech_path = self.work_dir / "speech.wav"
        from dub_farm.models.schemas import TranscriptSegment

        segments = load_json(self.work_dir / "transcript.json", TranscriptSegment)
        run_emotion_analysis(speech_path, self.work_dir, self.config, segments)

    def _step_scene_detect(self) -> None:
        run_scene_detect(self.video_path, self.work_dir, self.config)

    def _step_scene_analysis(self) -> None:
        from dub_farm.models.schemas import SceneBoundary

        scenes = load_json(self.work_dir / "scenes.json", SceneBoundary)
        run_scene_analysis(self.video_path, self.work_dir, self.config, scenes)

    def _step_translate(self) -> None:
        from dub_farm.models.schemas import EmotionSegment, SceneAnalysis, TranscriptSegment

        segments = load_json(self.work_dir / "transcript.json", TranscriptSegment)
        emotions = load_json(self.work_dir / "emotions.json", EmotionSegment)
        scene_analysis = load_json(self.work_dir / "scene_analysis.json", SceneAnalysis)
        run_translate(self.work_dir, self.config, segments, emotions, scene_analysis)

    def _step_tts(self) -> None:
        from dub_farm.models.schemas import TranslationSegment

        translations = load_json(self.work_dir / "translation.json", TranslationSegment)
        run_tts(self.work_dir, self.config, translations, self.config.audio.sample_rate)

    def _step_mix(self) -> None:
        background = self.work_dir / "background.wav"
        speech = self.work_dir / "generated_speech.wav"
        run_mix(background, speech, self.work_dir, self.config)

    def _step_export(self) -> None:
        final_mix = self.work_dir / "final_mix.wav"
        run_export(self.video_path, final_mix, self.output_path, self.config)
