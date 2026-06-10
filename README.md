# dub-farm

Система автоматического дубляжа видео: извлечение аудио, разделение речи и фона, транскрибация, анализ эмоций и сцен, контекстный перевод, нейро-озвучка и финальный экспорт.

## Возможности

- Извлечение и нормализация аудио через FFmpeg.
- Разделение голоса и фоновой дорожки через Demucs.
- Транскрибация через `faster-whisper`, `WhisperX`, vLLM OpenAI API или Transformers fallback.
- Эвристический анализ эмоций.
- Детекция сцен через PySceneDetect и опциональный VLM-анализ.
- Контекстный перевод через Ollama или OpenAI-compatible API.
- Синтез речи через Qwen3-TTS.
- Сведение `background.wav` + `generated_speech.wav` и экспорт `dubbed_video.mp4`.

Промежуточные файлы сохраняются в `.dub-farm/<video_name>/`:

```text
audio.wav
speech.wav
background.wav
transcript.json
emotions.json
scenes.json
scene_analysis.json
translation.json
generated_speech.wav
final_mix.wav
pipeline_state.json
```

## Важно для Windows

vLLM не запускается напрямую в native Windows Python. Основной вариант:

```text
Windows -> Docker / WSL2 -> Linux-среда -> vLLM
```

Пример запуска vLLM в Docker с GPU-доступом:

```powershell
docker run --gpus all `
  -p 8000:8000 `
  --ipc=host `
  -v ./models:/models `
  vllm/vllm-openai:latest `
  --model /models/whisper-v3 `
  --gpu-memory-utilization 0.8
```

После этого dub-farm обращается к контейнеру по HTTP:

```powershell
py -m dub_farm.cli run video.mp4 --config config/vllm.yaml
```

Альтернатива: WSL2 Ubuntu + CUDA + PyTorch + vLLM. Если vLLM не запускается или нестабилен, используется fallback `Transformers + PyTorch` (`transcription.fallback_backend: transformers`).

Whisper, VLM и TTS не должны одновременно висеть в памяти. По умолчанию тяжелые этапы `separate`, `transcribe`, `scene_analysis`, `tts` запускаются отдельными процессами. После завершения процесса модель выгружается вместе с RAM/VRAM контекстом.

## Требования

- Python 3.12 рекомендуется для лучшей совместимости ML-библиотек.
- FFmpeg и ffprobe в `PATH`.
- NVIDIA GPU с CUDA рекомендуется для ML-этапов.
- Для Docker vLLM: Docker Desktop, WSL2 backend, NVIDIA Container Toolkit/GPU support.

## Установка

```powershell
cd C:\Users\Kiruha\Documents\Projecnts\dub-farm
py -m pip install -e ".[ml]"
py -m dub_farm.cli download-models --weights-dir weights
```

Или:

```powershell
.\scripts\install.ps1
```

## Быстрый запуск

```powershell
py -m dub_farm.cli run video.mp4 --target-lang ru
```

Результат: `video.dubbed.mp4`.

Полезные варианты:

```powershell
py -m dub_farm.cli run video.mp4 --from-step translate
py -m dub_farm.cli run video.mp4 --translation-backend passthrough
py -m dub_farm.cli run video.mp4 --transcription-backend vllm-openai --vllm-url http://127.0.0.1:8000/v1
py -m dub_farm.cli download-models --tts-only
py -m dub_farm.cli speakers
```

## Конфигурация

Основной файл: `config/default.yaml`.

Ключевые секции:

- `pipeline`: рабочая папка, resume, изоляция тяжелых воркеров.
- `separation`: Demucs или `skip`.
- `transcription`: `faster-whisper`, `whisperx`, `vllm-openai`, `transformers`, `skip`.
- `emotion`: эвристика эмоций.
- `scene`: PySceneDetect и опциональный VLM.
- `translation`: Ollama, OpenAI-compatible API или `passthrough`.
- `tts`: Qwen3-TTS.
- `mix`: уровни и нормализация.
- `export`: H.264/H.265/AV1 через FFmpeg.

## Архитектура

```text
Video
  -> Extract audio
  -> Separate speech/background
  -> Transcribe
  -> Emotion analysis
  -> Scene detection
  -> Scene analysis
  -> Translate and adapt timings
  -> TTS
  -> Mix
  -> Export
```

Каждый тяжелый ML-шаг может быть выполнен в отдельном Python-процессе. Это важнее простого `del model`: на Windows и CUDA память часто возвращается полностью только после завершения процесса.

## Текущие ограничения

- Качество разделения речи зависит от исходного микса.
- VLM-анализ сцен выключен по умолчанию.
- Для студийного уровня потребуется ручная настройка модели перевода, TTS-голосов и параметров сведения под конкретный материал.
