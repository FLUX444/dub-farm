# LLM models (translation only)

Сюда кладётся **Qwen3.6-27B** или другая LLM для vLLM / Ollama.

**Не путать с TTS** — озвучка лежит в `weights/tts/`.

## Ollama (рекомендуется для 8GB VRAM)

```bash
ollama pull qwen2.5:7b
```

## vLLM (qwen3.6-windows-server)

Укажите в `user_config.json` лаунчера:

```json
{
  "model_dir": "C:\\Users\\Kiruha\\Documents\\Projecnts\\dub-farm\\llm-models\\Qwen3.6-27B-int4-AutoRound"
}
```

Скачайте веса через лаунчер при первом запуске.
